"""Observatorio electoral — Entrega 3: preparación de los datos.

La entrega se presenta como un informe de auditoría de calidad de datos para las tres
fuentes de la Registraduría (preconteo, escrutinio y novedades):

- Un motor de reglas declarativas evalúa cada archivo en vivo (completitud, integridad
  física, validez, unicidad, consistencia, cobertura, integridad referencial y plausibilidad).
- Un registro de hallazgos documenta cada problema con evidencia, decisión y estado.
- El dictamen ejecutivo se genera a partir del resultado de las reglas.
- La vista de linaje registra la huella del archivo y las transformaciones.
- El informe completo se puede descargar en Markdown y CSV.

El escrutinio se cruza con el preconteo y las novedades para construir la variable objetivo
del proyecto (`y_corregida`); novedades reproduce el diagnóstico y la limpieza del notebook.
"""
import fnmatch
import hashlib
import io
import os
import platform
import re
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from services.data_loader import PRECONTEO_PATH, load_preconteo, load_resumen_mesas

# ============================================================
# CONSTANTES
# ============================================================

CLAVE_MESA = ["departamento", "municipio", "zona", "puesto", "mesa"]
CLAVE_REGISTRO = CLAVE_MESA + [
    "jal", "comunicado", "circunscripcion", "lista_partido", "candidato",
]

# Esquema oficial del registro de preconteo (Estructuras Básicas, RNEC).
CAMPOS = [
    ("departamento", 2, "numérico"),
    ("municipio", 3, "numérico"),
    ("zona", 2, "alfanumérico"),
    ("puesto", 2, "alfanumérico"),
    ("mesa", 6, "numérico"),
    ("jal", 2, "numérico"),
    ("comunicado", 4, "numérico"),
    ("circunscripcion", 1, "numérico"),
    ("lista_partido", 5, "numérico"),
    ("candidato", 3, "numérico"),
    ("votos", 8, "numérico"),
]
LONGITUD_REGISTRO = sum(longitud for _, longitud, _ in CAMPOS)


def _posiciones_alfanumericas():
    posiciones, inicio = set(), 0
    for _, longitud, tipo in CAMPOS:
        if tipo == "alfanumérico":
            posiciones.update(range(inicio, inicio + longitud))
        inicio += longitud
    return frozenset(posiciones)


POSICIONES_ALFANUMERICAS = _posiciones_alfanumericas()

OPCIONES_ESPECIALES = {
    "996": "Voto en blanco",
    "997": "Votos nulos",
    "998": "Votos no marcados",
}

FUENTES = ["Preconteo", "Escrutinio", "Novedades"]

VISTAS = [
    "Resumen ejecutivo",
    "Reglas de calidad",
    "Registro de hallazgos",
    "Preguntas de la plantilla",
    "Linaje y reproducibilidad",
]

SECCIONES = [
    "1.1 · Valores faltantes",
    "1.2 · Duplicados",
    "1.3 · Valores atípicos",
    "1.4 · Inconsistencias",
    "1.5 · Datos corruptos",
    "1.6 · Selección de datos",
]

ORDEN_SEVERIDAD = {"Crítica": 0, "Alta": 1, "Media": 2, "Baja": 3, "Informativa": 4}

# Preguntas de la plantilla del curso.
PREGUNTAS = {
    "1.1": [
        "¿Cómo afectan estos valores a la integridad y representatividad de la información?",
        "¿Cómo se identificaron los valores faltantes en el conjunto de datos?",
        "¿Cuáles fueron los criterios para decidir si rellenar o eliminar? ¿Qué método se usó y por qué?",
        "¿Se analizó la distribución de los faltantes en relación con las variables clave?",
    ],
    "1.2": [
        "¿Cómo se identificaron y manejaron los valores duplicados?",
        "¿Hubo consideraciones especiales según ciertas variables?",
        "¿Pueden estos duplicados sesgar el análisis, y en qué medida?",
        "¿Se justifica su eliminación para mantener la coherencia de los datos?",
        "¿Cómo se aseguró que la eliminación no afectara la representación del conjunto?",
    ],
    "1.3": [
        "¿Son errores de medición o información válida pero excepcional?",
        "¿Qué criterios o técnicas se utilizaron para identificarlos?",
        "¿Se aplicaron métodos estadísticos o visuales?",
        "¿Cuál fue la decisión final sobre su tratamiento y por qué?",
    ],
    "1.4": [
        "¿Cómo se identificaron las inconsistencias?",
        "¿Son lo suficientemente significativas para justificar una estandarización?",
        "¿Cómo se homogeneizaron unidades y formatos?",
        "¿Qué estrategias se implementaron para asegurar la coherencia?",
        "¿Se establecieron reglas de validación específicas?",
    ],
    "1.5": [
        "¿Cómo se determinó la presencia de datos corruptos?",
        "¿Qué medidas se tomaron para corregirlos o eliminarlos?",
        "¿Hubo casos que no pudieron manejarse y cómo se justificó?",
    ],
    "1.6": [
        "¿Qué criterios se utilizaron para seleccionar los datos relevantes?",
        "¿Se aplicaron técnicas de muestreo o filtrado?",
        "¿Cómo se justificó la inclusión o exclusión de variables?",
    ],
}


# ============================================================
# UTILIDADES DE PRESENTACIÓN
# ============================================================

def numero(n):
    """Entero con punto como separador de miles."""
    return f"{int(round(n)):,}".replace(",", ".")


def porcentaje(x, decimales=1):
    """Porcentaje con coma decimal."""
    return f"{x:.{decimales}f} %".replace(".", ",")


def explicacion(texto, titulo="Metodología, respuestas y justificación académica"):
    with st.expander(titulo, expanded=True):
        st.markdown(texto)


def grafico(fig, alto=410):
    fig.update_layout(margin=dict(l=15, r=15, t=65, b=20), height=alto)
    st.plotly_chart(fig, use_container_width=True)


def etiqueta_opcion(codigo):
    return f"{codigo} · {OPCIONES_ESPECIALES[codigo]}" if codigo in OPCIONES_ESPECIALES else codigo


# ============================================================
# CÁLCULOS DEL PRECONTEO (en caché)
# ============================================================
# Los argumentos con guion bajo no forman parte de la clave de caché: los
# resultados se recalculan al reiniciar la aplicación o limpiar la caché.

@st.cache_data(show_spinner=False)
def perfil(_df, _mesas):
    faltantes = {}
    for c in _df.columns:
        s = _df[c]
        nulos = int(s.isna().sum())
        vacios = (
            int(s.str.strip().eq("").fillna(False).sum())
            if pd.api.types.is_string_dtype(s) else 0
        )
        faltantes[c] = nulos + vacios

    opciones = _df[["lista_partido", "candidato"]].drop_duplicates()
    listas_por_candidato = opciones.groupby("candidato")["lista_partido"].nunique()

    return {
        "faltantes": faltantes,
        "duplicados_exactos": int(_df.duplicated().sum()),
        "duplicados_logicos": int(_df.duplicated(subset=CLAVE_REGISTRO).sum()),
        "ceros": int(_df["votos"].eq(0).sum()),
        "positivos": int(_df["votos"].gt(0).sum()),
        "negativos": int(_df["votos"].lt(0).sum()),
        "votos_totales": int(_df["votos"].sum()),
        "opciones": len(opciones),
        "candidatos_multilista": int(listas_por_candidato.gt(1).sum()),
        "mesas_menores_16": int(_mesas["registros"].lt(16).sum()),
        "mesas_mayores_16": int(_mesas["registros"].gt(16).sum()),
        "mesas_suma_cero": int(_mesas["total_votos"].eq(0).sum()),
        "comunicado_distinto": int(_df["comunicado"].ne("9999").sum()),
        "percentiles": _df["votos"].quantile([.25, .5, .75, .95, .99]).to_dict(),
        "constantes": [c for c in _df.columns if _df[c].nunique(dropna=False) == 1],
        "cardinalidad": {c: int(_df[c].nunique()) for c in _df.columns},
        "puestos": int(_df[["departamento", "municipio", "zona", "puesto"]].drop_duplicates().shape[0]),
    }


@st.cache_data(show_spinner="Verificando la estructura física del archivo...")
def verificar_estructura():
    """Lee el archivo original línea por línea, sin pasar por pandas."""
    longitudes, vacias, con_letras, letras_fuera = {}, 0, 0, 0
    with PRECONTEO_PATH.open("rb") as archivo:
        for linea in archivo:
            registro = linea.rstrip(b"\r\n")
            longitudes[len(registro)] = longitudes.get(len(registro), 0) + 1
            if not registro:
                vacias += 1
                continue
            if not registro.isdigit():
                con_letras += 1
                letras_fuera += any(
                    not 48 <= byte <= 57 and i not in POSICIONES_ALFANUMERICAS
                    for i, byte in enumerate(registro)
                )
    total = sum(longitudes.values())
    return {
        "total": total,
        "longitud_correcta": longitudes.get(LONGITUD_REGISTRO, 0),
        "longitud_incorrecta": total - longitudes.get(LONGITUD_REGISTRO, 0),
        "vacias": vacias,
        "con_letras": con_letras,
        "letras_fuera": letras_fuera,
    }


@st.cache_data(show_spinner="Calculando la huella del archivo...")
def huella_archivo():
    """SHA-256, tamaño y fecha de modificación del archivo original."""
    sha = hashlib.sha256()
    with PRECONTEO_PATH.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            sha.update(bloque)
    info = PRECONTEO_PATH.stat()
    return {
        "sha256": sha.hexdigest(),
        "bytes": info.st_size,
        "modificado": datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "nombre": PRECONTEO_PATH.name,
    }


@st.cache_data(show_spinner=False)
def estadisticas_mesas(_mesas):
    votos = _mesas["total_votos"]
    q1, q3 = votos.quantile([.25, .75])
    bajo, alto = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    consulado = _mesas["departamento"].eq("88")
    debajo, encima = votos.lt(bajo), votos.gt(alto)
    return {
        "bajo": float(bajo),
        "alto": float(alto),
        "fuera": int((debajo | encima).sum()),
        "debajo": int(debajo.sum()),
        "encima": int(encima.sum()),
        "encima_consulado": int((encima & consulado).sum()),
        "debajo_consulado": int((debajo & consulado).sum()),
        "maximo": int(votos.max()),
        "top10_consulado": int(_mesas.nlargest(10, "total_votos")["departamento"].eq("88").sum()),
    }


@st.cache_data(show_spinner=False)
def formato_por_departamento(_mesas):
    """Porcentaje de mesas con las 16 filas en cada departamento."""
    return (
        _mesas.assign(completa=_mesas["registros"].eq(16))
        .groupby("departamento")
        .agg(mesas=("completa", "size"), completas=("completa", "mean"))
        .assign(completas=lambda t: t["completas"] * 100)
        .sort_values("completas")
        .reset_index()
        .rename(columns={
            "departamento": "Departamento (código)",
            "mesas": "Mesas",
            "completas": "Mesas con 16 filas (%)",
        })
    )


@st.cache_data(show_spinner=False)
def ceros_explicitos_por_formato(_mesas):
    """Porcentaje de mesas con al menos un cero escrito, según su formato."""
    return (
        _mesas.assign(
            formato=np.where(_mesas["registros"].eq(16), "16 filas", "Menos de 16"),
            tiene_cero=_mesas["registros_cero"].gt(0),
        )
        .groupby("formato")["tiene_cero"].mean().mul(100)
        .to_dict()
    )


@st.cache_data(show_spinner=False)
def ausencia_por_departamento_y_opcion(_df, _mesas):
    """Porcentaje de mesas sin fila para cada opción, por departamento."""
    mesas_dep = _mesas.groupby("departamento").size()
    presentes = _df.groupby(["departamento", "candidato"]).size().unstack(fill_value=0)
    ausencia = 100 - presentes.div(mesas_dep, axis=0) * 100
    orden_dep = mesas_dep.sort_values(ascending=False).index
    orden_opc = ausencia.mean().sort_values(ascending=False).index
    ausencia = ausencia.loc[orden_dep, orden_opc]
    ausencia.columns = [etiqueta_opcion(c) for c in ausencia.columns]
    return ausencia.round(1)


@st.cache_data(show_spinner=False)
def percentiles_por_opcion(_df):
    tabla = (
        _df.groupby("candidato")["votos"]
        .describe(percentiles=[.5, .95, .99])
        .rename(columns={"count": "registros", "mean": "media", "50%": "p50",
                         "95%": "p95", "99%": "p99", "max": "máximo"})
        [["registros", "media", "p50", "p95", "p99", "máximo"]]
    )
    tabla.index = [etiqueta_opcion(c) for c in tabla.index]
    return tabla.round(1).rename_axis("Opción").reset_index()


# ============================================================
# MOTOR DE REGLAS DE CALIDAD
# ============================================================

@dataclass
class Regla:
    id: str
    dimension: str
    regla: str
    severidad: str
    esperado: str
    observado: str
    estado: str          # "Cumple", "No cumple" o "Informativa"
    hallazgo: str = ""   # ID del hallazgo relacionado, si existe


def _regla(id_, dimension, texto, severidad, observado, cumple, esperado="0", hallazgo=""):
    estado = "Informativa" if cumple is None else ("Cumple" if cumple else "No cumple")
    return Regla(id_, dimension, texto, severidad, esperado, numero(observado), estado, hallazgo)


def evaluar_reglas(p, e, m):
    """Evalúa todas las reglas sobre el archivo cargado."""
    faltantes = sum(p["faltantes"].values())
    return [
        _regla("R-01", "Completitud", "Sin nulos ni cadenas vacías en las 11 variables",
               "Crítica", faltantes, faltantes == 0),
        _regla("R-02", "Integridad física", f"Todas las líneas tienen {LONGITUD_REGISTRO} caracteres",
               "Crítica", e["longitud_incorrecta"], e["longitud_incorrecta"] == 0),
        _regla("R-03", "Integridad física", "Sin líneas vacías",
               "Alta", e["vacias"], e["vacias"] == 0),
        _regla("R-04", "Validez", "Caracteres no numéricos solo en campos alfanuméricos (zona, puesto)",
               "Alta", e["letras_fuera"], e["letras_fuera"] == 0),
        _regla("R-05", "Validez", "El campo comunicado vale 9999 en todos los registros",
               "Media", p["comunicado_distinto"], p["comunicado_distinto"] == 0),
        _regla("R-06", "Validez", "Votos no negativos",
               "Crítica", p["negativos"], p["negativos"] == 0),
        _regla("R-07", "Unicidad", "Sin registros idénticos",
               "Crítica", p["duplicados_exactos"], p["duplicados_exactos"] == 0),
        _regla("R-08", "Unicidad", "Clave de 10 campos territoriales y electorales única",
               "Crítica", p["duplicados_logicos"], p["duplicados_logicos"] == 0),
        _regla("R-09", "Consistencia", "Cada código de candidato pertenece a una sola lista",
               "Alta", p["candidatos_multilista"], p["candidatos_multilista"] == 0),
        _regla("R-10", "Consistencia", "Ninguna mesa supera 16 opciones",
               "Alta", p["mesas_mayores_16"], p["mesas_mayores_16"] == 0),
        _regla("R-11", "Cobertura", "Todas las mesas transmiten las 16 opciones",
               "Media", p["mesas_menores_16"], p["mesas_menores_16"] == 0, hallazgo="H-01"),
        _regla("R-12", "Plausibilidad", "Ninguna mesa con suma total de cero votos",
               "Media", p["mesas_suma_cero"], p["mesas_suma_cero"] == 0, hallazgo="H-02"),
        _regla("R-13", "Plausibilidad", "Mesas fuera del intervalo 1,5 × IQR de votos por mesa",
               "Informativa", m["fuera"], None, esperado="—", hallazgo="H-03"),
        _regla("R-14", "Utilidad", "Variables con un único valor en todo el archivo",
               "Informativa", len(p["constantes"]), None, esperado="—", hallazgo="H-04"),
    ]


def tabla_reglas(reglas):
    return pd.DataFrame([asdict(r) for r in reglas]).rename(columns={
        "id": "ID", "dimension": "Dimensión", "regla": "Regla", "severidad": "Severidad",
        "esperado": "Esperado", "observado": "Observado", "estado": "Estado",
        "hallazgo": "Hallazgo",
    })


def resumen_por_dimension(reglas):
    filas = {}
    for r in reglas:
        if r.estado == "Informativa":
            continue
        d = filas.setdefault(r.dimension, {"Reglas": 0, "Cumplen": 0})
        d["Reglas"] += 1
        d["Cumplen"] += r.estado == "Cumple"
    t = pd.DataFrame.from_dict(filas, orient="index").rename_axis("Dimensión").reset_index()
    t["Estado"] = np.where(t["Reglas"] == t["Cumplen"], "Conforme", "Con observaciones")
    return t


def dictamen(reglas, observaciones=None):
    verificables = [r for r in reglas if r.estado != "Informativa"]
    cumplen = sum(r.estado == "Cumple" for r in verificables)
    criticas = [r for r in verificables if r.severidad == "Crítica"]
    criticas_fallidas = [r for r in criticas if r.estado == "No cumple"]
    otras_fallidas = [r for r in verificables if r.estado == "No cumple" and r.severidad != "Crítica"]

    texto = f"El archivo cumple **{cumplen} de {len(verificables)} reglas verificables**. "
    if criticas_fallidas:
        texto += (
            "**Hay reglas críticas incumplidas** ("
            + ", ".join(r.id for r in criticas_fallidas)
            + "), por lo que el archivo no debe usarse sin corregirlas primero."
        )
        return "No apto sin correcciones", texto
    texto += (
        f"Las {len(criticas)} reglas críticas (integridad física, validez de votos, unicidad y "
        "completitud) se cumplen en su totalidad. "
    )
    if otras_fallidas:
        ids = ", ".join(r.id for r in otras_fallidas)
        if observaciones is None:
            texto += (
                "Las observaciones (" + ids + ") son de "
                "cobertura y plausibilidad; están documentadas en el registro de hallazgos con su "
                "tratamiento."
            )
        else:
            texto += observaciones.format(ids=ids)
        return "Apto con observaciones", texto
    return "Apto", texto


# ============================================================
# REGISTRO DE HALLAZGOS
# ============================================================

def registrar_hallazgos(p, m, n_mesas, ceros_implicitos):
    return [
        {
            "ID": "H-01",
            "Hallazgo": "Dos formatos de transmisión según el departamento",
            "Severidad": "Alta",
            "Evidencia": (
                f"{numero(p['mesas_menores_16'])} mesas ({porcentaje(100 * p['mesas_menores_16'] / n_mesas)}) "
                "tienen menos de 16 filas. Los departamentos forman dos grupos casi puros, y las "
                "mesas de 16 filas casi siempre traen ceros explícitos, mientras que las demás casi nunca."
            ),
            "Impacto": (
                "Conteos de filas, proporciones de ceros y promedios por opción mezclan dos "
                "representaciones del mismo fenómeno; el número de filas por mesa separa "
                "departamentos y no resultados."
            ),
            "Decisión": (
                f"Estandarizar a 16 opciones por mesa completando {numero(ceros_implicitos)} ceros "
                "implícitos, con la marca `cero_implicito` y la variable `formato`. El archivo "
                "original no se modifica."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-02",
            "Hallazgo": "Mesas con suma total de cero votos",
            "Severidad": "Media",
            "Evidencia": (
                f"{numero(p['mesas_suma_cero'])} mesas suman cero votos y tienen una sola opción "
                "registrada. El formato que omite las opciones sin votos no explica este caso."
            ),
            "Impacto": "Podrían ser mesas no instaladas, sin votantes o no transmitidas.",
            "Decisión": "Conservar y marcar para verificación con escrutinio y novedades.",
            "Estado": "Abierto",
        },
        {
            "ID": "H-03",
            "Hallazgo": "Valores extremos de votos por mesa concentrados en consulados",
            "Severidad": "Baja",
            "Evidencia": (
                f"{numero(m['fuera'])} mesas fuera del intervalo [{numero(m['bajo'])}, {numero(m['alto'])}]; "
                f"{numero(m['encima_consulado'])} de las {numero(m['encima'])} superiores son consulados, "
                f"incluidas {m['top10_consulado']} de las 10 mesas con más votos (máximo {numero(m['maximo'])})."
            ),
            "Impacto": "Un umbral global marcaría como anómalas mesas legítimas del exterior.",
            "Decisión": "Conservar; en la detección de anomalías comparar dentro de grupos territoriales.",
            "Estado": "Cerrado",
        },
        {
            "ID": "H-04",
            "Hallazgo": "Variables sin variación",
            "Severidad": "Baja",
            "Evidencia": "Constantes: " + ", ".join(f"`{c}`" for c in p["constantes"]) + ".",
            "Impacto": "No aportan información a ningún modelo.",
            "Decisión": "Conservar en el original para trazabilidad y excluir de la tabla de trabajo.",
            "Estado": "Mitigado",
        },
        {
            "ID": "H-05",
            "Hallazgo": "Riesgo de fuga de información por el formato de transmisión",
            "Severidad": "Media",
            "Evidencia": "El número de filas por mesa depende del departamento (ver H-01).",
            "Impacto": (
                "Un modelo podría aprender el formato del departamento en lugar del "
                "comportamiento electoral."
            ),
            "Decisión": "No usar el número de filas por mesa como predictor; `formato` solo como control.",
            "Estado": "Mitigado",
        },
        {
            "ID": "H-06",
            "Hallazgo": "Códigos especiales 996, 997 y 998",
            "Severidad": "Informativa",
            "Evidencia": "Se interpretan como voto en blanco, votos nulos y votos no marcados.",
            "Impacto": "Mezclarlos con candidaturas distorsiona los análisis por candidato.",
            "Decisión": "Tratarlos como opciones separadas; confirmar con la documentación oficial.",
            "Estado": "Abierto",
        },
    ]


# ============================================================
# INFORME DESCARGABLE
# ============================================================

def _celda(valor):
    return str(valor).replace("|", "\\|").replace("\n", " ")


def _tabla_md(df):
    columnas = list(df.columns)
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    lineas += ["| " + " | ".join(_celda(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(lineas)


def informe_markdown(huella, veredicto, texto_dictamen, reglas, hallazgos, n_reg, n_mesas, votos):
    return f"""# Informe de auditoría de calidad de datos: preconteo presidencial 2026

Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Archivo auditado

| Campo | Valor |
|---|---|
| Archivo | {huella['nombre']} |
| SHA-256 | `{huella['sha256']}` |
| Tamaño | {numero(huella['bytes'])} bytes |
| Última modificación | {huella['modificado']} |
| Registros | {numero(n_reg)} |
| Mesas | {numero(n_mesas)} |
| Votos registrados | {numero(votos)} |

## Dictamen: {veredicto}

{texto_dictamen}

## Reglas de calidad

{_tabla_md(tabla_reglas(reglas))}

## Registro de hallazgos

{_tabla_md(pd.DataFrame(hallazgos))}

## Alcance y limitaciones

Este informe evalúa la calidad técnica del archivo de preconteo. No certifica resultados
electorales, no contrasta los valores con las actas E-14 y no atribuye irregularidades a
mesas o territorios. Una regla cumplida indica conformidad con el criterio evaluado, no
exactitud del dato frente a la realidad.
"""


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

def render():
    st.header("Entrega 3 · Preparación de los datos")
    st.caption("CRISP-DM · Elecciones presidenciales de Colombia 2026, primera vuelta")
    st.markdown(
        "Informe de auditoría de **calidad de datos** de los archivos de la Registraduría. "
        "Evalúa si los datos son aptos para el análisis; no certifica resultados ni atribuye "
        "irregularidades a mesas o territorios."
    )

    fuente = st.radio("Fuente de datos", FUENTES, horizontal=True, key="e3_fuente")
    st.divider()

    if fuente == "Preconteo":
        render_preconteo()
    elif fuente == "Escrutinio":
        render_escrutinio()
    else:
        render_novedades()


# ============================================================
# PRECONTEO
# ============================================================

def render_preconteo():
    if not st.session_state.get("e3_activa", False):
        st.markdown(
            "Cada fila del preconteo corresponde a una opción electoral registrada en una mesa. "
            "Los indicadores se recalculan sobre el archivo local; las interpretaciones provienen "
            "del notebook de preparación."
        )
        if st.button("Ejecutar auditoría del preconteo", type="primary"):
            st.session_state.e3_activa = True
            st.rerun()
        st.info("El archivo se lee solo cuando lo solicitas, para no bloquear las demás entregas.")
        return

    with st.spinner("Leyendo el archivo y evaluando las reglas..."):
        df = load_preconteo()
        mesas = load_resumen_mesas()
        p = perfil(df, mesas)
        e = verificar_estructura()
        m = estadisticas_mesas(mesas)

    n_mesas, n_reg = len(mesas), len(df)
    n_filas_estandar = n_mesas * p["opciones"]
    ceros_implicitos = n_filas_estandar - n_reg

    reglas = evaluar_reglas(p, e, m)
    hallazgos = registrar_hallazgos(p, m, n_mesas, ceros_implicitos)
    veredicto, texto_dictamen = dictamen(reglas)

    a, b, c, d = st.columns(4)
    a.metric("Registros", numero(n_reg))
    b.metric("Mesas", numero(n_mesas))
    c.metric("Opciones electorales", numero(p["opciones"]))
    d.metric("Votos registrados", numero(p["votos_totales"]))

    vista = st.radio("Vista", VISTAS, horizontal=True, key="e3_vista")
    st.divider()

    if vista == "Resumen ejecutivo":
        vista_resumen(df, mesas, p, reglas, hallazgos, veredicto, texto_dictamen,
                      n_reg, n_mesas, ceros_implicitos)
    elif vista == "Reglas de calidad":
        vista_reglas(reglas)
    elif vista == "Registro de hallazgos":
        vista_hallazgos(hallazgos)
    elif vista == "Preguntas de la plantilla":
        vista_secciones(df, mesas, p, e, m, n_reg, n_mesas, n_filas_estandar, ceros_implicitos)
    else:
        vista_linaje(p, reglas, n_reg, n_mesas, n_filas_estandar, ceros_implicitos)

    st.divider()
    st.caption(
        "Resultados metodológicos del notebook del equipo; indicadores recalculados sobre el "
        "archivo local. La detección estadística no constituye una auditoría electoral."
    )


# ------------------------------------------------------------
# Vistas
# ------------------------------------------------------------

def vista_resumen(df, mesas, p, reglas, hallazgos, veredicto, texto_dictamen,
                  n_reg, n_mesas, ceros_implicitos):
    st.subheader(f"Dictamen: {veredicto}")
    st.markdown(texto_dictamen)

    verificables = [r for r in reglas if r.estado != "Informativa"]
    abiertos = [h for h in hallazgos if h["Estado"] == "Abierto"]
    a, b, c, d = st.columns(4)
    a.metric("Reglas cumplidas", f"{sum(r.estado == 'Cumple' for r in verificables)} de {len(verificables)}")
    b.metric("Hallazgos registrados", len(hallazgos))
    c.metric("Hallazgos abiertos", len(abiertos))
    d.metric("Ceros implícitos completados", numero(ceros_implicitos))

    izquierda, derecha = st.columns([2, 3])
    with izquierda:
        st.markdown("**Conformidad por dimensión**")
        st.dataframe(resumen_por_dimension(reglas), hide_index=True, use_container_width=True)
        st.markdown("**Hallazgos abiertos**")
        for h in abiertos:
            st.markdown(f"- **{h['ID']} · {h['Hallazgo']}.** {h['Decisión']}")
    with derecha:
        formato = formato_por_departamento(mesas)
        grafico(px.bar(formato, x="Departamento (código)", y="Mesas con 16 filas (%)",
                       range_y=[0, 101], hover_data=["Mesas"],
                       title="Hallazgo principal: formato de transmisión por departamento"), alto=380)

    st.markdown("**Descargar el informe**")
    huella = huella_archivo()
    informe = informe_markdown(huella, veredicto, texto_dictamen, reglas, hallazgos,
                               n_reg, n_mesas, p["votos_totales"])
    x, y, z = st.columns(3)
    x.download_button("Informe completo (Markdown)", informe,
                      file_name="informe_auditoria_preconteo.md", mime="text/markdown")
    y.download_button("Reglas de calidad (CSV)",
                      tabla_reglas(reglas).to_csv(index=False).encode("utf-8-sig"),
                      file_name="reglas_calidad_preconteo.csv", mime="text/csv")
    z.download_button("Registro de hallazgos (CSV)",
                      pd.DataFrame(hallazgos).to_csv(index=False).encode("utf-8-sig"),
                      file_name="hallazgos_preconteo.csv", mime="text/csv")


def vista_reglas(reglas, clave="e3_filtro_reglas"):
    st.subheader("Reglas de calidad")
    st.markdown(
        "Cada verificación del notebook se expresa como una regla declarativa, con dimensión, "
        "severidad y valor esperado. Las reglas se evalúan en vivo sobre el archivo cargado; "
        "las informativas describen el archivo sin un umbral de aprobación."
    )

    tabla = tabla_reglas(reglas)
    estados = st.multiselect("Filtrar por estado", ["Cumple", "No cumple", "Informativa"],
                             default=["Cumple", "No cumple", "Informativa"], key=clave)
    st.dataframe(tabla[tabla["Estado"].isin(estados)], hide_index=True, use_container_width=True)

    conteo = (
        tabla.groupby(["Dimensión", "Estado"]).size().rename("Reglas").reset_index()
    )
    grafico(px.bar(conteo, x="Dimensión", y="Reglas", color="Estado", barmode="stack",
                   color_discrete_map={"Cumple": "#2A9D8F", "No cumple": "#E76F51",
                                       "Informativa": "#8D99AE"},
                   title="Reglas por dimensión de calidad"), alto=360)

    explicacion("""
**Dimensiones.** Se siguen las dimensiones habituales de calidad de datos: completitud,
integridad física, validez, unicidad, consistencia, cobertura y plausibilidad, más una de
utilidad para el modelado.

**Severidad.** *Crítica* significa que un incumplimiento invalida el uso del archivo; *Alta*,
que exige corrección antes de modelar; *Media*, que exige un tratamiento documentado; y
*Informativa*, que describe el archivo sin umbral de aprobación.

**Trazabilidad.** Las reglas incumplidas o informativas enlazan con un hallazgo del registro,
donde se documentan la evidencia, el impacto y la decisión tomada.
""", titulo="Cómo leer esta tabla")


def vista_hallazgos(hallazgos):
    st.subheader("Registro de hallazgos")
    t = pd.DataFrame(hallazgos)

    a, b, c = st.columns(3)
    for col, estado in zip((a, b, c), ("Abierto", "Mitigado", "Cerrado")):
        col.metric(estado, int(t["Estado"].eq(estado).sum()))

    for h in sorted(hallazgos, key=lambda h: ORDEN_SEVERIDAD[h["Severidad"]]):
        with st.expander(f"{h['ID']} · {h['Hallazgo']} — {h['Severidad']} · {h['Estado']}",
                         expanded=h["Estado"] == "Abierto"):
            st.markdown(f"**Evidencia.** {h['Evidencia']}")
            st.markdown(f"**Impacto.** {h['Impacto']}")
            st.markdown(f"**Decisión.** {h['Decisión']}")

    st.caption(
        "Abierto: requiere verificación adicional. Mitigado: se aplicó un tratamiento que "
        "neutraliza el impacto. Cerrado: se analizó y no requiere tratamiento."
    )


def vista_linaje(p, reglas, n_reg, n_mesas, n_filas_estandar, ceros_implicitos):
    st.subheader("Linaje y reproducibilidad")
    huella = huella_archivo()

    st.markdown(
        "El diagrama muestra el recorrido de los datos desde el archivo original hasta la tabla "
        "de trabajo. Los nodos grises se calculan en este tablero; los azules, en el notebook."
    )
    columnas_trabajo = 11 - len(p["constantes"]) + 2
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#EEF1F4", color="#8D99AE"];
  edge [color="#5C677D"];
  original  [label="Archivo original\\n{huella['nombre']}\\nancho fijo, {LONGITUD_REGISTRO} caracteres"];
  tabla     [label="df_preconteo\\n{numero(n_reg)} × 11\\ncódigos como texto"];
  reglas    [label="Motor de reglas\\n{len(reglas)} reglas"];
  mesas     [label="Resumen por mesa\\n{numero(n_mesas)} mesas"];
  divipol   [label="Validación DIVIPOL\\n14.438 puestos", fillcolor="#DCEAF6", color="#4A78A8"];
  estandar  [label="df_preconteo_estandar\\n{numero(n_filas_estandar)} filas\\n+{numero(ceros_implicitos)} ceros implícitos", fillcolor="#DCEAF6", color="#4A78A8"];
  trabajo   [label="Tabla de trabajo\\n{numero(n_filas_estandar)} × {columnas_trabajo}\\nsin variables constantes", fillcolor="#DCEAF6", color="#4A78A8"];
  original -> tabla [label="read_fwf"];
  original -> reglas [label="lectura binaria"];
  tabla -> reglas;
  tabla -> mesas [label="groupby"];
  mesas -> divipol;
  mesas -> estandar [label="mesas × 16 opciones"];
  tabla -> estandar [label="merge"];
  estandar -> trabajo [label="selección"];
}}
""")

    st.markdown("**Huella del archivo auditado**")
    st.dataframe(pd.DataFrame({
        "Campo": ["Archivo", "SHA-256", "Tamaño", "Última modificación"],
        "Valor": [huella["nombre"], huella["sha256"], f"{numero(huella['bytes'])} bytes",
                  huella["modificado"]],
    }), hide_index=True, use_container_width=True)
    st.caption(
        "Si el paquete de la Registraduría trae un archivo de hashes (por ejemplo HASH.txt), "
        "compara este valor con el oficial para verificar que el archivo no fue alterado."
    )

    st.markdown("**Entorno de ejecución**")
    st.dataframe(pd.DataFrame({
        "Componente": ["Python", "pandas", "Streamlit", "Esquema del registro"],
        "Versión o valor": [
            platform.python_version(), pd.__version__, st.__version__,
            ", ".join(f"{n} ({l})" for n, l, _ in CAMPOS),
        ],
    }), hide_index=True, use_container_width=True)


def vista_secciones(df, mesas, p, e, m, n_reg, n_mesas, n_filas_estandar, ceros_implicitos):
    apartado = st.selectbox("Apartado", SECCIONES, key="e3_apartado")
    with st.expander("Preguntas de la plantilla para este apartado"):
        clave = apartado.split(" ")[0]
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(PREGUNTAS[clave], start=1)))

    if apartado.startswith("1.1"):
        seccion_faltantes(df, p, n_reg)
    elif apartado.startswith("1.2"):
        seccion_duplicados(p, n_mesas)
    elif apartado.startswith("1.3"):
        seccion_atipicos(df, mesas, p, m)
    elif apartado.startswith("1.4"):
        seccion_inconsistencias(df, mesas, p, n_reg, n_filas_estandar, ceros_implicitos)
    elif apartado.startswith("1.5"):
        seccion_corruptos(p, e)
    elif apartado.startswith("1.6"):
        seccion_seleccion(df, p, n_reg, n_filas_estandar)


# ------------------------------------------------------------
# Secciones de la plantilla
# ------------------------------------------------------------

def seccion_faltantes(df, p, n_reg):
    st.subheader("1.1 · Valores faltantes")
    x, y, z = st.columns(3)
    x.metric("Nulos y cadenas vacías", numero(sum(p["faltantes"].values())))
    y.metric("Variables revisadas", len(df.columns))
    z.metric("Mesas con menos de 16 filas", numero(p["mesas_menores_16"]))

    t = pd.DataFrame({
        "Variable": list(p["faltantes"]),
        "Faltantes": list(p["faltantes"].values()),
    })
    t["Completitud (%)"] = (1 - t["Faltantes"] / n_reg) * 100
    grafico(px.bar(t, x="Variable", y="Completitud (%)", range_y=[0, 101],
                   title="Completitud por variable (nulos y cadenas vacías)"))

    explicacion("""
**Identificación y distribución.** Se verificaron las 11 columnas con `isna()` y, en las de
texto, con `str.strip()` para detectar cadenas vacías. El resultado es **0 % de nulos y vacíos**
en todas las variables, incluidas las críticas (identificadores territoriales, candidato y votos).
Regla asociada: R-01.

**Impacto y decisión.** Los faltantes en identificadores o votos impedirían agregaciones
confiables. Como no se detectaron, no se imputaron ni eliminaron registros.

**Completitud estructural.** La ausencia de nulos no garantiza que estén todas las combinaciones
mesa–opción: **61.623 mesas** tienen menos de 16 filas. El análisis mostró que esas ausencias
dependen del departamento y corresponden a un formato que omite las opciones sin votos. Se
tratan como una inconsistencia de formato en 1.4 (hallazgo H-01), no como faltantes para
imputar con medias o medianas.
""")


def seccion_duplicados(p, n_mesas):
    st.subheader("1.2 · Valores duplicados")
    x, y, z = st.columns(3)
    x.metric("Duplicados exactos", numero(p["duplicados_exactos"]))
    y.metric("Duplicados por clave", numero(p["duplicados_logicos"]))
    z.metric("Mesas únicas", numero(n_mesas))
    st.markdown("**Clave lógica:** " + ", ".join(f"`{c}`" for c in CLAVE_REGISTRO))

    explicacion("""
**Detección.** Se compararon filas idénticas (R-07) y se comprobó la unicidad de una clave
compuesta de diez campos territoriales y electorales (R-08). No se identificaron duplicados.

**Criterio de tratamiento.** Repetir departamento, puesto o mesa no constituye duplicidad:
una mesa contiene votos de varias opciones. Eliminar filas con una clave demasiado corta
distorsionaría los totales y la representatividad. No fue necesario eliminar registros.
""")


def seccion_atipicos(df, mesas, p, m):
    st.subheader("1.3 · Valores atípicos")
    x, y, z = st.columns(3)
    x.metric("Mediana de votos por registro", numero(p["percentiles"][.5]))
    y.metric("Percentil 99", numero(p["percentiles"][.99]))
    z.metric("Votos negativos", numero(p["negativos"]))

    dist = (
        df["votos"].value_counts().sort_index()
        .rename_axis("Votos").reset_index(name="Registros")
    )
    grafico(px.bar(dist, x="Votos", y="Registros", log_y=True,
                   title="Distribución de votos por registro (eje vertical logarítmico)"))

    st.markdown("**Distribución por opción electoral**")
    st.dataframe(percentiles_por_opcion(df), hide_index=True, use_container_width=True)
    st.caption("Las candidaturas tienen escalas muy distintas: un umbral global no sirve para todas.")

    a, b, c, d = st.columns(4)
    a.metric("Límite inferior IQR", numero(m["bajo"]))
    b.metric("Límite superior IQR", numero(m["alto"]))
    c.metric("Mesas fuera del intervalo", numero(m["fuera"]))
    d.metric("Mesas con cero votos en total", numero(p["mesas_suma_cero"]))

    fuera = pd.DataFrame({
        "Lado": ["Por debajo", "Por debajo", "Por encima", "Por encima"],
        "Ámbito": ["Consulados", "Territorio nacional"] * 2,
        "Mesas": [m["debajo_consulado"], m["debajo"] - m["debajo_consulado"],
                  m["encima_consulado"], m["encima"] - m["encima_consulado"]],
    })
    grafico(px.bar(fuera, x="Lado", y="Mesas", color="Ámbito", barmode="group",
                   title="Mesas fuera del intervalo 1,5 × IQR por ámbito"), alto=360)

    hist = (
        mesas["total_votos"].value_counts().sort_index()
        .rename_axis("Votos por mesa").reset_index(name="Mesas")
    )
    fig = px.bar(hist, x="Votos por mesa", y="Mesas",
                 title="Suma de votos por mesa, con los límites IQR")
    fig.add_vline(x=m["bajo"], line_dash="dash", line_color="#E76F51")
    fig.add_vline(x=m["alto"], line_dash="dash", line_color="#E76F51")
    grafico(fig)

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("**Las 10 mesas con más votos**")
        top = mesas.nlargest(10, "total_votos").assign(
            ambito=lambda t: np.where(t["departamento"].eq("88"), "Consulado", "Nacional")
        )
        st.dataframe(top, hide_index=True, use_container_width=True)
    with derecha:
        st.markdown("**Mesas con suma total de cero votos**")
        st.dataframe(mesas[mesas["total_votos"].eq(0)], hide_index=True, use_container_width=True)

    explicacion("""
**Criterios.** Se inspeccionaron distribuciones, percentiles globales y por opción, y la regla
de Tukey (1,5 × IQR) sobre la suma de votos por mesa. El intervalo obtenido es **[14, 386]** y
**1.508 mesas (1,24 %)** quedan fuera de él: 1.103 por debajo y 405 por encima. Los consulados
concentran 353 de las 405 mesas superiores (87 %), y las diez mesas de mayor votación son
consulados (hallazgo H-03).

**Extremo inferior.** Hay **15 mesas cuya suma total es cero**, con una sola opción registrada.
No se explican por el formato de transmisión, porque incluso el formato que omite las opciones
sin votos debería traer al menos una fila con votos. Requieren verificación (hallazgo H-02).

**Decisión.** Los extremos no se eliminaron ni winsorizaron. Un valor extremo puede ser
legítimo, y su presencia no prueba un error ni una anomalía electoral. Cualquier validación
posterior requiere fuentes adicionales y comparaciones dentro de grupos territoriales.
""")


def seccion_inconsistencias(df, mesas, p, n_reg, n_filas_estandar, ceros_implicitos):
    st.subheader("1.4 · Datos inconsistentes")

    formato = formato_por_departamento(mesas)
    grafico(px.bar(formato, x="Departamento (código)", y="Mesas con 16 filas (%)",
                   range_y=[0, 101], hover_data=["Mesas"],
                   title="Formato de transmisión por departamento"))

    ceros = ceros_explicitos_por_formato(mesas)
    c1, c2 = st.columns(2)
    c1.metric("Mesas de 16 filas con algún cero explícito", porcentaje(ceros.get("16 filas", 0)))
    c2.metric("Mesas de menos de 16 filas con algún cero explícito",
              porcentaje(ceros.get("Menos de 16", 0)))

    ausencia = ausencia_por_departamento_y_opcion(df, mesas)
    fig = px.imshow(
        ausencia, color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto",
        labels=dict(x="Opción electoral", y="Departamento (código)", color="% sin fila"),
        title="Mesas sin fila por departamento y opción (%)",
    )
    grafico(fig, alto=650)
    st.caption(
        "Departamentos ordenados de mayor a menor número de mesas; opciones de mayor a menor "
        "ausencia nacional. El código 88 corresponde a consulados."
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Combinaciones observadas", numero(n_reg))
    c2.metric("Combinaciones estandarizadas", numero(n_filas_estandar))
    c3.metric("Ceros implícitos", numero(ceros_implicitos))

    ceros_totales = p["ceros"] + ceros_implicitos
    composicion = pd.DataFrame({
        "Tipo de fila": ["Con votos", "Cero explícito", "Cero implícito"],
        "Filas": [p["positivos"], p["ceros"], ceros_implicitos],
    })
    grafico(px.bar(composicion, x="Tipo de fila", y="Filas", color="Tipo de fila",
                   title="Composición de la tabla estandarizada"), alto=360)
    st.caption(
        f"Tras estandarizar, los ceros pasan de {porcentaje(100 * p['ceros'] / n_reg)} a "
        f"{porcentaje(100 * ceros_totales / n_filas_estandar)} de las filas."
    )

    explicacion("""
**Detección.** El análisis identificó dos formatos de transmisión según el departamento
(hallazgo H-01). En uno, cada mesa trae las 16 opciones; en el otro, solo las que recibieron
votos. Además se validaron tipos y longitudes (R-02, R-04), se conservaron los ceros iniciales
y los códigos alfanuméricos válidos, y se contrastaron las claves territoriales con DIVIPOL:
**122.020 mesas** y **14.438 puestos** con correspondencia completa.

**Por qué es válido completar con cero.** El 100 % de las mesas con 16 filas tiene ceros
escritos, frente al 2,9 % de las mesas con menos filas. Además, el archivo de escrutinio tiene
exactamente 122.020 × 16 = 1.952.320 filas. La evidencia indica que la fila se omite solo
cuando la opción no tuvo votos; la marca `cero_implicito` permite distinguir ambos casos.

**Estandarización.** Se construye una tabla nueva sin modificar el archivo fuente:
**1.952.320 filas**, **472.814 ceros implícitos** y **23.978.304 votos** antes y después.
Sin esta estandarización, conteos de filas, proporciones de ceros y promedios por opción
mezclarían dos representaciones distintas del mismo fenómeno.
""")


def seccion_corruptos(p, e):
    st.subheader("1.5 · Datos corruptos")
    huella = huella_archivo()

    a, b, c = st.columns(3)
    a.metric("Líneas leídas", numero(e["total"]))
    b.metric(f"Con {LONGITUD_REGISTRO} caracteres", numero(e["longitud_correcta"]))
    c.metric("Líneas vacías", numero(e["vacias"]))

    a, b, c = st.columns(3)
    a.metric("Con letras", numero(e["con_letras"]))
    b.metric("Con letras fuera de zona o puesto", numero(e["letras_fuera"]))
    c.metric("Comunicado distinto de 9999", numero(p["comunicado_distinto"]))

    st.markdown("**Esquema oficial verificado**")
    st.dataframe(pd.DataFrame(CAMPOS, columns=["Campo", "Longitud", "Tipo"]),
                 hide_index=True, use_container_width=True)
    st.markdown(f"**Huella SHA-256 del archivo:** `{huella['sha256']}`")

    explicacion("""
**Detección.** Se leyó el archivo original línea por línea, sin pasar por pandas, para
comprobar la longitud fija de **38 caracteres** (R-02), buscar líneas vacías (R-03) y ubicar
los caracteres no numéricos (R-04). Los **12.517 registros** con letras las tienen en el campo
`puesto`, que la documentación define como alfanumérico, por lo que no se consideran corrupción.

**Integridad del archivo.** Se calcula la huella SHA-256 del archivo, que permite verificar que
el análisis se hizo sobre exactamente el mismo archivo y detectar cualquier alteración posterior.
Para certificar que coincide con el publicado, debe compararse con el hash oficial del paquete.

**Tratamiento y límites.** No se eliminaron ni reconstruyeron registros. La validación
estructural no equivale a contrastar cada valor con las actas E-14: un dato puede ser
sintácticamente válido y aun así incorrecto.
""")


def seccion_seleccion(df, p, n_reg, n_filas_estandar):
    st.subheader("1.6 · Selección de datos")
    a, b, c = st.columns(3)
    a.metric("Registros del archivo original", numero(n_reg))
    b.metric("Filas de la tabla de trabajo", numero(n_filas_estandar))
    c.metric("Variables constantes", len(p["constantes"]))

    t = pd.DataFrame({
        "Variable": df.columns,
        "Valores distintos": [p["cardinalidad"][c] for c in df.columns],
        "Decisión": [
            "Excluir de la tabla de trabajo; conservar en el original"
            if c in p["constantes"] else "Conservar"
            for c in df.columns
        ],
    })
    grafico(px.bar(t, x="Variable", y="Valores distintos", log_y=True, color="Decisión",
                   title="Cardinalidad de variables (escala logarítmica)"))

    st.markdown("**Esquema de la tabla de trabajo**")
    columnas_trabajo = [c for c in df.columns if c not in p["constantes"]]
    esquema = pd.DataFrame({
        "Columna": columnas_trabajo + ["cero_implicito", "formato"],
        "Rol": (
            ["Identificador territorial"] * 5
            + ["Identificador electoral"] * 2
            + ["Medida"]
            + ["Control (no predictor)"] * 2
        )[: len(columnas_trabajo) + 2],
    })
    st.dataframe(esquema, hide_index=True, use_container_width=True)

    explicacion("""
**Criterios.** Se conservaron las variables territoriales para identificar mesas y cruzarlas
con DIVIPOL, `lista_partido` y `candidato` para diferenciar opciones, y `votos` como medida
cuantitativa. No se aplicó muestreo para reducir los datos.

**Exclusiones.** `jal`, `comunicado` y `circunscripcion` son constantes en este archivo (R-14,
hallazgo H-04): se conservan en el original para trazabilidad, pero no aportan variación.

**Tabla de trabajo.** Parte de la versión estandarizada (16 opciones por mesa) y agrega
`cero_implicito` y `formato` como variables de control. El número de filas por mesa no se usa
como predictor, porque refleja el formato de transmisión del departamento y no el resultado
electoral (hallazgo H-05).
""")


# ============================================================
# ESCRUTINIO Y NOVEDADES: LOCALIZACIÓN Y CARGA DE ARCHIVOS
# ============================================================
# Si services/data_loader.py define ESCRUTINIO_PATH o NOVEDADES_PATH, se usan esas
# rutas. Si no, los archivos se buscan a partir de la carpeta del preconteo (y de las
# carpetas superiores), tanto sueltos como dentro de los ZIP del paquete electoral, sin
# descomprimir nada en disco.

try:
    from services.data_loader import ESCRUTINIO_PATH
except ImportError:
    ESCRUTINIO_PATH = None

try:
    from services.data_loader import NOVEDADES_PATH
except ImportError:
    NOVEDADES_PATH = None

ARCHIVO_ESCRUTINIO = "_ficheros_MMV_4_MMV_9999_ESCRUTINIO.csv"
ARCHIVO_HASH_ESCRUTINIO = "HASH_ficheros_MMV_4_MMV_9999_ESCRUTINIO.txt"
ARCHIVO_NOVEDADES = "Novedad20260603.csv"
PATRON_DIVIPOL = "DIVIPOL_*.TXT"
PATRON_CANDIDATOS = "CANDIDATOS_*.TXT"
NIVELES_BUSQUEDA = 3
CARPETAS_IGNORADAS = {"node_modules", "__pycache__", "site-packages", "venv", "env"}

# Llave de la mesa común a las tres fuentes, con el ancho estándar de cada código.
LLAVE = ["dep", "mun", "zona", "puesto", "mesa"]
PUESTO = LLAVE[:4]
ANCHOS_LLAVE = {"dep": 2, "mun": 3, "zona": 3, "puesto": 2, "mesa": 6}
RETIRADOS = ["009", "013"]
CODIGOS_FORMATO = ["dep", "mun", "zona", "puesto", "mesa", "partido"]

COLS_ESC = ["comunicado", "dep", "mun", "zona", "puesto", "mesa", "comuna",
            "corporacion", "circ", "partido", "candidato", "votos"]

RENOMBRE_NOV = {"departamentoCodigo": "dep", "municipioCodigo": "mun", "zonaNombre": "zona",
                "puestoCodigo": "puesto", "mesaNumero": "mesa"}
CLAVE_MESA_NOV = list(RENOMBRE_NOV)
COLS_NUM_NOV = ["mesaNumero", "potencial", "sufragantesSegunE11", "firmasJurados", "votosUrna",
                "votosIncinerados", "sufragantesE11SegunE14", "totalVotos"]
COLS_VOTOS_NOV = ["sufragantesSegunE11", "votosUrna", "votosIncinerados",
                  "sufragantesE11SegunE14", "totalVotos"]
COLS_IQR_NOV = ["sufragantesSegunE11", "votosUrna", "sufragantesE11SegunE14", "totalVotos"]
COLS_SI_NO = ["enTerminos", "tachadurasEnmendaduras", "recontadaJurados",
              "recontadaComision", "excluida"]
CODIGOS_NOV = ["departamentoCodigo", "municipioCodigo", "zonaNombre", "puestoCodigo",
               "corporacionCodigo"]
MARCADORES_AUSENTE = {"", "-", "--", "?", "NA", "N/A", "NULL", "NAN", "NONE", "S/D", "SD",
                      "SIN DATO", "NO APLICA"}
JERARQUIA_NOV = {
    "Departamento": (["departamentoCodigo"], "departamentoNombre"),
    "Municipio": (["departamentoCodigo", "municipioCodigo"], "municipioNombre"),
    "Puesto": (["departamentoCodigo", "municipioCodigo", "zonaNombre", "puestoCodigo"],
               "puestoNombre"),
}
PARES_CONTEO = [
    ("votosUrna", "sufragantesSegunE11"),
    ("votosUrna", "sufragantesE11SegunE14"),
    ("sufragantesSegunE11", "sufragantesE11SegunE14"),
]
VARS_VOTANTES = ["sufragantesSegunE11", "sufragantesE11SegunE14", "votosUrna", "totalVotos"]
UMBRAL_URNA_E11 = 50


def _carpetas_busqueda():
    base = Path(PRECONTEO_PATH).resolve().parent
    carpetas = [base, *list(base.parents)[:NIVELES_BUSQUEDA]]
    extraidos = Path(tempfile.gettempdir()) / "mmv_presidente_2026"
    if extraidos.is_dir():
        carpetas.append(extraidos)
    return carpetas


def _recorrer(carpeta):
    """Archivos de una carpeta y sus subcarpetas, sin entrar en carpetas ocultas o de entorno."""
    for raiz, subcarpetas, archivos in os.walk(carpeta):
        subcarpetas[:] = [
            s for s in subcarpetas
            if not s.startswith(".") and s not in CARPETAS_IGNORADAS
        ]
        for nombre in archivos:
            yield Path(raiz) / nombre


def localizar(patron, ruta_preferida=None):
    """Busca un archivo y devuelve (ruta, miembro); miembro es el nombre dentro del ZIP o None."""
    if ruta_preferida and Path(ruta_preferida).is_file():
        return str(ruta_preferida), None
    patron = patron.upper()
    carpetas = _carpetas_busqueda()
    for carpeta in carpetas:
        for ruta in _recorrer(carpeta):
            if fnmatch.fnmatch(ruta.name.upper(), patron):
                return str(ruta), None
    revisados = set()
    for carpeta in carpetas:
        for ruta in _recorrer(carpeta):
            if ruta.suffix.lower() != ".zip" or ruta in revisados:
                continue
            revisados.add(ruta)
            try:
                with zipfile.ZipFile(ruta) as z:
                    for miembro in z.namelist():
                        if fnmatch.fnmatch(Path(miembro).name.upper(), patron):
                            return str(ruta), miembro
            except zipfile.BadZipFile:
                continue
    return None


@contextmanager
def abrir_recurso(recurso):
    """Abre en modo binario un archivo suelto o un miembro de un ZIP."""
    ruta, miembro = recurso
    if miembro is None:
        with open(ruta, "rb") as archivo:
            yield archivo
    else:
        with zipfile.ZipFile(ruta) as z, z.open(miembro) as archivo:
            yield archivo


def describir_recurso(recurso):
    ruta, miembro = recurso
    return f"{Path(ruta).name} › {miembro}" if miembro else Path(ruta).name


def ayuda_ubicacion(nombre_archivo):
    st.markdown(
        f"La página busca `{nombre_archivo}` a partir de la carpeta del preconteo "
        f"(`{Path(PRECONTEO_PATH).parent}`) y hasta {NIVELES_BUSQUEDA} carpetas más arriba, "
        "incluidos los ZIP del paquete electoral. Si el archivo está en otro lugar, define su "
        "ruta en `services/data_loader.py`:"
    )
    st.code(
        'ESCRUTINIO_PATH = Path("ruta/a/_ficheros_MMV_4_MMV_9999_ESCRUTINIO.csv")\n'
        'NOVEDADES_PATH = Path("ruta/a/Novedad20260603.csv")',
        language="python",
    )


@st.cache_data(show_spinner="Calculando la huella del archivo...")
def huella_recurso(ruta, miembro):
    """SHA-256, MD5, tamaño y fecha de modificación de un archivo (suelto o dentro de un ZIP)."""
    sha, md5, tamano = hashlib.sha256(), hashlib.md5(), 0
    with abrir_recurso((ruta, miembro)) as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            sha.update(bloque)
            md5.update(bloque)
            tamano += len(bloque)
    return {
        "nombre": Path(miembro).name if miembro else Path(ruta).name,
        "contenedor": Path(ruta).name if miembro else "",
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "bytes": tamano,
        "modificado": datetime.fromtimestamp(Path(ruta).stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
    }


@st.cache_data(show_spinner=False)
def md5_oficial_escrutinio():
    """MD5 que publica la Registraduría junto al archivo de escrutinio, si está disponible."""
    recurso = localizar(ARCHIVO_HASH_ESCRUTINIO)
    if recurso is None:
        return None
    with abrir_recurso(recurso) as archivo:
        datos = archivo.read()
    for codificacion in ("utf-16", "utf-8", "latin-1"):
        try:
            texto = datos.decode(codificacion)
        except UnicodeDecodeError:
            continue
        coincidencia = re.search(r"MD5\s*:\s*([0-9a-fA-F]{32})", texto)
        if coincidencia:
            return coincidencia.group(1).lower()
    return None


def normalizar_llave(df):
    """Completa con ceros a la izquierda los códigos de la llave de la mesa (en el mismo df)."""
    for columna, ancho in ANCHOS_LLAVE.items():
        if columna in df:
            df[columna] = df[columna].astype(str).str.strip().str.zfill(ancho)
    return df


def _normalizar_categoria(serie, ancho):
    categorias = serie.cat.categories
    nuevas = [str(c).strip().zfill(ancho) for c in categorias]
    if len(set(nuevas)) == len(nuevas):
        return serie.cat.rename_categories(nuevas)
    return serie.astype(str).str.strip().str.zfill(ancho).astype("category")


def _vacios_categoria(serie):
    vacias = [c for c in serie.cat.categories if str(c).strip() == ""]
    return int(serie.isin(vacias).sum()) if vacias else 0


def _longitudes(valores):
    return sorted({len(str(v)) for v in valores})


def leer_escrutinio(recurso):
    """Lee el escrutinio con los códigos como categorías (texto) para ahorrar memoria."""
    with abrir_recurso(recurso) as archivo:
        return pd.read_csv(
            archivo, sep=";", header=None, usecols=range(12), names=COLS_ESC,
            dtype="category", encoding="latin-1", keep_default_na=False, na_filter=False,
        )


@st.cache_data(show_spinner="Revisando el archivo de escrutinio línea por línea...")
def escanear_escrutinio(ruta, miembro):
    """Cuenta campos por línea y caracteres no numéricos sin pasar por pandas."""
    campos, no_numericas, fuera_puesto, total = Counter(), 0, 0, 0
    with abrir_recurso((ruta, miembro)) as archivo:
        for linea in archivo:
            total += 1
            partes = linea.rstrip(b"\r\n").split(b";")
            campos[len(partes)] += 1
            datos = partes[:12]
            if not all(p.isdigit() for p in datos):
                no_numericas += 1
                fuera_puesto += any(not p.isdigit() for i, p in enumerate(datos) if i != 4)
    return {"total": total, "campos": dict(campos), "no_numericas": no_numericas,
            "fuera_puesto": fuera_puesto}


@st.cache_data(show_spinner="Leyendo el archivo de novedades...")
def cargar_novedades():
    """Lee novedades en UTF-8 y reúne los campos de las líneas con comas en el nombre del puesto."""
    recurso = localizar(ARCHIVO_NOVEDADES, str(NOVEDADES_PATH) if NOVEDADES_PATH else None)
    if recurso is None:
        raise FileNotFoundError(f"No se encontró el archivo de novedades `{ARCHIVO_NOVEDADES}`.")
    filas, total, reparadas, malformadas = [], 0, 0, 0
    with abrir_recurso(recurso) as binario:
        texto = io.TextIOWrapper(binario, encoding="utf-8-sig", newline="")
        encabezado = texto.readline().rstrip("\r\n").split(",")
        n_campos = len(encabezado)
        i_nombre = encabezado.index("puestoNombre")
        n_cola = n_campos - i_nombre - 1
        for linea in texto:
            linea = linea.rstrip("\r\n")
            if not linea:
                continue
            total += 1
            partes = linea.split(",")
            if len(partes) > n_campos:
                partes = (partes[:i_nombre]
                          + [",".join(partes[i_nombre:len(partes) - n_cola])]
                          + partes[-n_cola:])
                reparadas += 1
            if len(partes) != n_campos:
                malformadas += 1
                continue
            filas.append(partes)
    df = pd.DataFrame(filas, columns=encabezado)
    info = {"recurso": recurso, "lineas": total, "reparadas": reparadas,
            "malformadas": malformadas, "campos": n_campos}
    return df, info


def novedades_por_mesa(nov):
    """Variables de novedades por mesa, con la llave estandarizada."""
    t = nov.rename(columns=RENOMBRE_NOV)
    columnas = LLAVE + ["potencial", "sufragantesSegunE11", "sufragantesE11SegunE14", "votosUrna",
                        "totalVotos", "excluida", "enTerminos", "estadoSobre", "recontadaComision"]
    t = normalizar_llave(t[columnas].copy())
    for c in ["potencial", "sufragantesSegunE11", "sufragantesE11SegunE14", "votosUrna", "totalVotos"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


@st.cache_data(show_spinner=False)
def cargar_divipol():
    recurso = localizar(PATRON_DIVIPOL)
    if recurso is None:
        return None
    with abrir_recurso(recurso) as archivo:
        d = pd.read_fwf(archivo, colspecs=[(0, 2), (2, 5), (5, 7), (7, 9), (51, 91)],
                        names=PUESTO + ["puesto_nombre"], dtype=str, encoding="latin-1")
    d = d.apply(lambda s: s.str.strip())
    return normalizar_llave(d)


@st.cache_data(show_spinner=False)
def cargar_codigos_candidatos():
    recurso = localizar(PATRON_CANDIDATOS)
    if recurso is None:
        return None
    with abrir_recurso(recurso) as archivo:
        c = pd.read_fwf(archivo, colspecs=[(16, 19)], names=["candidato"], dtype=str,
                        encoding="latin-1")
    return sorted(c["candidato"].dropna().str.strip().str.zfill(3).unique().tolist())


@st.cache_data(show_spinner="Agregando el preconteo por mesa y opción...")
def preconteo_por_mesa_y_opcion():
    """Preconteo en formato ancho (una fila por mesa, una columna por opción)."""
    df = load_preconteo()
    ancho = (
        df.groupby(CLAVE_MESA + ["candidato"], observed=True)["votos"].sum()
        .unstack("candidato", fill_value=0)
    )
    ancho.columns = [str(c).strip().zfill(3) for c in ancho.columns]
    ancho = ancho.reset_index().rename(columns=dict(zip(CLAVE_MESA, LLAVE)))
    normalizar_llave(ancho)
    pares = df[["candidato", "lista_partido"]].drop_duplicates().astype(str)
    pares = pd.DataFrame({
        "candidato": pares["candidato"].str.strip().str.zfill(3),
        "partido": pares["lista_partido"].str.strip().str.zfill(5),
    })
    originales = dict(zip(CLAVE_MESA, LLAVE)) | {"lista_partido": "partido"}
    longitudes = {nuevo: _longitudes(df[orig].unique()) for orig, nuevo in originales.items()}
    return ancho, pares, longitudes


# ============================================================
# CÁLCULOS DEL ESCRUTINIO (en caché)
# ============================================================

@st.cache_data(show_spinner=False)
def auditar_escrutinio():
    """Reproduce las verificaciones del notebook sobre el escrutinio y sus cruces."""
    recurso = localizar(ARCHIVO_ESCRUTINIO, str(ESCRUTINIO_PATH) if ESCRUTINIO_PATH else None)
    if recurso is None:
        raise FileNotFoundError(f"No se encontró el archivo de escrutinio `{ARCHIVO_ESCRUTINIO}`.")
    esc = leer_escrutinio(recurso)
    a = {"recurso": recurso, "filas": len(esc)}

    # --- Completitud y validez de los votos ---------------------------------
    a["faltantes"] = {c: _vacios_categoria(esc[c]) for c in COLS_ESC}
    categorias = pd.to_numeric(pd.Series(esc["votos"].cat.categories), errors="coerce").to_numpy()
    valores = categorias[esc["votos"].cat.codes.to_numpy()]
    a["votos_invalidos"] = int(np.isnan(valores).sum())
    esc["votos"] = np.nan_to_num(valores, nan=0).astype("int64")
    a["longitudes"] = {c: _longitudes(esc[c].cat.categories) for c in CODIGOS_FORMATO}

    # --- Resumen por mesa (llave cruda y estandarizada) ---------------------
    por_mesa = (
        esc.groupby(LLAVE, observed=True)["votos"].agg(total="sum", filas="size").reset_index()
    )
    por_mesa[LLAVE] = por_mesa[LLAVE].astype(str)
    claves_crudas = por_mesa.loc[por_mesa["dep"].ne("88"), LLAVE].copy()
    normalizar_llave(por_mesa)

    esc["partido"] = _normalizar_categoria(esc["partido"], 5)
    esc["candidato"] = _normalizar_categoria(esc["candidato"], 3)
    nacional = esc["dep"].astype(str).ne("88").to_numpy()

    # --- Unicidad ------------------------------------------------------------
    a["dup_exactos"] = int(esc.duplicated().sum())
    a["dup_clave"] = int(esc.duplicated(subset=LLAVE + ["candidato"]).sum())
    esc_nac = esc[nacional]
    a["filas_nacional"] = len(esc_nac)
    a["numeros_mesa"] = int(esc_nac["mesa"].nunique())
    a["falsos_dup_mesa"] = int(esc_nac.duplicated(subset=["mesa", "candidato"]).sum())
    a["falsos_dup_puesto"] = int(esc_nac.duplicated(subset=["puesto", "mesa", "candidato"]).sum())

    # --- Validez y selección ------------------------------------------------
    a["negativos"] = int(esc["votos"].lt(0).sum())
    a["pct_ceros"] = (
        esc["votos"].eq(0).groupby(esc["candidato"], observed=True).mean().mul(100).round(1)
        .rename(index=str).to_dict()
    )
    a["cardinalidad"] = {c: int(esc_nac[c].nunique()) for c in COLS_ESC}
    a["constantes"] = [c for c, n in a["cardinalidad"].items() if n == 1]
    pares_esc = esc_nac[["candidato", "partido"]].drop_duplicates().astype(str)

    alfa = [c for c in esc["puesto"].cat.categories if not str(c).isdigit()]
    filas_alfa = esc["puesto"].isin(alfa)
    a["filas_alfa"] = int(filas_alfa.sum())
    a["zonas_alfa"] = esc.loc[filas_alfa, "zona"].astype(str).value_counts().to_dict()

    ancho = (
        esc.groupby(LLAVE + ["candidato"], observed=True)["votos"].sum()
        .unstack("candidato", fill_value=0)
    )
    ancho.columns = [str(c) for c in ancho.columns]
    ancho = ancho.reset_index()
    ancho[LLAVE] = ancho[LLAVE].astype(str)
    normalizar_llave(ancho)
    del esc, esc_nac

    opciones = [c for c in ancho.columns if c not in LLAVE]
    a["opciones"] = len(opciones)
    a["mesas"] = len(por_mesa)
    a["mesas_filas_distintas_16"] = int(por_mesa["filas"].ne(16).sum())
    a["filas_por_mesa"] = por_mesa["filas"].value_counts().sort_index().to_dict()
    mesas_alfa = por_mesa[~por_mesa["puesto"].str.isdigit()]
    a["mesas_alfa"] = len(mesas_alfa)
    puestos_alfa = mesas_alfa[PUESTO].drop_duplicates()
    a["puestos_alfa"] = len(puestos_alfa)

    # --- Novedades y preconteo para los cruces -------------------------------
    nov, _ = cargar_novedades()
    nov_m = novedades_por_mesa(nov)
    nov_crudo = nov.rename(columns=RENOMBRE_NOV)
    a["longitudes_novedades"] = {c: _longitudes(nov_crudo[c].unique()) for c in LLAVE}
    a["mesas_nov_repetidas"] = int(nov_m.duplicated(LLAVE).sum())
    nov_u = nov_m.drop_duplicates(LLAVE)
    pre_ancho, pares_pre, long_pre = preconteo_por_mesa_y_opcion()
    a["longitudes_preconteo"] = long_pre

    consulado = por_mesa["dep"].eq("88")
    cruce = por_mesa[LLAVE].merge(nov_u[LLAVE].assign(en_novedades=True), on=LLAVE, how="left")
    sin_nov = cruce["en_novedades"].isna()
    a["mesas_sin_novedades"] = int(sin_nov.sum())
    a["dep_sin_novedades"] = cruce.loc[sin_nov, "dep"].value_counts().to_dict()
    a["mesas_consulado"] = int(consulado.sum())
    a["votos_total"] = int(por_mesa["total"].sum())
    a["votos_consulados"] = int(por_mesa.loc[consulado, "total"].sum())
    a["votos_nacional"] = a["votos_total"] - a["votos_consulados"]
    a["votos_novedades"] = int(nov_m["totalVotos"].sum())
    a["deps_total"] = int(por_mesa["dep"].nunique())
    a["deps_nacional"] = int(por_mesa.loc[~consulado, "dep"].nunique())

    # --- Mesas sin votos -----------------------------------------------------
    cero = por_mesa[por_mesa["total"].eq(0)].merge(nov_u[LLAVE + ["excluida"]], on=LLAVE, how="left")
    cero["origen"] = np.where(cero["dep"].eq("88"), "Consulado",
                              np.where(cero["excluida"].eq("SI"), "Excluida en escrutinio", "Otra"))
    a["mesas_cero"] = len(cero)
    a["mesas_cero_origen"] = cero["origen"].value_counts().to_dict()
    a["excluidas_novedades"] = int(nov_m["excluida"].eq("SI").sum())
    pre_total = pre_ancho[LLAVE].assign(votos_pre=pre_ancho.drop(columns=LLAVE).sum(axis=1))
    excluidas = cero[cero["origen"].eq("Excluida en escrutinio")].merge(pre_total, on=LLAVE, how="left")
    a["excluidas_con_votos_pre"] = int(excluidas["votos_pre"].gt(0).sum())
    a["excluidas_cero"] = int(cero["origen"].eq("Excluida en escrutinio").sum())

    # --- Mesas nacionales con las variables de novedades ---------------------
    vm = por_mesa[~consulado].merge(
        nov_u[LLAVE + ["potencial", "sufragantesSegunE11", "sufragantesE11SegunE14", "totalVotos",
                       "excluida"]],
        on=LLAVE, how="left",
    )
    a["mesas_nacional"] = len(vm)
    con_nov = vm["totalVotos"].notna()
    a["pct_total_coincide"] = float(vm.loc[con_nov, "total"].eq(vm.loc[con_nov, "totalVotos"]).mean() * 100) if con_nov.any() else 0.0
    a["total_distinto_novedades"] = int((con_nov & vm["total"].ne(vm["totalVotos"])).sum())

    # --- Atípicos (IQR sobre el total de votos por mesa nacional) -----------
    q1, q3 = vm["total"].quantile([.25, .75])
    iqr = q3 - q1
    bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    vm["atipico"] = np.select([vm["total"] < bajo, vm["total"] > alto], ["Bajo", "Alto"], "Normal")
    a.update({
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr), "mediana": float(vm["total"].median()),
        "lim_inf": float(bajo), "lim_sup": float(alto),
        "atipicos_bajos": int(vm["atipico"].eq("Bajo").sum()),
        "atipicos_altos": int(vm["atipico"].eq("Alto").sum()),
        "sobre_potencial": int((vm["total"] > vm["potencial"]).sum()),
        "max_ratio_potencial": float((vm["total"] / vm["potencial"]).max()),
    })
    vm["n_mesa"] = pd.to_numeric(vm["mesa"], errors="coerce")
    vm["ultima_mesa"] = vm["n_mesa"].eq(vm.groupby(PUESTO)["n_mesa"].transform("max"))
    bajos, altos = vm[vm["atipico"].eq("Bajo")], vm[vm["atipico"].eq("Alto")]
    a["bajos_ultima_pct"] = float(bajos["ultima_mesa"].mean() * 100) if len(bajos) else 0.0
    a["todas_ultima_pct"] = float(vm["ultima_mesa"].mean() * 100)
    a["bajos_excluidas"] = int(bajos["excluida"].eq("SI").sum())
    a["altos_potencial"] = {int(k): int(v) for k, v in altos["potencial"].value_counts().items()}
    a["potencial_todas"] = {int(k): int(v) for k, v in vm["potencial"].value_counts().items()}
    a["hist_total"] = (
        vm["total"].value_counts().sort_index().rename_axis("Votos en la mesa")
        .reset_index(name="Mesas")
    )

    # --- Actas idénticas -----------------------------------------------------
    an = ancho[ancho["dep"].ne("88")].copy()
    an["total"] = an[opciones].sum(axis=1)
    an["opciones_con_votos"] = an[opciones].gt(0).sum(axis=1)
    igual_puesto = an.duplicated(subset=PUESTO + opciones, keep=False) & an["total"].gt(0)
    igual_pais = an.duplicated(subset=opciones, keep=False) & an["total"].gt(0)
    iguales = an[igual_pais]
    a.update({
        "actas_igual_puesto": int(igual_puesto.sum()),
        "actas_igual_pais": int(igual_pais.sum()),
        "iguales_mediana": float(iguales["total"].median()) if len(iguales) else 0.0,
        "iguales_30": float(iguales["total"].le(30).mean() * 100) if len(iguales) else 0.0,
        "iguales_1o2": float(iguales["opciones_con_votos"].le(2).mean() * 100) if len(iguales) else 0.0,
        "iguales_rural": float(iguales["zona"].eq("099").mean() * 100) if len(iguales) else 0.0,
        "todas_rural": float(an["zona"].eq("099").mean() * 100),
        "mesas_esc_repetidas": int(an.duplicated(LLAVE).sum()),
    })
    union = an[LLAVE].merge(nov_u[LLAVE], on=LLAVE, how="outer", indicator=True)
    a["union_filas"] = len(union)
    a["union"] = union["_merge"].astype(str).value_counts().to_dict()

    # --- Formatos de los códigos entre fuentes ------------------------------
    nov_crudo_llave = nov_crudo[LLAVE].astype(str)
    a["emparejadas_crudo"] = len(claves_crudas.merge(nov_crudo_llave, on=LLAVE))
    a["emparejadas_estandar"] = len(an[LLAVE].merge(nov_u[LLAVE], on=LLAVE))

    # --- Candidatos retirados ------------------------------------------------
    pre_nac = pre_ancho[pre_ancho["dep"].ne("88")]
    tot_pre = pre_nac.drop(columns=LLAVE).sum()
    tot_esc = an[opciones].sum()
    filas_retiro = [c for c in RETIRADOS + ["998"] if c in tot_esc.index or c in tot_pre.index]
    retiro = pd.DataFrame({
        "Preconteo": [int(tot_pre.get(c, 0)) for c in filas_retiro],
        "Escrutinio": [int(tot_esc.get(c, 0)) for c in filas_retiro],
    }, index=[etiqueta_opcion(c) for c in filas_retiro])
    retiro.loc["Suma de las tres"] = retiro.sum()
    a["retiro"] = retiro.rename_axis("Opción").reset_index()
    a["retirados_con_votos"] = int(sum(an[c].gt(0).sum() for c in RETIRADOS if c in an))

    # --- Reglas de validación entre fuentes y catálogos ----------------------
    codigos = cargar_codigos_candidatos()
    if codigos is None:
        a["candidatos_fuera_catalogo"] = None
    else:
        validos = set(codigos) | set(OPCIONES_ESPECIALES)
        a["candidatos_fuera_catalogo"] = sorted(set(opciones) - validos)
    pares = pares_esc.merge(pares_pre, how="outer", indicator=True)
    a["pares_distintos"] = int(pares["_merge"].ne("both").sum()) + int(pares_esc["candidato"].duplicated().sum())
    a["pares_candidato_partido"] = len(pares_esc)

    divipol = cargar_divipol()
    puestos_nac = an[PUESTO].drop_duplicates()
    a["puestos_nacional"] = len(puestos_nac)
    if divipol is None:
        a["puestos_fuera_divipol"] = None
        a["alfa_en_divipol"] = None
        a["alfa_ejemplos"] = pd.DataFrame()
    else:
        dv = divipol[PUESTO + ["puesto_nombre"]].drop_duplicates(PUESTO)
        a["puestos_fuera_divipol"] = int(
            puestos_nac.merge(dv, on=PUESTO, how="left", indicator=True)["_merge"].eq("left_only").sum()
        )
        alfa_dv = puestos_alfa.merge(dv, on=PUESTO, how="left", indicator=True)
        a["alfa_en_divipol"] = int(alfa_dv["_merge"].eq("both").sum())
        a["alfa_ejemplos"] = alfa_dv.loc[alfa_dv["_merge"].eq("both"), PUESTO + ["puesto_nombre"]].head(5)

    # --- Registros del E-11 sospechosos -------------------------------------
    e11_potencial = vm["sufragantesSegunE11"].eq(vm["potencial"])
    e11_menor = vm["sufragantesSegunE11"] < vm["total"]
    vm["e11_sospechoso"] = (e11_potencial | e11_menor).astype(int)
    a.update({
        "e11_igual_potencial": int(e11_potencial.sum()),
        "e11_menor_total": int(e11_menor.sum()),
        "e11_sospechoso": int(vm["e11_sospechoso"].sum()),
        "e11_pot_mediana_e11": float(vm.loc[e11_potencial, "sufragantesSegunE11"].median()) if e11_potencial.any() else 0.0,
        "e11_pot_mediana_total": float(vm.loc[e11_potencial, "total"].median()) if e11_potencial.any() else 0.0,
        "tabla_e11_menor": vm.loc[e11_menor, LLAVE + ["total", "sufragantesSegunE11", "sufragantesE11SegunE14"]]
        .rename(columns={"total": "votos_escrutinio"}),
    })

    # --- Variable objetivo ---------------------------------------------------
    esc_w = an.set_index(LLAVE)[opciones]
    pre_w = pre_nac.set_index(LLAVE).reindex(index=esc_w.index, columns=esc_w.columns, fill_value=0)
    dif = esc_w - pre_w
    traslado = sum(pre_w[c] for c in RETIRADOS if c in pre_w)
    activos = [c for c in opciones if c not in RETIRADOS + list(OPCIONES_ESPECIALES)]
    blanco_nulo = [c for c in ("996", "997") if c in dif]
    cambio_998 = (dif["998"] - traslado).abs() if "998" in dif else 0
    objetivo = pd.DataFrame(index=esc_w.index)
    objetivo["magnitud"] = dif[activos + blanco_nulo].abs().sum(axis=1) + cambio_998
    objetivo["y_corregida"] = objetivo["magnitud"].gt(0).astype(int)
    objetivo = objetivo.reset_index()
    n1 = int(objetivo["y_corregida"].sum())
    a.update({
        "corregidas": n1,
        "pct_corregidas": float(n1 / len(objetivo) * 100) if len(objetivo) else 0.0,
        "relacion": float((len(objetivo) - n1) / n1) if n1 else float("nan"),
        "magnitudes": objetivo.loc[objetivo["y_corregida"].eq(1), "magnitud"].describe(percentiles=[.5, .9, .99]).to_dict(),
    })

    fuga = objetivo.merge(nov_u[LLAVE + ["enTerminos", "estadoSobre", "recontadaComision", "excluida"]],
                          on=LLAVE, how="left")
    a["fuga"] = pd.DataFrame([
        {"Variable": f"{col} = {valor}",
         "Mesas": int(fuga[col].eq(valor).sum()),
         "% corregidas": round(float(fuga.loc[fuga[col].eq(valor), "y_corregida"].mean() * 100), 1)
         if fuga[col].eq(valor).any() else 0.0}
        for col, valor in [("recontadaComision", "SI"), ("excluida", "SI"),
                           ("enTerminos", "NO"), ("estadoSobre", "MAL")]
    ])
    a["pct_corregidas_general"] = float(fuga["y_corregida"].mean() * 100) if len(fuga) else 0.0

    a["tabla_escrutinio"] = objetivo.merge(vm[LLAVE + ["e11_sospechoso"]], on=LLAVE, how="left")
    a["tabla_escrutinio"]["e11_sospechoso"] = a["tabla_escrutinio"]["e11_sospechoso"].fillna(0).astype(int)
    return a


# ============================================================
# UTILIDADES COMPARTIDAS POR ESCRUTINIO Y NOVEDADES
# ============================================================

TEXTO_OBSERVACIONES = (
    "Las observaciones ({ids}) están documentadas en el registro de hallazgos con su "
    "evidencia, impacto y tratamiento."
)


def _regla_valor(id_, dimension, texto, severidad, valor, esperado_cero=True, hallazgo=""):
    """Regla sobre un conteo; si el valor no se pudo calcular, queda como informativa."""
    if valor is None:
        return Regla(id_, dimension, texto, severidad, "0" if esperado_cero else "—",
                     "No disponible", "Informativa", hallazgo)
    return _regla(id_, dimension, texto, severidad, valor,
                  (valor == 0) if esperado_cero else None,
                  esperado="0" if esperado_cero else "—", hallazgo=hallazgo)


def informe_fuente_markdown(titulo, filas_archivo, veredicto, texto_dictamen, reglas, hallazgos,
                            alcance):
    filas = "\n".join(f"| {_celda(c)} | {_celda(v)} |" for c, v in filas_archivo)
    return f"""# Informe de auditoría de calidad de datos: {titulo}

Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Archivo auditado

| Campo | Valor |
|---|---|
{filas}

## Dictamen: {veredicto}

{texto_dictamen}

## Reglas de calidad

{_tabla_md(tabla_reglas(reglas))}

## Registro de hallazgos

{_tabla_md(pd.DataFrame(hallazgos))}

## Alcance y limitaciones

{alcance}
"""


def bloque_dictamen(reglas, hallazgos, veredicto, texto_dictamen, metrica_extra):
    st.subheader(f"Dictamen: {veredicto}")
    st.markdown(texto_dictamen)
    verificables = [r for r in reglas if r.estado != "Informativa"]
    abiertos = [h for h in hallazgos if h["Estado"] == "Abierto"]
    a, b, c, d = st.columns(4)
    a.metric("Reglas cumplidas",
             f"{sum(r.estado == 'Cumple' for r in verificables)} de {len(verificables)}")
    b.metric("Hallazgos registrados", len(hallazgos))
    c.metric("Hallazgos abiertos", len(abiertos))
    d.metric(*metrica_extra)
    return abiertos


def descargas(prefijo, informe, reglas, hallazgos, extra=None):
    st.markdown("**Descargar el informe**")
    columnas = st.columns(4 if extra else 3)
    columnas[0].download_button("Informe completo (Markdown)", informe,
                                file_name=f"informe_auditoria_{prefijo}.md", mime="text/markdown")
    columnas[1].download_button("Reglas de calidad (CSV)",
                                tabla_reglas(reglas).to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"reglas_calidad_{prefijo}.csv", mime="text/csv")
    columnas[2].download_button("Registro de hallazgos (CSV)",
                                pd.DataFrame(hallazgos).to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"hallazgos_{prefijo}.csv", mime="text/csv")
    if extra:
        etiqueta, datos, nombre = extra
        columnas[3].download_button(etiqueta, datos, file_name=nombre, mime="text/csv")


def tabla_huella(h, oficial=None):
    campos = ["Archivo", "SHA-256", "MD5", "Tamaño", "Última modificación"]
    valores = [h["nombre"], h["sha256"], h["md5"], f"{numero(h['bytes'])} bytes", h["modificado"]]
    if h["contenedor"]:
        campos.insert(1, "Contenido en")
        valores.insert(1, h["contenedor"])
    if oficial is not None:
        campos.append("MD5 oficial (Registraduría)")
        valores.append(oficial)
    st.dataframe(pd.DataFrame({"Campo": campos, "Valor": valores}),
                 hide_index=True, use_container_width=True)


def tabla_entorno(esquema):
    st.markdown("**Entorno de ejecución**")
    st.dataframe(pd.DataFrame({
        "Componente": ["Python", "pandas", "Streamlit", "Esquema del registro"],
        "Versión o valor": [platform.python_version(), pd.__version__, st.__version__, esquema],
    }), hide_index=True, use_container_width=True)


def selector_apartado(clave_fuente, fuente):
    apartado = st.selectbox("Apartado", SECCIONES, key=f"e3_apartado_{clave_fuente}")
    with st.expander("Preguntas de la plantilla para este apartado"):
        clave = apartado.split(" ")[0]
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(PREGUNTAS[clave], start=1)))
    return apartado


# ============================================================
# ESCRUTINIO: REGLAS Y HALLAZGOS
# ============================================================

def evaluar_reglas_escrutinio(a, e, md5_ok):
    faltantes = sum(a["faltantes"].values())
    campos_distintos = sum(n for k, n in e["campos"].items() if k != 13)
    md5_regla = (
        Regla("R-04", "Integridad física", "La huella MD5 coincide con la publicada por la Registraduría",
              "Crítica", "Coincide", "No disponible", "Informativa")
        if md5_ok is None else
        Regla("R-04", "Integridad física", "La huella MD5 coincide con la publicada por la Registraduría",
              "Crítica", "Coincide", "Coincide" if md5_ok else "No coincide",
              "Cumple" if md5_ok else "No cumple")
    )
    fuera_catalogo = a["candidatos_fuera_catalogo"]
    return [
        _regla("R-01", "Completitud", "Sin celdas vacías en las 12 columnas", "Crítica",
               faltantes, faltantes == 0),
        _regla("R-02", "Integridad física", "Todas las líneas tienen 13 campos (12 datos y uno final vacío)",
               "Crítica", campos_distintos, campos_distintos == 0),
        _regla("R-03", "Integridad física", "Las líneas leídas coinciden con las filas cargadas",
               "Crítica", abs(e["total"] - a["filas"]), e["total"] == a["filas"]),
        md5_regla,
        _regla("R-05", "Validez", "Caracteres no numéricos solo en el código de puesto",
               "Alta", e["fuera_puesto"], e["fuera_puesto"] == 0, hallazgo="H-08"),
        _regla("R-06", "Validez", "Votos numéricos y no negativos", "Crítica",
               a["votos_invalidos"] + a["negativos"], a["votos_invalidos"] + a["negativos"] == 0),
        _regla("R-07", "Unicidad", "Sin filas idénticas", "Crítica",
               a["dup_exactos"], a["dup_exactos"] == 0),
        _regla("R-08", "Unicidad", "Clave mesa × candidatura única (llave completa de la mesa)",
               "Crítica", a["dup_clave"], a["dup_clave"] == 0),
        _regla("R-09", "Cobertura", "Cada mesa tiene exactamente 16 filas, una por opción",
               "Alta", a["mesas_filas_distintas_16"], a["mesas_filas_distintas_16"] == 0),
        _regla_valor("R-10", "Consistencia", "Códigos de candidatura que no están en el catálogo",
                     "Alta", None if fuera_catalogo is None else len(fuera_catalogo)),
        _regla("R-11", "Consistencia", "Cada candidatura tiene un solo partido, igual que en el preconteo",
               "Alta", a["pares_distintos"], a["pares_distintos"] == 0),
        _regla("R-12", "Consistencia", "Candidatos retirados (009 y 013) con 0 votos",
               "Media", a["retirados_con_votos"], a["retirados_con_votos"] == 0, hallazgo="H-06"),
        _regla_valor("R-13", "Integridad referencial", "Puestos nacionales que no existen en DIVIPOL",
                     "Alta", a["puestos_fuera_divipol"]),
        _regla("R-14", "Cobertura entre fuentes", "Mesas del escrutinio con registro en novedades",
               "Media", a["mesas_sin_novedades"], a["mesas_sin_novedades"] == 0, hallazgo="H-01"),
        _regla("R-15", "Consistencia entre fuentes", "Total de la mesa = totalVotos de novedades",
               "Alta", a["total_distinto_novedades"], a["total_distinto_novedades"] == 0),
        _regla("R-16", "Plausibilidad", "Total de la mesa ≤ potencial máximo de la mesa",
               "Alta", a["sobre_potencial"], a["sobre_potencial"] == 0),
        _regla("R-17", "Consistencia entre fuentes", "Total de la mesa ≤ votantes del E-11",
               "Media", a["e11_menor_total"], a["e11_menor_total"] == 0, hallazgo="H-03"),
        _regla("R-18", "Unicidad", "Sin actas idénticas dentro del mismo puesto",
               "Media", a["actas_igual_puesto"], a["actas_igual_puesto"] == 0),
        _regla("R-19", "Plausibilidad", "Mesas con cero votos en total",
               "Informativa", a["mesas_cero"], None, esperado="—", hallazgo="H-02"),
        _regla("R-20", "Plausibilidad", "Mesas nacionales fuera del intervalo 1,5 × IQR",
               "Informativa", a["atipicos_bajos"] + a["atipicos_altos"], None, esperado="—",
               hallazgo="H-04"),
        _regla("R-21", "Unicidad", "Mesas con resultado idéntico a otra de otro puesto",
               "Informativa", a["actas_igual_pais"], None, esperado="—", hallazgo="H-07"),
        _regla("R-22", "Utilidad", "Variables con un único valor en las mesas nacionales",
               "Informativa", len(a["constantes"]), None, esperado="—", hallazgo="H-10"),
    ]


def registrar_hallazgos_escrutinio(a, e):
    ret = a["retiro"].set_index("Opción")
    pre_ret, esc_ret = int(ret.loc["Suma de las tres", "Preconteo"]), int(ret.loc["Suma de las tres", "Escrutinio"])
    retirados = [i for i in ret.index if i in RETIRADOS]
    pre_cand, esc_cand = int(ret.loc[retirados, "Preconteo"].sum()), int(ret.loc[retirados, "Escrutinio"].sum())
    origen = a["mesas_cero_origen"]
    fuga = a["fuga"].set_index("Variable")["% corregidas"]
    relacion = f"{a['relacion']:.1f}".replace(".", ",")
    sin_nov_88 = "todas" if set(a["dep_sin_novedades"]) <= {"88"} else numero(a["dep_sin_novedades"].get("88", 0))
    return [
        {
            "ID": "H-01",
            "Hallazgo": "Las mesas de consulados no tienen registro en novedades",
            "Severidad": "Alta",
            "Evidencia": (
                f"{numero(a['mesas_sin_novedades'])} mesas del escrutinio no cruzan con novedades "
                f"({sin_nov_88} del departamento 88). Total del escrutinio: {numero(a['votos_total'])}; "
                f"de novedades: {numero(a['votos_novedades'])}; de los consulados: "
                f"{numero(a['votos_consulados'])}"
                + (", que es exactamente la diferencia." if a["votos_total"] - a["votos_novedades"] == a["votos_consulados"] else ".")
            ),
            "Impacto": (
                "En la tabla unida esas mesas quedarían sin E-11, E-14 ni votos en urna, que son "
                "las variables más importantes del modelo."
            ),
            "Decisión": (
                f"Excluir los consulados del conjunto de modelado en lugar de imputar. Se conservan "
                f"{porcentaje(100 * a['mesas_nacional'] / a['mesas'])} de las mesas y "
                f"{porcentaje(100 * a['votos_nacional'] / a['votos_total'])} de los votos; los "
                "resultados no deben extenderse al voto en el exterior."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-02",
            "Hallazgo": "Mesas con cero votos en total",
            "Severidad": "Baja",
            "Evidencia": (
                f"{numero(a['mesas_cero'])} mesas suman cero: "
                f"{numero(origen.get('Excluida en escrutinio', 0))} excluidas por la comisión (las mismas "
                f"que novedades marca con excluida = SI) y {numero(origen.get('Consulado', 0))} de "
                f"consulados. {numero(a['excluidas_con_votos_pre'])} de las excluidas sí tenían "
                "votos en el preconteo."
            ),
            "Impacto": "Podrían confundirse con actas no transmitidas.",
            "Decisión": (
                "Conservar: son resultados reales del escrutinio. Las excluidas son además mesas "
                "corregidas, es decir, casos positivos de la variable objetivo."
            ),
            "Estado": "Cerrado",
        },
        {
            "ID": "H-03",
            "Hallazgo": "Registros del E-11 no confiables en novedades",
            "Severidad": "Media",
            "Evidencia": (
                f"{numero(a['e11_igual_potencial'])} mesas tienen el E-11 igual al potencial máximo "
                f"(mediana del E-11 {numero(a['e11_pot_mediana_e11'])} frente a "
                f"{numero(a['e11_pot_mediana_total'])} votos en el escrutinio) y "
                f"{numero(a['e11_menor_total'])} tienen menos votantes en el E-11 que votos en el "
                "escrutinio, que el E-14 sí confirma."
            ),
            "Impacto": (
                "La diferencia entre el preconteo y el E-11 es la principal señal del modelo; un E-11 "
                "mal registrado parecería una corrección."
            ),
            "Decisión": (
                f"Marcar las {numero(a['e11_sospechoso'])} mesas con `e11_sospechoso` y dejar como "
                "faltante su diferencia con el E-11. El escrutinio no se modifica y el valor real del "
                "E-11 no se puede recuperar sin el formulario físico."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-04",
            "Hallazgo": "Mesas atípicas por el total de votos",
            "Severidad": "Baja",
            "Evidencia": (
                f"{numero(a['atipicos_bajos'])} mesas por debajo de {numero(a['lim_inf'])} votos y "
                f"{numero(a['atipicos_altos'])} por encima de {numero(a['lim_sup'])}; "
                f"{numero(a['sobre_potencial'])} superan su potencial."
                + (f" El {porcentaje(a['bajos_ultima_pct'])} de las bajas es la última mesa de su puesto "
                   f"(frente al {porcentaje(a['todas_ultima_pct'])} del total)." if a["atipicos_bajos"] else "")
                + (" Potencial de las altas: " + ", ".join(f"{k} ({numero(v)})" for k, v in a["altos_potencial"].items())
                   + "." if a["altos_potencial"] else "")
            ),
            "Impacto": "Eliminarlas sesgaría el conjunto y quitaría mesas corregidas.",
            "Decisión": "Conservar sin recortar ni transformar; evaluar escalado solo en el modelado.",
            "Estado": "Cerrado",
        },
        {
            "ID": "H-05",
            "Hallazgo": "La misma mesa se escribe distinto en cada fuente",
            "Severidad": "Alta",
            "Evidencia": (
                "Zona, mesa y partido tienen distinto número de dígitos según la fuente. Con los "
                f"códigos originales se emparejan {numero(a['emparejadas_crudo'])} mesas con novedades; "
                f"estandarizados, {numero(a['emparejadas_estandar'])}."
            ),
            "Impacto": "Sin estandarizar no se pueden unir las fuentes ni construir la variable objetivo.",
            "Decisión": (
                "Completar con ceros a la izquierda: zona a 3 dígitos, mesa a 6 y partido a 5. Los "
                "códigos se mantienen como texto."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-06",
            "Hallazgo": "Votos de candidatos retirados trasladados al código 998",
            "Severidad": "Alta",
            "Evidencia": (
                f"Los candidatos 009 y 013 tienen {numero(pre_cand)} votos en el preconteo nacional y "
                f"{numero(esc_cand)} en el escrutinio. Sumando 009, 013 y 998, el preconteo tiene "
                f"{numero(pre_ret)} votos y el escrutinio {numero(esc_ret)}."
            ),
            "Impacto": (
                "Comparar las fuentes sin tenerlo en cuenta haría parecer corregidas casi todas las "
                "mesas con votos para esos candidatos."
            ),
            "Decisión": (
                "Descontar el traslado mesa por mesa al construir `y_corregida`; solo cuenta como "
                "corrección el cambio del 998 que va más allá del traslado."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-07",
            "Hallazgo": "Mesas con resultados idénticos en distintos puestos",
            "Severidad": "Baja",
            "Evidencia": (
                f"{numero(a['actas_igual_puesto'])} coincidencias dentro del mismo puesto y "
                f"{numero(a['actas_igual_pais'])} entre puestos distintos."
                + (f" Estas últimas tienen una mediana de {numero(a['iguales_mediana'])} votos, el "
                   f"{porcentaje(a['iguales_1o2'])} con votos en una o dos opciones y el "
                   f"{porcentaje(a['iguales_rural'])} en zona rural." if a["actas_igual_pais"] else "")
            ),
            "Impacto": "Si fueran actas copiadas, contarían doble en el entrenamiento.",
            "Decisión": "Conservar: la coincidencia se explica por azar en mesas muy pequeñas.",
            "Estado": "Cerrado",
        },
        {
            "ID": "H-08",
            "Hallazgo": "Códigos de puesto alfanuméricos",
            "Severidad": "Informativa",
            "Evidencia": (
                f"{numero(e['no_numericas'])} líneas tienen letras, todas en el código de puesto: "
                f"{numero(a['puestos_alfa'])} puestos y {numero(a['mesas_alfa'])} mesas"
                + ("." if a["alfa_en_divipol"] is None else
                   f"; {numero(a['alfa_en_divipol'])} de esos puestos existen en DIVIPOL.")
            ),
            "Impacto": "Convertir los códigos a número produciría errores o perdería esos puestos.",
            "Decisión": "No son datos corruptos; mantener los códigos como texto.",
            "Estado": "Cerrado",
        },
        {
            "ID": "H-09",
            "Hallazgo": "Variables con fuga de información",
            "Severidad": "Alta",
            "Evidencia": (
                f"Mesas corregidas: {porcentaje(fuga.get('recontadaComision = SI', 0))} de las "
                f"recontadas por la comisión y {porcentaje(fuga.get('excluida = SI', 0))} de las "
                f"excluidas, frente al {porcentaje(a['pct_corregidas_general'])} general."
            ),
            "Impacto": (
                "Son decisiones o registros del propio escrutinio: el modelo acertaría por una razón "
                "que no existe al momento de predecir."
            ),
            "Decisión": (
                "Excluir como entradas los votos del escrutinio, totalVotos, recontadaComision, "
                "excluida, estadoSobre y enTerminos."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-10",
            "Hallazgo": "Variables sin variación y redundantes",
            "Severidad": "Baja",
            "Evidencia": (
                "Constantes: " + ", ".join(f"`{c}`" for c in a["constantes"]) + ". `partido` es "
                f"redundante con `candidato` ({a['pares_candidato_partido']} combinaciones para "
                f"{a['opciones']} candidaturas)."
            ),
            "Impacto": "No aportan información al modelo.",
            "Decisión": "Excluirlas de la tabla de trabajo.",
            "Estado": "Mitigado",
        },
        {
            "ID": "H-11",
            "Hallazgo": "Desbalance de la variable objetivo",
            "Severidad": "Media",
            "Evidencia": (
                f"{numero(a['corregidas'])} mesas corregidas ({porcentaje(a['pct_corregidas'], 2)}), "
                f"unas {relacion} sin corrección por cada corregida."
            ),
            "Impacto": "La exactitud sería engañosa como métrica principal.",
            "Decisión": (
                "No muestrear; tratar el desbalance en el modelado (pesos de clase, PR-AUC, "
                "sensibilidad y precisión), solo sobre el conjunto de entrenamiento."
            ),
            "Estado": "Abierto",
        },
    ]


# ============================================================
# ESCRUTINIO: PÁGINA
# ============================================================

ALCANCE_ESCRUTINIO = (
    "Este informe evalúa la calidad técnica del archivo de escrutinio y su coherencia con el "
    "preconteo, las novedades y los catálogos oficiales. No certifica resultados electorales, no "
    "contrasta los valores con las actas físicas y no atribuye irregularidades a mesas o "
    "territorios. La coincidencia de la huella MD5 garantiza que el archivo es el publicado, no "
    "que lo publicado coincida con las actas."
)


def render_escrutinio():
    if not st.session_state.get("e3_activa_escrutinio", False):
        st.markdown(
            "Cada fila del escrutinio corresponde a una opción electoral en una mesa, después de "
            "la revisión de las comisiones escrutadoras. Además de revisar el archivo, la "
            "auditoría lo cruza con el preconteo y las novedades para construir la variable "
            "objetivo del proyecto (`y_corregida`)."
        )
        if st.button("Ejecutar auditoría del escrutinio", type="primary"):
            st.session_state.e3_activa_escrutinio = True
            st.rerun()
        st.info(
            "Lee tres archivos (escrutinio, preconteo y novedades), por lo que la primera "
            "ejecución puede tardar unos minutos; después los resultados quedan en caché."
        )
        return

    try:
        with st.spinner("Leyendo el escrutinio y cruzándolo con el preconteo y las novedades..."):
            a = auditar_escrutinio()
            e = escanear_escrutinio(*a["recurso"])
            h = huella_recurso(*a["recurso"])
            oficial = md5_oficial_escrutinio()
    except FileNotFoundError as error:
        st.error(str(error))
        ayuda_ubicacion(ARCHIVO_ESCRUTINIO if "escrutinio" in str(error) else ARCHIVO_NOVEDADES)
        return

    md5_ok = None if oficial is None else oficial == h["md5"]
    reglas = evaluar_reglas_escrutinio(a, e, md5_ok)
    hallazgos = registrar_hallazgos_escrutinio(a, e)
    veredicto, texto = dictamen(reglas, observaciones=TEXTO_OBSERVACIONES)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", numero(a["filas"]))
    c2.metric("Mesas", numero(a["mesas"]))
    c3.metric("Opciones electorales", numero(a["opciones"]))
    c4.metric("Votos escrutados", numero(a["votos_total"]))

    vista = st.radio("Vista", VISTAS, horizontal=True, key="e3_vista_escrutinio")
    st.divider()

    if vista == "Resumen ejecutivo":
        esc_resumen(a, h, oficial, reglas, hallazgos, veredicto, texto)
    elif vista == "Reglas de calidad":
        vista_reglas(reglas, clave="e3_filtro_reglas_escrutinio")
    elif vista == "Registro de hallazgos":
        vista_hallazgos(hallazgos)
    elif vista == "Preguntas de la plantilla":
        apartado = selector_apartado("escrutinio", "Escrutinio")
        secciones = {
            "1.1": esc_faltantes, "1.2": esc_duplicados, "1.3": esc_atipicos,
            "1.4": esc_inconsistencias, "1.5": esc_corruptos, "1.6": esc_seleccion,
        }
        clave = apartado.split(" ")[0]
        if clave == "1.5":
            esc_corruptos(a, e, h, oficial)
        else:
            secciones[clave](a)
    else:
        esc_linaje(a, h, oficial, reglas)

    st.divider()
    st.caption(
        "Resultados metodológicos del notebook del equipo; indicadores recalculados sobre los "
        "archivos locales. La detección estadística no constituye una auditoría electoral."
    )


def esc_resumen(a, h, oficial, reglas, hallazgos, veredicto, texto):
    abiertos = bloque_dictamen(reglas, hallazgos, veredicto, texto,
                               ("Mesas corregidas (y = 1)", numero(a["corregidas"])))
    izquierda, derecha = st.columns([2, 3])
    with izquierda:
        st.markdown("**Conformidad por dimensión**")
        st.dataframe(resumen_por_dimension(reglas), hide_index=True, use_container_width=True)
        st.markdown("**Hallazgos abiertos**")
        for hz in abiertos:
            st.markdown(f"- **{hz['ID']} · {hz['Hallazgo']}.** {hz['Decisión']}")
    with derecha:
        objetivo = pd.DataFrame({
            "Clase": ["Sin corrección (y = 0)", "Corregida (y = 1)"],
            "Mesas": [a["mesas_nacional"] - a["corregidas"], a["corregidas"]],
        })
        grafico(px.bar(objetivo, x="Clase", y="Mesas", color="Clase", text="Mesas",
                       color_discrete_sequence=["#8D99AE", "#E76F51"],
                       title=f"Variable objetivo: {porcentaje(a['pct_corregidas'], 2)} de mesas corregidas"),
                alto=380)

    filas = [
        ("Archivo", h["nombre"]), ("SHA-256", f"`{h['sha256']}`"), ("MD5", f"`{h['md5']}`"),
        ("MD5 oficial", f"`{oficial}`" if oficial else "No disponible"),
        ("Tamaño", f"{numero(h['bytes'])} bytes"), ("Registros", numero(a["filas"])),
        ("Mesas", numero(a["mesas"])), ("Votos escrutados", numero(a["votos_total"])),
    ]
    informe = informe_fuente_markdown("escrutinio presidencial 2026", filas, veredicto, texto,
                                      reglas, hallazgos, ALCANCE_ESCRUTINIO)
    descargas("escrutinio", informe, reglas, hallazgos, extra=(
        "Tabla de escrutinio (CSV)",
        a["tabla_escrutinio"].to_csv(index=False).encode("utf-8-sig"),
        "objetivo_escrutinio.csv",
    ))
    st.caption(
        "La tabla de escrutinio tiene una fila por mesa nacional con la llave, `magnitud`, "
        "`y_corregida` y `e11_sospechoso`. Para conservar los ceros a la izquierda, léela con "
        "`dtype=str` en las columnas de la llave."
    )


def esc_linaje(a, h, oficial, reglas):
    st.subheader("Linaje y reproducibilidad")
    st.markdown(
        "El diagrama muestra cómo se llega desde los archivos originales hasta la tabla de "
        "escrutinio. Los nodos grises son archivos y verificaciones; los azules, tablas derivadas."
    )
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#EEF1F4", color="#8D99AE"];
  edge [color="#5C677D"];
  escru    [label="Escrutinio\\n{h['nombre']}\\n{numero(a['filas'])} filas"];
  precon   [label="Preconteo\\nformato ancho por mesa"];
  noved    [label="Novedades\\n{numero(a['mesas_nacional'])} mesas"];
  catalog  [label="Catálogos\\nDIVIPOL y candidatos"];
  reglas   [label="Motor de reglas\\n{len(reglas)} reglas"];
  llave    [label="Llave estandarizada\\nzona 3 · mesa 6 · partido 5", fillcolor="#DCEAF6", color="#4A78A8"];
  nacional [label="Escrutinio nacional\\n{numero(a['filas_nacional'])} filas\\nsin consulados", fillcolor="#DCEAF6", color="#4A78A8"];
  objetivo [label="Variable objetivo\\ny_corregida y magnitud\\n(descuenta el traslado de 009 y 013)", fillcolor="#DCEAF6", color="#4A78A8"];
  e11      [label="Bandera e11_sospechoso\\n{numero(a['e11_sospechoso'])} mesas", fillcolor="#DCEAF6", color="#4A78A8"];
  tabla    [label="tabla_escrutinio\\n{numero(len(a['tabla_escrutinio']))} × {a['tabla_escrutinio'].shape[1]}", fillcolor="#DCEAF6", color="#4A78A8"];
  escru -> reglas [label="lectura binaria y MD5"];
  escru -> llave; precon -> llave; noved -> llave;
  catalog -> reglas;
  llave -> reglas;
  llave -> nacional [label="excluir dep. 88"];
  nacional -> objetivo; precon -> objetivo;
  noved -> e11; nacional -> e11;
  objetivo -> tabla; e11 -> tabla;
}}
""")
    st.markdown("**Huella del archivo auditado**")
    tabla_huella(h, oficial if oficial else "No disponible")
    st.caption(
        "La Registraduría publica una huella MD5 junto al archivo de escrutinio "
        f"(`{ARCHIVO_HASH_ESCRUTINIO}`). Si coincide con la calculada, el archivo no se dañó ni se "
        "modificó después de su publicación."
    )
    tabla_entorno("CSV separado por «;», sin encabezado: " + ", ".join(COLS_ESC))


# ------------------------------------------------------------
# Escrutinio: secciones de la plantilla
# ------------------------------------------------------------

def esc_faltantes(a):
    st.subheader("1.1 · Valores faltantes")
    x, y, z = st.columns(3)
    x.metric("Celdas vacías", numero(sum(a["faltantes"].values())))
    y.metric("Mesas con 16 filas", numero(a["mesas"] - a["mesas_filas_distintas_16"]))
    z.metric("Filas esperadas (mesas × 16)", numero(a["mesas"] * 16))

    ceros = pd.DataFrame({"Opción": list(a["pct_ceros"]), "% de mesas con 0 votos": list(a["pct_ceros"].values())})
    ceros["Tipo"] = np.where(ceros["Opción"].isin(RETIRADOS), "Candidato retirado", "Opción activa")
    ceros["Opción"] = ceros["Opción"].map(etiqueta_opcion)
    grafico(px.bar(ceros, x="Opción", y="% de mesas con 0 votos", color="Tipo",
                   color_discrete_map={"Candidato retirado": "#E76F51", "Opción activa": "#2A78D6"},
                   title="Porcentaje de mesas con 0 votos por opción"))

    origen = a["mesas_cero_origen"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mesas con 0 votos en total", numero(a["mesas_cero"]))
    c2.metric("Excluidas por la comisión", numero(origen.get("Excluida en escrutinio", 0)))
    c3.metric("Consulados sin votantes", numero(origen.get("Consulado", 0)))

    st.markdown("**Faltantes al cruzar con novedades**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mesas sin registro en novedades", numero(a["mesas_sin_novedades"]))
    c2.metric("Mesas que se conservan", porcentaje(100 * a["mesas_nacional"] / a["mesas"]))
    c3.metric("Votos que se conservan", porcentaje(100 * a["votos_nacional"] / a["votos_total"]))
    deps = ", ".join(f"{d} ({numero(n)})" for d, n in a["dep_sin_novedades"].items()) or "ninguno"

    explicacion(f"""
**Identificación.** Se revisaron tres formas en que un dato puede faltar: valores nulos, celdas
vacías o con solo espacios, y filas ausentes. En un archivo en formato largo, un dato también
falta si una mesa no tiene fila para alguna opción. Resultado: **{numero(sum(a['faltantes'].values()))}
celdas vacías**, y {numero(a['mesas_filas_distintas_16'])} de las {numero(a['mesas'])} mesas
tienen un número de filas distinto de 16 ({numero(a['filas'])} filas frente a
{numero(a['mesas'] * 16)} esperadas). A diferencia del preconteo, al que le faltan filas, el
escrutinio sirve de referencia completa al comparar las fuentes.

**Distribución en variables clave.** Un 0 en `votos` es un valor registrado, no un faltante. La
proporción de ceros depende de cada candidatura. La excepción son **009 y 013, con 0 votos en el
100 % de las mesas**: los dos candidatos se retiraron y sus votos pasaron al código 998 (hallazgo
H-06). No se rellenan ni se reconstruyen; el traslado se descuenta al construir la variable
objetivo.

**Mesas sin votos.** Las {numero(a['mesas_cero'])} mesas con 0 votos tienen explicación:
{numero(origen.get('Excluida en escrutinio', 0))} son las mesas que novedades marca con
`excluida = SI` (en el preconteo sí tenían votos, así que son mesas corregidas) y
{numero(origen.get('Consulado', 0))} son consulados donde nadie votó. Se conservan (hallazgo H-02).

**Faltantes al unir fuentes.** {numero(a['mesas_sin_novedades'])} mesas no tienen registro en
novedades; por departamento: {deps}. Se decidió **excluir los consulados en lugar de rellenar**:
les faltan justo las variables más importantes (E-11, E-14), faltan para todo el grupo y no al
azar, y los consulados siguen un proceso de escrutinio distinto. En las mesas nacionales, el total
del escrutinio coincide con `totalVotos` de novedades en el {porcentaje(a['pct_total_coincide'])}
de las mesas.

**Representatividad.** El conjunto conserva {numero(a['mesas_nacional'])} de {numero(a['mesas'])}
mesas y {numero(a['votos_nacional'])} de {numero(a['votos_total'])} votos, con
{a['deps_nacional']} de {a['deps_total']} departamentos. Es representativo del escrutinio
nacional, no del voto en el exterior. **Método de relleno:** no aplica.
""")


def esc_duplicados(a):
    st.subheader("1.2 · Valores duplicados")
    texto_pais = (
        f"En todo el país hay {numero(a['actas_igual_pais'])}, pero son mesas pequeñas (mediana de "
        f"{numero(a['iguales_mediana'])} votos; {porcentaje(a['iguales_30'])} con 30 o menos), con "
        f"votos en una o dos opciones ({porcentaje(a['iguales_1o2'])}) y mayoritariamente rurales "
        f"({porcentaje(a['iguales_rural'])} frente a {porcentaje(a['todas_rural'])} del total). Con tan "
        "pocos votos, la coincidencia es esperable por azar: se conservan (hallazgo H-07). Eliminarlas "
        "sesgaría el conjunto hacia las mesas urbanas."
        if a["actas_igual_pais"] else
        "Tampoco hay coincidencias entre mesas de puestos distintos."
    )
    x, y, z = st.columns(3)
    x.metric("Filas idénticas", numero(a["dup_exactos"]))
    y.metric("Mesa × candidatura repetida", numero(a["dup_clave"]))
    z.metric("Actas idénticas en el mismo puesto", numero(a["actas_igual_puesto"]))

    llaves = pd.DataFrame({
        "Llave usada": ["Mesa + candidatura", "Puesto + mesa + candidatura",
                        "Llave completa (dep, mun, zona, puesto, mesa) + candidatura"],
        "«Duplicados» encontrados": [a["falsos_dup_mesa"], a["falsos_dup_puesto"], a["dup_clave"]],
    })
    grafico(px.bar(llaves, x="«Duplicados» encontrados", y="Llave usada", orientation="h",
                   text="«Duplicados» encontrados",
                   title="Una llave incompleta produce duplicados falsos"), alto=320)

    union = a["union"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mesas repetidas en escrutinio", numero(a["mesas_esc_repetidas"]))
    c2.metric("Mesas repetidas en novedades", numero(a["mesas_nov_repetidas"]))
    c3.metric("Mesas emparejadas 1:1", numero(union.get("both", 0)))

    explicacion(f"""
**Identificación.** Se buscaron duplicados en dos niveles: filas idénticas
(**{numero(a['dup_exactos'])}**) y la misma mesa con la misma candidatura más de una vez, aunque los
votos fueran distintos (**{numero(a['dup_clave'])}**). Este segundo caso es el más peligroso,
porque significaría que una mesa reportó dos resultados para la misma opción.

**Consideración especial: la llave completa.** El número de mesa solo identifica una mesa dentro
de su puesto: hay {numero(a['numeros_mesa'])} números de mesa distintos, pero
{numero(a['mesas_nacional'])} mesas nacionales. Con una llave incompleta aparecerían
{numero(a['falsos_dup_mesa'])} duplicados falsos (mesa + candidatura) o
{numero(a['falsos_dup_puesto'])} (puesto + mesa + candidatura), y eliminarlos borraría casi todo
el conjunto.

**Actas copiadas.** Dos mesas con exactamente los mismos votos en las 16 opciones podrían indicar
un resultado transcrito dos veces. Dentro del mismo puesto, donde una copia sería más probable,
hay {numero(a['actas_igual_puesto'])} mesas con un resultado idéntico a otra. {texto_pais}

**Unión con novedades.** La unión se valida como uno a uno: {numero(union.get('both', 0))} mesas
emparejadas, {numero(union.get('left_only', 0))} solo en escrutinio y
{numero(union.get('right_only', 0))} solo en novedades. No se eliminó ningún registro.
""")


def esc_atipicos(a):
    st.subheader("1.3 · Valores atípicos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q1 · mediana · Q3", f"{numero(a['q1'])} · {numero(a['mediana'])} · {numero(a['q3'])}")
    c2.metric("Límites IQR", f"{numero(a['lim_inf'])} a {numero(a['lim_sup'])}")
    c3.metric("Atípicos bajos", numero(a["atipicos_bajos"]))
    c4.metric("Atípicos altos", numero(a["atipicos_altos"]))

    fig = px.bar(a["hist_total"], x="Votos en la mesa", y="Mesas",
                 title="Total de votos por mesa nacional, con los límites IQR")
    fig.add_vline(x=a["lim_inf"], line_dash="dash", line_color="#E76F51")
    fig.add_vline(x=a["lim_sup"], line_dash="dash", line_color="#E76F51")
    grafico(fig)

    izquierda, derecha = st.columns(2)
    with izquierda:
        ultima = pd.DataFrame({
            "Grupo": ["Atípicos bajos", "Todas las mesas"],
            "% que es la última mesa del puesto": [a["bajos_ultima_pct"], a["todas_ultima_pct"]],
        })
        grafico(px.bar(ultima, x="Grupo", y="% que es la última mesa del puesto", range_y=[0, 100],
                       title="Los atípicos bajos son la última mesa del puesto"), alto=340)
    with derecha:
        potencial = pd.DataFrame([
            {"Potencial": str(k), "Grupo": g, "Mesas": v}
            for g, d in [("Atípicos altos", a["altos_potencial"]), ("Todas las mesas", a["potencial_todas"])]
            for k, v in d.items()
        ])
        if len(potencial):
            grafico(px.bar(potencial, x="Potencial", y="Mesas", color="Grupo", barmode="group",
                           log_y=True, title="Potencial máximo de las mesas (escala logarítmica)"),
                    alto=340)

    explicacion(f"""
**Técnicas.** Se usaron un método estadístico y uno visual sobre el **total de votos por mesa**:
el rango intercuartílico (Q1 = {numero(a['q1'])}, Q3 = {numero(a['q3'])}, IQR = {numero(a['iqr'])};
límites de {numero(a['lim_inf'])} y {numero(a['lim_sup'])} votos), elegido porque no supone una
distribución normal, y el histograma, que muestra una cola de mesas pequeñas. Resultado:
**{numero(a['atipicos_bajos'])} atípicos bajos y {numero(a['atipicos_altos'])} altos**.

**¿Errores o información válida?** Se aplicó una regla de validación: una mesa no puede tener más
votos que su potencial máximo. **{numero(a['sobre_potencial'])} mesas lo superan**; la que más se
acerca llega al {porcentaje(100 * a['max_ratio_potencial'])}. Cada grupo tiene una explicación:
el {porcentaje(a['bajos_ultima_pct'])} de los atípicos bajos es la última mesa de su puesto
(frente al {porcentaje(a['todas_ultima_pct'])} del total), que recibe solo los votantes que
sobran, e incluye {numero(a['bajos_excluidas'])} mesas excluidas por la comisión. Los altos son
mesas con un potencial mayor que el habitual de 360. Son **información válida pero excepcional**.

**Decisión.** Se conservan sin eliminar ni recortar (hallazgo H-04): no son errores, eliminarlos
quitaría sobre todo últimas mesas de puesto y mesas corregidas, y lo atípico puede estar asociado
a lo que el modelo debe aprender. Si se usan modelos sensibles a la escala, se evaluará
estandarizar o aplicar logaritmos en el modelado.
""")


def esc_inconsistencias(a):
    st.subheader("1.4 · Datos inconsistentes")
    fuentes = {"Escrutinio": a["longitudes"], "Preconteo": a["longitudes_preconteo"],
               "Novedades": a["longitudes_novedades"]}
    formatos = pd.DataFrame({
        f: [", ".join(map(str, d[c])) if c in d else "—" for c in CODIGOS_FORMATO]
        for f, d in fuentes.items()
    }, index=CODIGOS_FORMATO).rename_axis("Código").reset_index()
    st.markdown("**Número de dígitos de cada código en los archivos originales**")
    st.dataframe(formatos, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Mesas emparejadas con los códigos originales", numero(a["emparejadas_crudo"]))
    c2.metric("Mesas emparejadas con los códigos estandarizados", numero(a["emparejadas_estandar"]))

    st.markdown("**Candidatos retirados: votos por fuente (mesas nacionales)**")
    st.dataframe(a["retiro"], hide_index=True, use_container_width=True)

    reglas = pd.DataFrame({
        "Regla": [
            "Votos no negativos", "Cada mesa tiene 16 filas",
            "Código de candidatura existe en el catálogo",
            "Cada candidatura tiene un solo partido, igual en ambas fuentes",
            "El puesto existe en DIVIPOL", "Candidatos retirados con 0 votos",
            "Total de la mesa ≤ potencial máximo", "Total de la mesa = totalVotos de novedades",
            "Total de la mesa ≤ votantes del E-11",
        ],
        "Incumplimientos": [
            numero(a["negativos"]), numero(a["mesas_filas_distintas_16"]),
            "No disponible" if a["candidatos_fuera_catalogo"] is None else numero(len(a["candidatos_fuera_catalogo"])),
            numero(a["pares_distintos"]),
            "No disponible" if a["puestos_fuera_divipol"] is None else numero(a["puestos_fuera_divipol"]),
            numero(a["retirados_con_votos"]), numero(a["sobre_potencial"]),
            numero(a["total_distinto_novedades"]), numero(a["e11_menor_total"]),
        ],
    })
    st.markdown("**Reglas de validación de la coherencia**")
    st.dataframe(reglas, hide_index=True, use_container_width=True)
    if len(a["tabla_e11_menor"]):
        st.markdown("**Mesas donde el escrutinio supera al E-11**")
        st.dataframe(a["tabla_e11_menor"], hide_index=True, use_container_width=True)

    ret = a["retiro"].set_index("Opción")
    pre_ret = int(ret.loc["Suma de las tres", "Preconteo"])
    esc_ret = int(ret.loc["Suma de las tres", "Escrutinio"])
    fallidas = reglas.loc[~reglas["Incumplimientos"].isin(["0", "No disponible"]), "Regla"].tolist()
    texto_reglas = (
        "Todas se cumplen." if not fallidas else
        "Tienen incumplimientos: " + "; ".join(f"«{r}»" for r in fallidas) + ". "
        + ("En las mesas donde el escrutinio supera al E-11, el E-14 confirma los votos del "
           "escrutinio: el valor que no cuadra es el E-11 de novedades, que se trata en 1.5."
           if a["e11_menor_total"] else "")
    )
    explicacion(f"""
**Identificación.** Se comparó el número de dígitos de los códigos que forman la llave de la mesa
en las tres fuentes (tabla). Zona, mesa y partido no tienen el mismo formato: la misma mesa se
escribe distinto en cada archivo (hallazgo H-05).

**¿Justifican una estandarización?** Sí, son críticas: con los códigos originales se emparejan
**{numero(a['emparejadas_crudo'])}** mesas del escrutinio con novedades, porque para pandas `001`
y `01` son textos distintos. Estandarizados, se emparejan **{numero(a['emparejadas_estandar'])}**.

**Homogeneización.** Se tomó como estándar el formato más largo de cada código y se completaron los
demás con ceros a la izquierda (`str.zfill`): zona a 3 dígitos, mesa a 6 y partido a 5. Los
códigos se mantienen como **texto**, para no perder los ceros ni los puestos alfanuméricos; `votos`
es la única variable numérica. No hubo que homogeneizar unidades: todo son conteos de votos.

**Inconsistencia de contenido.** Los candidatos 009 y 013 tienen votos en el preconteo y 0 en el
escrutinio, porque se retiraron y sus votos pasaron a 998. Por eso las tres categorías se comparan
juntas: suman {numero(pre_ret)} votos en el preconteo y {numero(esc_ret)} en el escrutinio. Al
construir la variable objetivo, el traslado se descuenta mesa por mesa (hallazgo H-06).

**Reglas de validación.** Se definieron nueve reglas (tabla): unas revisan el archivo por dentro y
otras lo comparan con las demás fuentes. {texto_reglas}
""")


def esc_corruptos(a, e, h, oficial):
    st.subheader("1.5 · Datos corruptos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Líneas leídas", numero(e["total"]))
    c2.metric("Filas cargadas", numero(a["filas"]))
    c3.metric("Líneas con letras", numero(e["no_numericas"]))
    c1, c2, c3 = st.columns(3)
    estado_md5 = "No disponible" if oficial is None else ("Coincide" if oficial == h["md5"] else "No coincide")
    c1.metric("Huella MD5 frente a la oficial", estado_md5)
    c2.metric("Puestos alfanuméricos", numero(a["puestos_alfa"]))
    c3.metric("Mesas con E-11 sospechoso", numero(a["e11_sospechoso"]))

    campos = pd.DataFrame({"Campos por línea": [str(k) for k in e["campos"]],
                           "Líneas": list(e["campos"].values())})
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("**Estructura de las líneas**")
        st.dataframe(campos, hide_index=True, use_container_width=True)
        st.markdown("**Zonas con puestos alfanuméricos (filas)**")
        st.dataframe(pd.DataFrame({"Zona": list(a["zonas_alfa"]), "Filas": list(a["zonas_alfa"].values())}),
                     hide_index=True, use_container_width=True)
    with derecha:
        st.markdown("**Ejemplos de puestos alfanuméricos en DIVIPOL**")
        if len(a["alfa_ejemplos"]):
            st.dataframe(a["alfa_ejemplos"], hide_index=True, use_container_width=True)
        else:
            st.caption("El catálogo DIVIPOL no está disponible.")
    st.markdown(f"**MD5 calculado:** `{h['md5']}`  \n**MD5 oficial:** `{oficial or 'no disponible'}`")

    comparacion = (
        "no se pudo comparar, porque no se encontró el archivo de huellas oficial"
        if oficial is None else
        ("es **idéntica a la oficial**" if oficial == h["md5"] else "**no coincide con la oficial**")
    )
    alfa_dv = ("" if a["alfa_en_divipol"] is None else
               f" **{numero(a['alfa_en_divipol'])} de los {numero(a['puestos_alfa'])} códigos existen "
               "en el catálogo oficial DIVIPOL**, que les asigna nombres de puestos reales.")
    explicacion(f"""
**Detección.** Se revisaron dos tipos de corrupción. (1) **Daño o alteración del archivo:** la
huella MD5 cambia por completo si se altera un solo byte; la calculada sobre el archivo
({numero(h['bytes'])} bytes) {comparacion}. (2) **Líneas mal formadas:**
se leyó el archivo línea por línea, sin pandas. Las {numero(e['total'])} líneas se cargaron en el
DataFrame, y {numero(e['no_numericas'])} tienen algún campo no numérico,
{numero(e['fuera_puesto'])} de ellas fuera del código de puesto.

**Puestos alfanuméricos.** Las letras están en el código de puesto de {numero(a['puestos_alfa'])}
puestos ({numero(a['mesas_alfa'])} mesas), sobre todo en zona rural.{alfa_dv} Es una forma válida de
codificar puestos y confirma por qué los códigos se mantienen como texto (hallazgo H-08).

**Medidas.** En el escrutinio no fue necesario corregir ni eliminar nada. Los registros con
problemas están en el **E-11 de novedades**: {numero(a['e11_igual_potencial'])} mesas con el E-11
igual al potencial máximo (mediana {numero(a['e11_pot_mediana_e11'])}, frente a
{numero(a['e11_pot_mediana_total'])} votos escrutados) y {numero(a['e11_menor_total'])} con menos
votantes que votos. Se marcan con `e11_sospechoso` ({numero(a['e11_sospechoso'])} mesas) en lugar
de corregirlos (hallazgo H-03).

**Casos no manejables.** El valor real del E-11 de esas mesas no se puede recuperar sin el
formulario físico. Estimarlo con el E-14 o con el escrutinio haría que las fuentes coincidieran de
forma artificial, justo lo que el modelo compara. La huella MD5 garantiza que el archivo es el
publicado, no que lo publicado coincida con las actas físicas.
""")


def esc_seleccion(a):
    st.subheader("1.6 · Selección de datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas del escrutinio nacional", numero(a["filas_nacional"]))
    c2.metric("Filas de la tabla final (una por mesa)", numero(len(a["tabla_escrutinio"])))
    c3.metric("Mesas corregidas", f"{numero(a['corregidas'])} ({porcentaje(a['pct_corregidas'], 2)})")

    excluir = set(a["constantes"]) | {"partido"}
    t = pd.DataFrame({
        "Variable": COLS_ESC,
        "Valores distintos": [a["cardinalidad"][c] for c in COLS_ESC],
        "Decisión": [
            "Excluir (un solo valor)" if c in a["constantes"]
            else "Excluir (redundante con candidato)" if c == "partido"
            else "Conservar"
            for c in COLS_ESC
        ],
    })
    grafico(px.bar(t, x="Variable", y="Valores distintos", color="Decisión", log_y=True,
                   title="Cardinalidad de las columnas del escrutinio nacional (escala logarítmica)"))

    st.markdown("**Fuga de información: % de mesas corregidas según variables del escrutinio**")
    st.dataframe(a["fuga"], hide_index=True, use_container_width=True)

    st.markdown("**Esquema de la tabla de escrutinio**")
    st.dataframe(pd.DataFrame({
        "Columna": LLAVE + ["magnitud", "y_corregida", "e11_sospechoso"],
        "Rol": ["Llave de la mesa"] * 5 + ["Complementaria (no es entrada)", "Variable objetivo",
                                           "Bandera de calidad del E-11"],
    }), hide_index=True, use_container_width=True)

    relacion = f"{a['relacion']:.1f}".replace(".", ",")
    excluidas = ", ".join(f"`{c}`" for c in COLS_ESC if c in excluir)
    explicacion(f"""
**Criterio.** El objetivo es predecir si una mesa será corregida en el escrutinio con información
disponible **antes** del escrutinio. Por eso el escrutinio solo aporta la **variable objetivo**: sus
votos se usan para saber si la mesa cambió, no como entradas. Se conservan la llave de la mesa,
`candidato` y `votos`; se excluyen {excluidas}, por tener un solo valor o ser redundantes
({a['pares_candidato_partido']} combinaciones candidatura–partido para {a['opciones']}
candidaturas).

**Variable objetivo.** `y_corregida` vale 1 si la mesa tuvo algún cambio entre el preconteo y el
escrutinio que no se explica por el retiro de 009 y 013: un cambio en una candidatura activa, en el
voto en blanco (996), en el nulo (997) o en el 998 más allá del traslado. `magnitud` cuenta los
votos que cambiaron y no se usa como entrada. Resultado: **{numero(a['corregidas'])} mesas
corregidas ({porcentaje(a['pct_corregidas'], 2)})**, unas {relacion} sin corrección por cada
corregida (hallazgo H-11).

**Muestreo y filtrado.** No se aplicó muestreo: reduciría las pocas mesas corregidas disponibles
para entrenar. Sí se aplicaron dos filtros: la exclusión de los consulados (1.1) y el paso de
{numero(a['filas_nacional'])} filas (mesa × opción) a {numero(len(a['tabla_escrutinio']))} (una
por mesa), que es la unidad de análisis.

**Fuga de información.** Se excluyen como entradas los votos del escrutinio y `totalVotos` (son el
resultado), `recontadaComision` y `excluida` (decisiones de la comisión, muy asociadas a la
corrección) y `estadoSobre` y `enTerminos` (se registran durante el escrutinio). La tabla final
tiene la llave, `magnitud`, `y_corregida` y `e11_sospechoso`, y se descarga desde el resumen
ejecutivo.
""")


# ============================================================
# CÁLCULOS DE NOVEDADES (en caché)
# ============================================================

def _limites_iqr(serie, k=1.5):
    q1, q3 = serie.quantile([.25, .75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def _problemas_formato(df, columnas):
    return pd.DataFrame({
        "Espacios en los extremos": {c: int((df[c] != df[c].str.strip()).sum()) for c in columnas},
        "Espacios dobles": {c: int(df[c].str.contains("  ", regex=False).sum()) for c in columnas},
        "Con minúsculas": {c: int((df[c] != df[c].str.upper()).sum()) for c in columnas},
    })


@st.cache_data(show_spinner=False)
def auditar_novedades():
    """Reproduce las verificaciones y la limpieza del notebook sobre novedades."""
    nov, info = cargar_novedades()
    a = {"info": info, "filas": len(nov), "columnas": nov.shape[1]}

    # --- Completitud ---------------------------------------------------------
    a["faltantes"] = {c: int(nov[c].isna().sum() + nov[c].str.strip().eq("").sum())
                      for c in nov.columns}
    a["marcadores"] = int(sum(
        nov[c].str.strip().str.upper().isin(MARCADORES_AUSENTE).sum() for c in nov.columns
    ))
    a["no_digitos"] = {c: int((~nov[c].str.fullmatch(r"\d+")).sum()) for c in COLS_NUM_NOV}
    a["codigos_no_numericos"] = {c: int((~nov[c].str.fullmatch(r"\d+")).sum()) for c in CODIGOS_NOV}
    alfa = nov.loc[~nov["puestoCodigo"].str.fullmatch(r"\d+"), "puestoCodigo"]
    a["ejemplos_puesto_alfa"] = sorted(alfa.unique().tolist())[:8]
    num = nov[COLS_NUM_NOV].apply(pd.to_numeric, errors="coerce")
    a["max_urna_texto"] = str(nov["votosUrna"].max())
    a["max_urna_numero"] = int(num["votosUrna"].max())

    # --- Unicidad ------------------------------------------------------------
    a["dup_exactos"] = int(nov.duplicated().sum())
    a["dup_id"] = int(nov["id"].duplicated().sum())
    a["dup_mesa"] = int(nov.duplicated(subset=CLAVE_MESA_NOV).sum())
    a["dup_sin_id"] = int(nov.drop(columns="id").duplicated().sum())

    # --- Validez de las banderas --------------------------------------------
    a["si_no_invalidos"] = int(sum((~nov[c].isin(["SI", "NO"])).sum() for c in COLS_SI_NO))
    a["sobre_invalidos"] = int((~nov["estadoSobre"].isin(["BUENO", "MAL"])).sum())
    a["potencial"] = {int(k): int(v) for k, v in num["potencial"].value_counts().items()}

    # --- Atípicos (IQR) ------------------------------------------------------
    limites = {c: _limites_iqr(num[c]) for c in COLS_VOTOS_NOV}
    a["iqr"] = pd.DataFrame([
        {"Columna": c, "Límite inferior": lo, "Límite superior": hi,
         "Atípicos": int(((num[c] < lo) | (num[c] > hi)).sum()),
         "Percentil 99": float(num[c].quantile(.99)), "Máximo": int(num[c].max())}
        for c, (lo, hi) in limites.items()
    ])
    a["iqr_lados"] = pd.DataFrame([
        {"Columna": c, "Lado": lado, "Mesas": int(m.sum())}
        for c in COLS_IQR_NOV
        for lado, m in [("Por debajo del límite inferior", num[c] < limites[c][0]),
                        ("Por encima del límite superior", num[c] > limites[c][1])]
    ])
    a["lim_inf"], a["lim_sup"] = limites["sufragantesSegunE11"]
    a["iqr_incinerados"] = int(a["iqr"].set_index("Columna").loc["votosIncinerados", "Atípicos"])
    a["potencial_no_360"] = int(num["potencial"].ne(360).sum())

    # --- Coherencia del identificador y la jerarquía ------------------------
    id_esperado = ("01" + nov["departamentoCodigo"] + nov["municipioCodigo"] + nov["zonaNombre"]
                   + nov["puestoCodigo"] + nov["mesaNumero"])
    a["id_distinto"] = int((id_esperado != nov["id"]).sum())
    a["corporaciones"] = sorted((nov["corporacionCodigo"] + " · " + nov["corporacionNombre"]).unique().tolist())
    a["jerarquia"] = {
        nivel: int(nov.groupby(claves)[nombre].nunique().gt(1).sum())
        for nivel, (claves, nombre) in JERARQUIA_NOV.items()
    }

    # --- Formato del texto ---------------------------------------------------
    cols_texto = [c for c in nov.columns if c not in COLS_NUM_NOV]
    formato = _problemas_formato(nov, cols_texto)
    a["formato"] = formato[formato.sum(axis=1) > 0].rename_axis("Columna").reset_index()
    nombre = nov["puestoNombre"]
    a["puesto_con_problema"] = int(((nombre != nombre.str.strip())
                                    | nombre.str.contains("  ", regex=False)
                                    | (nombre != nombre.str.upper())).sum())

    # --- Consistencia entre los conteos --------------------------------------
    cortes, etiquetas = [-1, 0, 5, UMBRAL_URNA_E11, float("inf")], ["0 (coinciden)", "1 a 5", f"6 a {UMBRAL_URNA_E11}", f"Más de {UMBRAL_URNA_E11}"]
    a["diferencias"] = pd.DataFrame({
        f"{x} vs {y}": pd.cut((num[x] - num[y]).abs(), bins=cortes, labels=etiquetas)
        .value_counts().reindex(etiquetas).astype(int)
        for x, y in PARES_CONTEO
    }).rename_axis("Diferencia absoluta").reset_index()
    a["urna_distinta_e11"] = int(num["votosUrna"].ne(num["sufragantesSegunE11"]).sum())
    a["excesos"] = {
        "totalVotos > votosUrna": int((num["totalVotos"] > num["votosUrna"]).sum()),
        "totalVotos > sufragantesSegunE11": int((num["totalVotos"] > num["sufragantesSegunE11"]).sum()),
        "totalVotos > sufragantesE11SegunE14": int((num["totalVotos"] > num["sufragantesE11SegunE14"]).sum()),
    }

    # --- Banderas de novedad --------------------------------------------------
    excluida, cero = nov["excluida"].eq("SI"), num["totalVotos"].eq(0)
    a["excluida_vs_cero"] = int((excluida != cero).sum())
    a["excluidas"] = int(excluida.sum())
    a["fuera_terminos"] = int(nov["enTerminos"].eq("NO").sum())
    a["fuera_terminos_excluidas"] = int((nov["enTerminos"].eq("NO") & excluida).sum())
    a["recontadas_ambas"] = int((nov["recontadaJurados"].eq("SI") & nov["recontadaComision"].eq("SI")).sum())
    a["sin_firmas"] = int(num["firmasJurados"].eq(0).sum())
    a["no_excluidas_sin_votantes"] = int((num["sufragantesSegunE11"].eq(0) & ~excluida).sum())
    tipos = {
        "Tachaduras o enmendaduras": nov["tachadurasEnmendaduras"].eq("SI"),
        "Recontada por jurados": nov["recontadaJurados"].eq("SI"),
        "Recontada por comisión": nov["recontadaComision"].eq("SI"),
        "Sobre en mal estado": nov["estadoSobre"].eq("MAL"),
        "Mesa excluida": excluida,
        "Fuera de términos": nov["enTerminos"].eq("NO"),
    }
    alguna = pd.concat(tipos, axis=1).any(axis=1)
    a["novedades"] = {k: int(v.sum()) for k, v in tipos.items()}
    a["mesas_con_novedad"] = int(alguna.sum())

    # --- E-11 frente al potencial y al total ---------------------------------
    a["e11_igual_potencial"] = int(num["sufragantesSegunE11"].eq(num["potencial"]).sum())
    a["e11_mayor_potencial"] = int((num["sufragantesSegunE11"] > num["potencial"]).sum())
    a["total_mayor_e11"] = a["excesos"]["totalVotos > sufragantesSegunE11"]

    # --- Comparación de las cuatro variables de votantes (mesas no excluidas)
    mesas = num.loc[~excluida, VARS_VOTANTES]
    a["mesas_comparadas"] = len(mesas)
    a["resumen_votantes"] = (
        mesas.agg(["sum", "mean", "median", "max"]).T.round(1)
        .rename(columns={"sum": "Suma", "mean": "Media", "median": "Mediana", "max": "Máximo"})
        .rename_axis("Variable").reset_index()
    )
    a["coincidencia"] = pd.DataFrame({
        x: {y: float((mesas[x] - mesas[y]).abs().eq(0).mean() * 100) for y in VARS_VOTANTES}
        for x in VARS_VOTANTES
    }).round(2)
    referencia = mesas["sufragantesSegunE11"]
    dif = pd.DataFrame({v: (mesas[v] - referencia).abs() for v in VARS_VOTANTES[1:]})
    a["dif_e11"] = pd.DataFrame({
        "Coinciden": dif.eq(0).mean(),
        "1 a 5 votos": (dif.gt(0) & dif.le(5)).mean(),
        "Más de 5 votos": dif.gt(5).mean(),
    }).mul(100).round(2).rename_axis("Variable").reset_index()

    # --- Limpieza: normalización, tipos, banderas y exclusión por IQR --------
    limpio = nov.copy()
    limpio["puestoNombre"] = (limpio["puestoNombre"].str.replace(r"\s+", " ", regex=True)
                              .str.strip().str.upper())
    a["formato_restante"] = int(_problemas_formato(limpio, cols_texto).sum().sum())
    claves, nombre_col = JERARQUIA_NOV["Puesto"]
    a["puestos_varios_nombres"] = int(limpio.groupby(claves)[nombre_col].nunique().gt(1).sum())
    limpio[COLS_NUM_NOV] = num.astype("Int64")
    if a["si_no_invalidos"] == 0:
        for c in COLS_SI_NO:
            limpio[c] = limpio[c].eq("SI")
    limpio["estadoSobre"] = limpio["estadoSobre"].astype("category")
    limpio["urna_difiere_e11"] = (limpio["votosUrna"] - limpio["sufragantesSegunE11"]).abs() > UMBRAL_URNA_E11
    limpio["total_mayor_urna"] = limpio["totalVotos"] > limpio["votosUrna"]
    limpio["total_mayor_e14"] = limpio["totalVotos"] > limpio["sufragantesE11SegunE14"]
    banderas = ["urna_difiere_e11", "total_mayor_urna", "total_mayor_e14"]
    a["banderas_antes"] = {b: int(limpio[b].sum()) for b in banderas}

    es_atipica = pd.concat(
        [(num[c] < limites[c][0]) | (num[c] > limites[c][1]) for c in COLS_IQR_NOV], axis=1
    ).any(axis=1)
    atipicos, limpio = limpio[es_atipica], limpio[~es_atipica]
    a["excluidas_iqr"] = len(atipicos)
    a["iqr_debajo"] = int(pd.concat([num[c] < limites[c][0] for c in COLS_IQR_NOV], axis=1).any(axis=1).sum())
    a["iqr_encima"] = int(pd.concat([num[c] > limites[c][1] for c in COLS_IQR_NOV], axis=1).any(axis=1).sum())
    a["pct_excluidas_iqr"] = float(len(atipicos) / len(nov) * 100) if len(nov) else 0.0
    a["novedades_perdidas"] = pd.DataFrame([
        {"Novedad": k, "Total": int(v.sum()), "Excluidas por IQR": int((v & es_atipica).sum()),
         "% perdido": round(float((v & es_atipica).sum() / v.sum() * 100), 1) if v.sum() else 0.0}
        for k, v in tipos.items()
    ])
    a["novedad_excluida_iqr"] = int((alguna & es_atipica).sum())
    a["banderas_despues"] = {b: int(limpio[b].sum()) for b in banderas}
    a["filas_limpias"] = len(limpio)
    a["columnas_limpias"] = limpio.shape[1]
    a["rango_limpio"] = (limpio[COLS_IQR_NOV].agg(["min", "max"]).T.astype(int)
                         .rename(columns={"min": "Mínimo", "max": "Máximo"})
                         .rename_axis("Columna").reset_index()) if len(limpio) else pd.DataFrame()
    a["nulos_limpios"] = int(limpio.isna().sum().sum())
    a["id_unico_limpio"] = bool(limpio["id"].is_unique)
    a["max_urna_limpio"] = int(limpio["votosUrna"].max()) if len(limpio) else 0
    a["tipos"] = pd.DataFrame({"Columna": limpio.columns, "Tipo": limpio.dtypes.astype(str).values})
    a["atipicos_csv"] = atipicos.to_csv(index=False).encode("utf-8-sig")
    a["limpio_csv"] = limpio.to_csv(index=False).encode("utf-8-sig")
    a["constantes"] = [c for c in nov.columns if nov[c].nunique() == 1]

    # --- Integridad referencial ----------------------------------------------
    claves_nov = normalizar_llave(nov[CLAVE_MESA_NOV].rename(columns=RENOMBRE_NOV).copy())
    mesas_pre = load_resumen_mesas()[CLAVE_MESA].rename(columns=dict(zip(CLAVE_MESA, LLAVE)))
    mesas_pre = normalizar_llave(mesas_pre.astype(str).copy()).drop_duplicates()
    cruce = claves_nov.merge(mesas_pre, on=LLAVE, how="outer", indicator=True)
    a["mesas_sin_preconteo"] = int(cruce["_merge"].eq("left_only").sum())
    solo_pre = cruce[cruce["_merge"].eq("right_only")]
    a["preconteo_nacional_sin_novedades"] = int(solo_pre["dep"].ne("88").sum())
    a["consulados_preconteo"] = int(solo_pre["dep"].eq("88").sum())
    divipol = cargar_divipol()
    if divipol is None:
        a["puestos_fuera_divipol"] = None
    else:
        puestos = claves_nov[PUESTO].drop_duplicates()
        a["puestos_fuera_divipol"] = int(
            puestos.merge(divipol[PUESTO].drop_duplicates(), on=PUESTO, how="left", indicator=True)
            ["_merge"].eq("left_only").sum()
        )
    a["puestos"] = int(claves_nov[PUESTO].drop_duplicates().shape[0])
    return a


# ============================================================
# NOVEDADES: REGLAS Y HALLAZGOS
# ============================================================

ALCANCE_NOVEDADES = (
    "Este informe evalúa la calidad técnica del archivo de novedades (formularios E-11 y E-14 "
    "digitados por mesa) y su coherencia con el preconteo y el catálogo DIVIPOL. No certifica "
    "resultados electorales ni contrasta los valores con los formularios físicos. Se asume que "
    "sufragantesSegunE11, votosUrna y sufragantesE11SegunE14 deberían coincidir; ese supuesto no "
    "está contrastado con la documentación oficial de los formularios."
)


def evaluar_reglas_novedades(a):
    info = a["info"]
    faltantes = sum(a["faltantes"].values())
    no_digitos = sum(a["no_digitos"].values())
    banderas = a["si_no_invalidos"] + a["sobre_invalidos"]
    jerarquia = sum(a["jerarquia"].values())
    ex = a["excesos"]
    return [
        _regla("R-01", "Integridad física",
               f"Todas las líneas tienen {info['campos']} campos tras reunir el nombre del puesto",
               "Crítica", info["malformadas"], info["malformadas"] == 0),
        _regla("R-02", "Integridad física", "Líneas con comas sin comillas en el nombre del puesto",
               "Informativa", info["reparadas"], None, esperado="—", hallazgo="H-01"),
        _regla("R-03", "Completitud", f"Sin nulos ni cadenas vacías en las {a['columnas']} columnas",
               "Crítica", faltantes, faltantes == 0),
        _regla("R-04", "Completitud", "Sin marcadores de dato ausente (-, NA, S/D, ...)",
               "Alta", a["marcadores"], a["marcadores"] == 0),
        _regla("R-05", "Validez", "Los conteos contienen solo dígitos",
               "Crítica", no_digitos, no_digitos == 0),
        _regla("R-06", "Validez", "Banderas solo con SI/NO y estado del sobre BUENO/MAL",
               "Alta", banderas, banderas == 0),
        _regla("R-07", "Unicidad", "Identificador id único", "Crítica", a["dup_id"], a["dup_id"] == 0),
        _regla("R-08", "Unicidad", "Una sola fila por mesa (llave completa)", "Crítica",
               a["dup_mesa"], a["dup_mesa"] == 0),
        _regla("R-09", "Unicidad", "Sin filas iguales aun ignorando id", "Alta",
               a["dup_sin_id"], a["dup_sin_id"] == 0),
        _regla("R-10", "Consistencia", "El id coincide con los códigos de la mesa", "Alta",
               a["id_distinto"], a["id_distinto"] == 0),
        _regla("R-11", "Consistencia", "Cada código territorial tiene un solo nombre", "Media",
               jerarquia, jerarquia == 0),
        _regla("R-12", "Consistencia", "puestoNombre sin espacios sobrantes ni minúsculas", "Baja",
               a["puesto_con_problema"], a["puesto_con_problema"] == 0, hallazgo="H-02"),
        _regla("R-13", "Consistencia", f"votosUrna y E-11 difieren en más de {UMBRAL_URNA_E11} votos",
               "Media", a["banderas_antes"]["urna_difiere_e11"],
               a["banderas_antes"]["urna_difiere_e11"] == 0, hallazgo="H-03"),
        _regla("R-14", "Consistencia", "totalVotos ≤ votosUrna", "Media",
               ex["totalVotos > votosUrna"], ex["totalVotos > votosUrna"] == 0, hallazgo="H-03"),
        _regla("R-15", "Consistencia", "totalVotos ≤ sufragantesE11SegunE14", "Media",
               ex["totalVotos > sufragantesE11SegunE14"], ex["totalVotos > sufragantesE11SegunE14"] == 0,
               hallazgo="H-03"),
        _regla("R-16", "Consistencia", "totalVotos ≤ sufragantesSegunE11", "Media",
               a["total_mayor_e11"], a["total_mayor_e11"] == 0, hallazgo="H-04"),
        _regla("R-17", "Consistencia", "excluida = SI si y solo si totalVotos = 0", "Alta",
               a["excluida_vs_cero"], a["excluida_vs_cero"] == 0),
        _regla("R-18", "Plausibilidad", "Votantes del E-11 ≤ potencial de la mesa", "Alta",
               a["e11_mayor_potencial"], a["e11_mayor_potencial"] == 0),
        _regla("R-19", "Plausibilidad", "E-11 igual al potencial máximo de la mesa", "Media",
               a["e11_igual_potencial"], a["e11_igual_potencial"] == 0, hallazgo="H-04"),
        _regla("R-20", "Integridad referencial", "Mesas de novedades que no están en el preconteo",
               "Alta", a["mesas_sin_preconteo"], a["mesas_sin_preconteo"] == 0),
        _regla_valor("R-21", "Integridad referencial", "Puestos que no existen en DIVIPOL", "Alta",
                     a["puestos_fuera_divipol"]),
        _regla("R-22", "Plausibilidad", "Mesas fuera del intervalo 1,5 × IQR en alguna columna de votantes",
               "Informativa", a["excluidas_iqr"], None, esperado="—", hallazgo="H-05"),
        _regla("R-23", "Plausibilidad", "Mesas con alguna novedad registrada", "Informativa",
               a["mesas_con_novedad"], None, esperado="—", hallazgo="H-07"),
        _regla("R-24", "Utilidad", "Variables con un único valor en todo el archivo", "Informativa",
               len(a["constantes"]), None, esperado="—", hallazgo="H-09"),
    ]


def registrar_hallazgos_novedades(a):
    info = a["info"]
    fmt = a["formato"].set_index("Columna")
    fila = fmt.loc["puestoNombre"] if "puestoNombre" in fmt.index else pd.Series(0, index=fmt.columns)
    antes, despues, ex = a["banderas_antes"], a["banderas_despues"], a["excesos"]
    perdidas = a["novedades_perdidas"].set_index("Novedad")
    return [
        {
            "ID": "H-01",
            "Hallazgo": "Comas sin comillas en el nombre del puesto",
            "Severidad": "Alta",
            "Evidencia": (
                f"{numero(info['reparadas'])} líneas traen más de {info['campos']} campos porque "
                "`puestoNombre` contiene comas sin entrecomillar."
            ),
            "Impacto": "Leerlas sin reparación desplaza las columnas o descarta mesas válidas.",
            "Decisión": (
                "Reunir los campos sobrantes en `puestoNombre` al leer el archivo, en UTF-8. "
                "Descartar las filas habría perdido mesas."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-02",
            "Hallazgo": "Formato irregular del nombre del puesto",
            "Severidad": "Baja",
            "Evidencia": (
                f"{numero(fila.get('Espacios en los extremos', 0))} filas con espacios en los extremos, "
                f"{numero(fila.get('Espacios dobles', 0))} con espacios dobles y "
                f"{numero(fila.get('Con minúsculas', 0))} con minúsculas; las demás columnas de texto "
                "están normalizadas."
            ),
            "Impacto": "Dificulta agrupar o cruzar por nombre de puesto.",
            "Decisión": (
                f"Quitar espacios y pasar a mayúsculas. Quedan {numero(a['formato_restante'])} "
                f"problemas y {numero(a['puestos_varios_nombres'])} puestos con más de un nombre."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-03",
            "Hallazgo": "Conteos de votantes que no coinciden entre sí",
            "Severidad": "Media",
            "Evidencia": (
                f"votosUrna difiere del E-11 en más de {UMBRAL_URNA_E11} votos en "
                f"{numero(antes['urna_difiere_e11'])} mesas; totalVotos supera a votosUrna en "
                f"{numero(ex['totalVotos > votosUrna'])} y al E-14 en "
                f"{numero(ex['totalVotos > sufragantesE11SegunE14'])}."
            ),
            "Impacto": "No se puede saber cuál de los conteos es el correcto.",
            "Decisión": (
                "No corregir: marcar con `urna_difiere_e11`, `total_mayor_urna` y `total_mayor_e14` "
                f"({numero(despues['urna_difiere_e11'])}, {numero(despues['total_mayor_urna'])} y "
                f"{numero(despues['total_mayor_e14'])} tras excluir atípicos). Falta contrastar el "
                "supuesto de igualdad con la documentación de los formularios; el umbral de "
                f"{UMBRAL_URNA_E11} votos es arbitrario."
            ),
            "Estado": "Abierto",
        },
        {
            "ID": "H-04",
            "Hallazgo": "Registros del E-11 no confiables",
            "Severidad": "Media",
            "Evidencia": (
                f"{numero(a['e11_igual_potencial'])} mesas tienen el E-11 igual al potencial máximo y "
                f"{numero(a['total_mayor_e11'])} tienen más votos totales que votantes en el E-11."
            ),
            "Impacto": (
                "La diferencia entre el preconteo y el E-11 es la principal señal del modelo; un "
                "E-11 mal registrado parecería una corrección."
            ),
            "Decisión": (
                "Marcar con `e11_sospechoso` (definida en la auditoría del escrutinio) y dejar como "
                "faltante su diferencia con el E-11. El valor real no se puede recuperar."
            ),
            "Estado": "Mitigado",
        },
        {
            "ID": "H-05",
            "Hallazgo": "La exclusión por IQR elimina mesas con novedades",
            "Severidad": "Media",
            "Evidencia": (
                f"Se excluyen {numero(a['excluidas_iqr'])} mesas ({porcentaje(a['pct_excluidas_iqr'], 2)}): "
                f"{numero(a['iqr_debajo'])} por debajo de {numero(a['lim_inf'])} votantes y "
                f"{numero(a['iqr_encima'])} por encima de {numero(a['lim_sup'])}. Entre ellas están "
                f"{numero(perdidas.loc['Mesa excluida', 'Excluidas por IQR'])} de las "
                f"{numero(perdidas.loc['Mesa excluida', 'Total'])} mesas excluidas y "
                f"{numero(perdidas.loc['Fuera de términos', 'Excluidas por IQR'])} de las "
                f"{numero(perdidas.loc['Fuera de términos', 'Total'])} fuera de términos."
            ),
            "Impacto": (
                "Si el análisis se centra en las novedades, se pierden casos poco frecuentes. "
                "La auditoría del escrutinio, en cambio, conserva sus atípicos."
            ),
            "Decisión": (
                "Excluir de `df_clean` y guardar en `df_atipicos` (descargable). Revisarlas antes de "
                "descartarlas en el modelado."
            ),
            "Estado": "Abierto",
        },
        {
            "ID": "H-06",
            "Hallazgo": "Códigos y conteos leídos como texto",
            "Severidad": "Media",
            "Evidencia": (
                f"`puestoCodigo` tiene {numero(a['codigos_no_numericos']['puestoCodigo'])} valores "
                "alfanuméricos y los demás códigos tienen ceros a la izquierda. Como texto, el máximo "
                f"de votosUrna es {a['max_urna_texto']}; como número, {numero(a['max_urna_numero'])}."
            ),
            "Impacto": "Ordenar o resumir conteos como texto da resultados engañosos.",
            "Decisión": "Conteos a `Int64`, banderas SI/NO a booleano y códigos como texto.",
            "Estado": "Mitigado",
        },
        {
            "ID": "H-07",
            "Hallazgo": "Banderas de novedad para revisión",
            "Severidad": "Baja",
            "Evidencia": (
                f"{numero(a['recontadas_ambas'])} mesas recontadas por jurados y por comisión a la "
                f"vez, {numero(a['sin_firmas'])} sin firmas de jurados y {numero(a['fuera_terminos'])} "
                f"fuera de términos ({numero(a['fuera_terminos_excluidas'])} de ellas excluidas)."
            ),
            "Impacto": "Son situaciones posibles pero poco frecuentes.",
            "Decisión": "Conservar: describen la mesa, no son errores de captura.",
            "Estado": "Abierto",
        },
        {
            "ID": "H-08",
            "Hallazgo": "Variables con fuga de información",
            "Severidad": "Alta",
            "Evidencia": (
                "`recontadaComision`, `excluida`, `estadoSobre`, `enTerminos` y `totalVotos` se "
                "registran durante el escrutinio o son su resultado (ver la auditoría del escrutinio)."
            ),
            "Impacto": "El modelo acertaría por información que no existe al momento de predecir.",
            "Decisión": "Excluirlas como variables de entrada del modelo.",
            "Estado": "Mitigado",
        },
        {
            "ID": "H-09",
            "Hallazgo": "Variables sin variación",
            "Severidad": "Baja",
            "Evidencia": "Constantes: " + (", ".join(f"`{c}`" for c in a["constantes"]) or "ninguna") + ".",
            "Impacto": "No aportan información a ningún modelo.",
            "Decisión": "Conservar en el original para trazabilidad y no usarlas como predictoras.",
            "Estado": "Mitigado",
        },
    ]


# ============================================================
# NOVEDADES: PÁGINA
# ============================================================

def render_novedades():
    if not st.session_state.get("e3_activa_novedades", False):
        st.markdown(
            "Cada fila del archivo de novedades es una mesa nacional, con los conteos digitados de "
            "los formularios E-11 y E-14 y las novedades registradas (tachaduras, recuentos, sobres "
            "en mal estado, mesas excluidas). La auditoría reproduce el diagnóstico y la limpieza "
            "del notebook."
        )
        if st.button("Ejecutar auditoría de novedades", type="primary"):
            st.session_state.e3_activa_novedades = True
            st.rerun()
        st.info("El archivo se lee solo cuando lo solicitas, para no bloquear las demás entregas.")
        return

    try:
        with st.spinner("Leyendo el archivo de novedades y evaluando las reglas..."):
            a = auditar_novedades()
            h = huella_recurso(*a["info"]["recurso"])
    except FileNotFoundError as error:
        st.error(str(error))
        ayuda_ubicacion(ARCHIVO_NOVEDADES)
        return

    reglas = evaluar_reglas_novedades(a)
    hallazgos = registrar_hallazgos_novedades(a)
    veredicto, texto = dictamen(reglas, observaciones=TEXTO_OBSERVACIONES)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mesas", numero(a["filas"]))
    c2.metric("Columnas", a["columnas"])
    c3.metric("Mesas con alguna novedad", numero(a["mesas_con_novedad"]))
    c4.metric("Filas tras la limpieza", numero(a["filas_limpias"]))

    vista = st.radio("Vista", VISTAS, horizontal=True, key="e3_vista_novedades")
    st.divider()

    if vista == "Resumen ejecutivo":
        nov_resumen(a, h, reglas, hallazgos, veredicto, texto)
    elif vista == "Reglas de calidad":
        vista_reglas(reglas, clave="e3_filtro_reglas_novedades")
    elif vista == "Registro de hallazgos":
        vista_hallazgos(hallazgos)
    elif vista == "Preguntas de la plantilla":
        apartado = selector_apartado("novedades", "Novedades")
        {
            "1.1": nov_faltantes, "1.2": nov_duplicados, "1.3": nov_atipicos,
            "1.4": nov_inconsistencias, "1.5": nov_corruptos, "1.6": nov_seleccion,
        }[apartado.split(" ")[0]](a)
    else:
        nov_linaje(a, h, reglas)

    st.divider()
    st.caption(
        "Resultados metodológicos del notebook del equipo; indicadores recalculados sobre el "
        "archivo local. La detección estadística no constituye una auditoría electoral."
    )


def nov_resumen(a, h, reglas, hallazgos, veredicto, texto):
    abiertos = bloque_dictamen(reglas, hallazgos, veredicto, texto,
                               ("Mesas excluidas por IQR", numero(a["excluidas_iqr"])))
    izquierda, derecha = st.columns([2, 3])
    with izquierda:
        st.markdown("**Conformidad por dimensión**")
        st.dataframe(resumen_por_dimension(reglas), hide_index=True, use_container_width=True)
        st.markdown("**Hallazgos abiertos**")
        for hz in abiertos:
            st.markdown(f"- **{hz['ID']} · {hz['Hallazgo']}.** {hz['Decisión']}")
    with derecha:
        conteo = pd.DataFrame({"Novedad": list(a["novedades"]), "Mesas": list(a["novedades"].values())})
        conteo["Etiqueta"] = [f"{numero(n)} ({porcentaje(100 * n / a['filas'], 2)})" for n in conteo["Mesas"]]
        grafico(px.bar(conteo.sort_values("Mesas"), x="Mesas", y="Novedad", orientation="h",
                       text="Etiqueta", title=f"Mesas con cada novedad (de {numero(a['filas'])})"),
                alto=380)

    filas = [
        ("Archivo", h["nombre"]), ("Contenido en", h["contenedor"] or "—"),
        ("SHA-256", f"`{h['sha256']}`"), ("Tamaño", f"{numero(h['bytes'])} bytes"),
        ("Mesas", numero(a["filas"])), ("Columnas", a["columnas"]),
        ("Mesas tras la limpieza", numero(a["filas_limpias"])),
    ]
    informe = informe_fuente_markdown("novedades presidenciales 2026", filas, veredicto, texto,
                                      reglas, hallazgos, ALCANCE_NOVEDADES)
    descargas("novedades", informe, reglas, hallazgos, extra=(
        "Mesas excluidas por IQR (CSV)", a["atipicos_csv"], "novedades_atipicos.csv",
    ))


def nov_linaje(a, h, reglas):
    st.subheader("Linaje y reproducibilidad")
    st.markdown(
        "El diagrama muestra el recorrido desde el archivo original hasta `df_clean`. Los nodos "
        "grises son archivos y verificaciones; los azules, tablas derivadas."
    )
    origen = f"{h['contenedor']}\\n› {h['nombre']}" if h["contenedor"] else h["nombre"]
    b = a["banderas_despues"]
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#EEF1F4", color="#8D99AE"];
  edge [color="#5C677D"];
  original  [label="{origen}\\nCSV en UTF-8"];
  df        [label="df (texto)\\n{numero(a['filas'])} × {a['columnas']}\\n{numero(a['info']['reparadas'])} líneas reparadas"];
  reglas    [label="Motor de reglas\\n{len(reglas)} reglas"];
  precon    [label="Preconteo y DIVIPOL\\nintegridad referencial"];
  normal    [label="Texto normalizado\\ny tipos (Int64, booleanos)", fillcolor="#DCEAF6", color="#4A78A8"];
  banderas  [label="Banderas de inconsistencia\\n{numero(b['urna_difiere_e11'])} · {numero(b['total_mayor_urna'])} · {numero(b['total_mayor_e14'])}", fillcolor="#DCEAF6", color="#4A78A8"];
  clean     [label="df_clean\\n{numero(a['filas_limpias'])} × {a['columnas_limpias']}", fillcolor="#DCEAF6", color="#4A78A8"];
  atipicos  [label="df_atipicos\\n{numero(a['excluidas_iqr'])} mesas (IQR)", fillcolor="#DCEAF6", color="#4A78A8"];
  original -> df [label="reunir comas de puestoNombre"];
  df -> reglas; precon -> reglas;
  df -> normal -> banderas;
  banderas -> clean [label="dentro del IQR"];
  banderas -> atipicos [label="fuera del IQR"];
}}
""")
    st.markdown("**Huella del archivo auditado**")
    tabla_huella(h)
    st.caption(
        "Si el paquete de la Registraduría trae una huella oficial para este archivo, compárala con "
        "estos valores para verificar que no fue alterado."
    )
    tabla_entorno(f"CSV separado por comas, con encabezado de {a['info']['campos']} columnas")


# ------------------------------------------------------------
# Novedades: secciones de la plantilla
# ------------------------------------------------------------

def nov_faltantes(a):
    st.subheader("1.1 · Valores faltantes")
    criticas = ["sufragantesSegunE11", "sufragantesE11SegunE14", "votosUrna", "totalVotos"]
    faltan_criticas = sum(a["faltantes"][c] + a["no_digitos"][c] for c in criticas)
    texto_decision = (
        "Las variables críticas para el proyecto (E-11, E-14, votos en urna y total de votos) están "
        "completas, así que no se imputó ni eliminó ningún registro. **Método de relleno:** no aplica."
        if faltan_criticas == 0 else
        f"Las variables críticas (E-11, E-14, votos en urna y total de votos) tienen "
        f"{numero(faltan_criticas)} valores faltantes o no numéricos que deben revisarse antes de "
        "decidir si se imputan o se excluyen."
    )
    x, y, z = st.columns(3)
    x.metric("Nulos y cadenas vacías", numero(sum(a["faltantes"].values())))
    y.metric("Marcadores de dato ausente", numero(a["marcadores"]))
    z.metric("Conteos no numéricos", numero(sum(a["no_digitos"].values())))

    t = pd.DataFrame({"Variable": list(a["faltantes"]), "Faltantes": list(a["faltantes"].values())})
    t["Completitud (%)"] = (1 - t["Faltantes"] / a["filas"]) * 100
    grafico(px.bar(t, x="Variable", y="Completitud (%)", range_y=[0, 101],
                   title="Completitud por variable (nulos y cadenas vacías)"))

    explicacion(f"""
**Identificación.** Se revisaron las {a['columnas']} columnas buscando nulos y cadenas vacías
(**{numero(sum(a['faltantes'].values()))}**). Como el archivo se lee todo como texto, `isna()` solo
detecta lo que pandas convierte en nulo, así que también se buscaron marcadores habituales de dato
ausente (`-`, `NA`, `S/D`, `SIN DATO`...): **{numero(a['marcadores'])}**. Además, las ocho columnas
de conteo contienen solo dígitos ({numero(sum(a['no_digitos'].values()))} excepciones), por lo que la
conversión a número no oculta faltantes.

**Impacto y decisión.** {texto_decision}

**Cobertura.** El archivo solo trae mesas nacionales: {numero(a['consulados_preconteo'])} mesas de
consulados del preconteo no tienen registro de novedades. Ese faltante aparece al unir fuentes y se
trata en la auditoría del escrutinio, donde se decide excluir los consulados del conjunto de
modelado.
""")


def nov_duplicados(a):
    st.subheader("1.2 · Valores duplicados")
    x, y, z, w = st.columns(4)
    x.metric("Filas idénticas", numero(a["dup_exactos"]))
    y.metric("id repetidos", numero(a["dup_id"]))
    z.metric("Mesas repetidas", numero(a["dup_mesa"]))
    w.metric("Filas iguales sin id", numero(a["dup_sin_id"]))
    st.markdown("**Llave de la mesa:** " + ", ".join(f"`{c}`" for c in CLAVE_MESA_NOV))

    explicacion(f"""
**Detección.** Se buscaron duplicados en cuatro niveles: filas idénticas, `id` repetidos, la misma
mesa (departamento, municipio, zona, puesto y número de mesa) más de una vez, y filas iguales aun
ignorando el `id`, que detectaría una mesa copiada con otro identificador. En todos los casos el
resultado es **0**: cada fila es una mesa única.

**Consideración especial.** El número de mesa solo es único dentro de su puesto, por eso la
verificación usa la llave completa. Una llave incompleta marcaría como duplicadas mesas distintas.

**Decisión.** No se eliminó ningún registro. La unicidad por mesa es la que permite unir novedades
con el preconteo y el escrutinio uno a uno, sin repetir votos.
""")


def nov_atipicos(a):
    st.subheader("1.3 · Valores atípicos")
    st.dataframe(a["iqr"].round(1), hide_index=True, use_container_width=True)
    grafico(px.bar(a["iqr_lados"], x="Mesas", y="Columna", color="Lado", barmode="group",
                   orientation="h", text="Mesas",
                   color_discrete_map={"Por debajo del límite inferior": "#2A78D6",
                                       "Por encima del límite superior": "#EB6834"},
                   title="Mesas fuera del rango IQR, según el lado del límite"), alto=380)
    pot = pd.DataFrame({"Potencial": [str(k) for k in a["potencial"]], "Mesas": list(a["potencial"].values())})
    st.markdown("**Potencial máximo de las mesas**")
    st.dataframe(pot, hide_index=True, use_container_width=True)

    explicacion(f"""
**Criterio.** Se usa el rango intercuartílico: son atípicos los valores fuera de
[Q1 − 1,5·IQR, Q3 + 1,5·IQR]. Se aplica a las cuatro columnas de votantes, que tienen los mismos
límites: de **{numero(a['lim_inf'])} a {numero(a['lim_sup'])} votos** en el E-11. Considerando
las cuatro columnas, {numero(a['iqr_debajo'])} mesas quedan por debajo del límite inferior (mesas
con muy pocos votantes, entre ellas las excluidas, con 0 votos) y {numero(a['iqr_encima'])} por
encima, donde están los máximos de cada columna (tabla).

**Columnas sin IQR.** A `potencial` y `votosIncinerados` no se les aplica: su IQR es 0 porque casi
todas las mesas tienen el mismo valor, y marcaría como atípicas {numero(a['potencial_no_360'])} y
{numero(a['iqr_incinerados'])} mesas legítimas.

**Decisión.** Se excluyen de `df_clean` las mesas fuera del IQR en al menos una de las cuatro
columnas (**{numero(a['excluidas_iqr'])}, {porcentaje(a['pct_excluidas_iqr'], 2)}**) y se guardan
en `df_atipicos`, descargable desde el resumen. Es un criterio único y reproducible, pero tiene
límites: no distingue por `potencial` y elimina novedades poco frecuentes (hallazgo H-05). Ver
1.6.
""")


def nov_inconsistencias(a):
    st.subheader("1.4 · Datos inconsistentes")
    c1, c2, c3 = st.columns(3)
    c1.metric("id que no coinciden con sus códigos", numero(a["id_distinto"]))
    c2.metric("Códigos con más de un nombre", numero(sum(a["jerarquia"].values())))
    c3.metric("Filas con problemas de formato en puestoNombre", numero(a["puesto_con_problema"]))
    if len(a["formato"]):
        st.markdown("**Problemas de formato en las columnas de texto**")
        st.dataframe(a["formato"], hide_index=True, use_container_width=True)

    st.markdown("**Diferencias entre los conteos de votantes (número de mesas)**")
    st.dataframe(a["diferencias"], hide_index=True, use_container_width=True)

    coincidencia = a["coincidencia"]
    fig = px.imshow(coincidencia, text_auto=".1f", color_continuous_scale="Blues", zmin=90, zmax=100,
                    labels=dict(color="% de mesas"),
                    title=f"% de mesas donde las variables son iguales ({numero(a['mesas_comparadas'])} mesas no excluidas)")
    grafico(fig, alto=430)

    dif = a["dif_e11"].melt(id_vars="Variable", var_name="Diferencia", value_name="% de mesas")
    grafico(px.bar(dif, x="% de mesas", y="Variable", color="Diferencia", orientation="h",
                   color_discrete_sequence=["#2A78D6", "#86B6EF", "#EB6834"],
                   title="Diferencia frente a sufragantesSegunE11"), alto=320)

    st.markdown("**Coherencia de las banderas de novedad**")
    st.dataframe(pd.DataFrame({
        "Verificación": [
            "excluida = SI sin totalVotos = 0 (o al revés)", "Fuera de términos",
            "Fuera de términos y excluidas", "Recontadas por jurados y por comisión",
            "Mesas con 0 firmas de jurados", "Mesas no excluidas con 0 sufragantes",
        ],
        "Mesas": [a["excluida_vs_cero"], a["fuera_terminos"], a["fuera_terminos_excluidas"],
                  a["recontadas_ambas"], a["sin_firmas"], a["no_excluidas_sin_votantes"]],
    }), hide_index=True, use_container_width=True)

    ex = a["excesos"]
    dif_min, dif_max = a["dif_e11"]["Coinciden"].min(), a["dif_e11"]["Coinciden"].max()
    columnas_fmt = a["formato"]["Columna"].tolist()
    texto_formato = (
        "Ninguna columna de texto tiene espacios sobrantes ni minúsculas." if not columnas_fmt else
        "Tienen espacios sobrantes o minúsculas: " + ", ".join(f"`{c}`" for c in columnas_fmt)
        + ". Se normalizan en 1.6."
    )
    texto_excluida = (
        "Las banderas de novedad son coherentes con los conteos: `excluida = SI` equivale a "
        "`totalVotos = 0`." if a["excluida_vs_cero"] == 0 else
        f"En {numero(a['excluida_vs_cero'])} mesas `excluida` no concuerda con `totalVotos = 0`."
    )
    explicacion(f"""
**Identificador y jerarquía.** El `id` se compone de un prefijo `01` más los códigos de la mesa:
coincide en todas las filas salvo {numero(a['id_distinto'])}. La corporación es constante
({', '.join(a['corporaciones'])}) y {numero(sum(a['jerarquia'].values()))} códigos territoriales
tienen más de un nombre.

**Formato del texto.** {texto_formato}

**Consistencia entre conteos.** Se asume que el E-11, los votos en urna y el E-14 describen la
participación de la misma mesa y deberían ser iguales o muy cercanos, y que `totalVotos` no debería
superarlos. `votosUrna` es la que más se aleja ({numero(a['urna_distinta_e11'])} mesas distintas
del E-11). `totalVotos` supera a `votosUrna` en {numero(ex['totalVotos > votosUrna'])} mesas, al
E-14 en {numero(ex['totalVotos > sufragantesE11SegunE14'])} y al E-11 en
{numero(ex['totalVotos > sufragantesSegunE11'])}. Frente al E-11, las otras tres variables coinciden
exactamente en entre el {porcentaje(dif_min)} y el {porcentaje(dif_max)} de las mesas no excluidas
(gráficos).

**Estrategia.** Como una diferencia puede deberse a un error en cualquiera de las columnas, **no
se corrige ningún valor**: se marcan con banderas (hallazgo H-03). {texto_excluida} El supuesto de
igualdad entre conteos falta contrastarlo con la documentación de los formularios.
""")


def nov_corruptos(a):
    st.subheader("1.5 · Datos corruptos")
    info = a["info"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Líneas de datos", numero(info["lineas"]))
    c2.metric("Líneas reparadas", numero(info["reparadas"]))
    c3.metric("Líneas descartadas", numero(info["malformadas"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("puestoCodigo alfanumérico", numero(a["codigos_no_numericos"]["puestoCodigo"]))
    c2.metric("E-11 igual al potencial", numero(a["e11_igual_potencial"]))
    c3.metric("Máximo de votosUrna (texto / número)", f"{a['max_urna_texto']} / {numero(a['max_urna_numero'])}")

    st.markdown("**Códigos no numéricos por columna**")
    st.dataframe(pd.DataFrame({"Código": list(a["codigos_no_numericos"]),
                               "Valores no numéricos": list(a["codigos_no_numericos"].values())}),
                 hide_index=True, use_container_width=True)
    if a["ejemplos_puesto_alfa"]:
        st.caption("Ejemplos de puestoCodigo: " + ", ".join(f"`{c}`" for c in a["ejemplos_puesto_alfa"]))

    explicacion(f"""
**Detección.** Se revisaron tres fuentes de corrupción. (1) **Errores de carga:**
{numero(info['reparadas'])} líneas tienen comas sin entrecomillar dentro de `puestoNombre`, lo que
produce más campos que el encabezado. No son líneas dañadas: al reunir los campos sobrantes, todas
quedan con {info['campos']} columnas ({numero(info['malformadas'])} descartadas). El archivo se lee
en UTF-8, su codificación real; leerlo en latin-1 alteraba los nombres con tildes y eñes (hallazgo
H-01). (2) **Códigos con letras:** `puestoCodigo` tiene valores como
{', '.join(f'`{c}`' for c in a['ejemplos_puesto_alfa'][:3]) or '—'}, que son códigos válidos del
catálogo; por eso los códigos se mantienen como texto. (3) **Valores digitados no confiables:**
{numero(a['e11_igual_potencial'])} mesas tienen el E-11 igual al potencial máximo, lo que sugiere
que se digitó el potencial en lugar de los votantes (hallazgo H-04).

**Medidas.** Las líneas con comas se reparan al leer; los conteos pasan a enteros (como texto, el
máximo de `votosUrna` sería {a['max_urna_texto']} en lugar de {numero(a['max_urna_numero'])}); y los
E-11 sospechosos se marcan con `e11_sospechoso` en lugar de corregirse.

**Casos no manejables.** El valor real del E-11 de esas mesas no se puede recuperar sin el
formulario físico, y estimarlo con el E-14 o el escrutinio haría coincidir artificialmente las
fuentes que el modelo compara.
""")


def nov_seleccion(a):
    st.subheader("1.6 · Selección de datos")
    st.caption("En el notebook, este contenido corresponde a los apartados 1.5 (transformaciones) "
               "y 1.6 (documentación de las decisiones) de Novedades.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas originales", numero(a["filas"]))
    c2.metric("Filas en df_clean", numero(a["filas_limpias"]))
    c3.metric("Excluidas por IQR", f"{numero(a['excluidas_iqr'])} ({porcentaje(a['pct_excluidas_iqr'], 2)})")

    antes, despues = a["banderas_antes"], a["banderas_despues"]
    tratamientos = pd.DataFrame({
        "Tratamiento": list(antes) + ["Excluidas por IQR"],
        "Mesas": [despues[b] for b in antes] + [a["excluidas_iqr"]],
        "Tipo": ["Marcadas con bandera (dentro de df_clean)"] * len(antes) + ["Excluidas de df_clean"],
    })
    grafico(px.bar(tratamientos, x="Mesas", y="Tratamiento", color="Tipo", orientation="h", text="Mesas",
                   color_discrete_map={"Marcadas con bandera (dentro de df_clean)": "#2A78D6",
                                       "Excluidas de df_clean": "#EB6834"},
                   title="Mesas afectadas por cada tratamiento de la limpieza"), alto=320)

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("**Novedades que se pierden con la exclusión por IQR**")
        st.dataframe(a["novedades_perdidas"], hide_index=True, use_container_width=True)
    with derecha:
        st.markdown("**Comparación antes y después**")
        st.dataframe(pd.DataFrame({
            "Conjunto": ["df (original)", "df_clean"],
            "Filas": [a["filas"], a["filas_limpias"]],
            "Columnas": [a["columnas"], a["columnas_limpias"]],
            "Nulos": [sum(a["faltantes"].values()), a["nulos_limpios"]],
            "Máx. votosUrna": [a["max_urna_numero"], a["max_urna_limpio"]],
        }), hide_index=True, use_container_width=True)

    with st.expander("Tipos de datos de df_clean"):
        st.dataframe(a["tipos"], hide_index=True, use_container_width=True)
    st.download_button("Descargar df_clean (CSV)", a["limpio_csv"], file_name="novedades_df_clean.csv",
                       mime="text/csv")

    st.markdown("**Registro de decisiones**")
    st.dataframe(pd.DataFrame([
        ("Comas sin comillas en puestoNombre", numero(a["info"]["reparadas"]),
         "Reunir los campos sobrantes al leer", "Descartar las filas (se perderían mesas)"),
        ("Valores faltantes", numero(sum(a["faltantes"].values())), "Ninguna", "—"),
        ("Duplicados (fila, id o mesa)", numero(a["dup_mesa"]), "Ninguna", "—"),
        ("Formato de puestoNombre", numero(a["puesto_con_problema"]),
         "Quitar espacios y pasar a mayúsculas", "Dejarlo igual"),
        ("Conteos y SI/NO como texto", "todas",
         "Conteos a Int64, SI/NO a booleano, códigos como texto", "Convertir los códigos a entero"),
        ("Atípicos por IQR en las 4 columnas de votantes", numero(a["excluidas_iqr"]),
         "Excluir de df_clean y guardar en df_atipicos", "Marcar sin excluir"),
        ("potencial y votosIncinerados", "—", "No aplicarles IQR", "Aplicar IQR a todo"),
        (f"votosUrna difiere del E-11 en más de {UMBRAL_URNA_E11}",
         f"{numero(antes['urna_difiere_e11'])} ({numero(despues['urna_difiere_e11'])} tras excluir)",
         "Bandera urna_difiere_e11", "Imputar o anular un conteo"),
        ("totalVotos > votosUrna o > E-14",
         f"{numero(antes['total_mayor_urna'])} / {numero(antes['total_mayor_e14'])}",
         "Banderas total_mayor_urna y total_mayor_e14", "Corregir totalVotos"),
        ("Mesas con alguna novedad", numero(a["mesas_con_novedad"]),
         f"Se conservan, salvo {numero(a['novedad_excluida_iqr'])} que caen en el IQR",
         "Proteger esas mesas de la exclusión"),
    ], columns=["Hallazgo", "Filas", "Decisión", "Alternativa descartada"]),
        hide_index=True, use_container_width=True)

    explicacion(f"""
**Criterios.** Se trabaja sobre una copia (`df_clean`) y el original queda intacto para comparar.
Los atípicos se excluyen con un único criterio (IQR sobre las cuatro columnas de votantes) y las
inconsistencias entre conteos no modifican valores: solo se marcan. No se aplicó muestreo.

**Transformaciones.** (1) `puestoNombre` sin espacios sobrantes y en mayúsculas: quedan
{numero(a['formato_restante'])} problemas de formato y {numero(a['puestos_varios_nombres'])}
puestos con más de un nombre. (2) Conteos a `Int64`, banderas SI/NO a booleano y `estadoSobre` a
categoría; los códigos siguen como texto. (3) Tres banderas de inconsistencia. (4) Exclusión de
{numero(a['excluidas_iqr'])} mesas por IQR, guardadas en `df_atipicos`. Resultado:
**{numero(a['filas_limpias'])} mesas y {a['columnas_limpias']} columnas**, sin nulos e `id` único.

**Variables.** Para el modelado, de este archivo se usan los conteos disponibles antes del
escrutinio (E-11, E-14, votos en urna, firmas) y las banderas de calidad. Se excluyen como
entradas `recontadaComision`, `excluida`, `estadoSobre`, `enTerminos` y `totalVotos`, porque se
registran durante el escrutinio o son su resultado (hallazgo H-08), y las variables constantes
({', '.join(f'`{c}`' for c in a['constantes']) or 'ninguna'}).

**Supuestos y limitaciones.** El IQR no distingue por `potencial`, así que una mesa grande puede
quedar excluida sin ser un error; los atípicos bajos no se verificaron como errores o participación
baja real; el umbral de {UMBRAL_URNA_E11} votos es arbitrario; y la exclusión elimina
{numero(a['novedad_excluida_iqr'])} mesas con novedades.
""")