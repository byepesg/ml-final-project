"""Observatorio electoral — Entrega 3: preparación de los datos.

La entrega se presenta como un informe de auditoría de calidad de datos:

- Un motor de reglas declarativas evalúa el archivo en vivo (completitud,
  integridad física, validez, unicidad, consistencia, cobertura y plausibilidad).
- Un registro de hallazgos documenta cada problema con evidencia, decisión y estado.
- El dictamen ejecutivo se genera a partir del resultado de las reglas.
- La vista de linaje registra la huella SHA-256 del archivo y las transformaciones.
- El informe completo se puede descargar en Markdown y CSV.

Fuentes: el preconteo está implementado. Escrutinio y novedades tienen la misma
estructura vacía para completar (ver `render_pendiente`).
"""
import hashlib
import platform
from dataclasses import asdict, dataclass
from datetime import datetime

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

# Indicaciones y reglas sugeridas para quien complete cada fuente pendiente.
FUENTES_PENDIENTES = {
    "Escrutinio": {
        "archivo": "_ficheros_MMV_4_MMV_9999_ESCRUTINIO.csv",
        "pista": (
            "Archivo separado por `;`, sin encabezado, con 12 columnas: comunicado, dep, mun, "
            "zona, puesto, mesa, comuna, corporación, circ, partido, candidato y votos. "
            "Tiene 1.952.320 filas (122.020 mesas × 16 opciones), es decir, trae todas las "
            "opciones de cada mesa. Es el punto de comparación natural para el preconteo."
        ),
        "reglas_sugeridas": [
            ("Completitud", "Sin nulos ni cadenas vacías en las 12 columnas"),
            ("Unicidad", "Clave mesa × opción única"),
            ("Cobertura", "Cada mesa tiene exactamente 16 opciones"),
            ("Validez", "Votos enteros y no negativos"),
            ("Integridad referencial", "Todas las mesas existen en el preconteo y en DIVIPOL"),
            ("Consistencia entre fuentes", "Diferencia de votos frente al preconteo por mesa y opción"),
        ],
    },
    "Novedades": {
        "archivo": "Novedad20260603.csv",
        "pista": (
            "CSV en UTF-8 (no latin-1). 45 líneas tienen comas dentro del nombre del puesto; "
            "el notebook las repara con la función `leer_novedades`. Tiene 118.350 filas y "
            "columnas numéricas como potencial, sufragantes, votos en urna, votos incinerados "
            "y total de votos."
        ),
        "reglas_sugeridas": [
            ("Integridad física", "Todas las líneas tienen el número de campos del encabezado tras la reparación"),
            ("Validez", "Columnas numéricas convertibles sin pérdida"),
            ("Plausibilidad", "Sufragantes no superan el potencial electoral de la mesa"),
            ("Consistencia", "Votos en urna menos incinerados coincide con total de votos"),
            ("Integridad referencial", "Todas las mesas existen en el preconteo y en DIVIPOL"),
            ("Unicidad", "Una novedad por mesa y tipo, o duplicidad justificada"),
        ],
    },
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


def dictamen(reglas):
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
        texto += (
            "Las observaciones (" + ", ".join(r.id for r in otras_fallidas) + ") son de "
            "cobertura y plausibilidad; están documentadas en el registro de hallazgos con su "
            "tratamiento."
        )
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
    else:
        render_pendiente(fuente)


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


def vista_reglas(reglas):
    st.subheader("Reglas de calidad")
    st.markdown(
        "Cada verificación del notebook se expresa como una regla declarativa, con dimensión, "
        "severidad y valor esperado. Las reglas se evalúan en vivo sobre el archivo cargado; "
        "las informativas describen el archivo sin un umbral de aprobación."
    )

    tabla = tabla_reglas(reglas)
    estados = st.multiselect("Filtrar por estado", ["Cumple", "No cumple", "Informativa"],
                             default=["Cumple", "No cumple", "Informativa"], key="e3_filtro_reglas")
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
# FUENTES PENDIENTES (escrutinio y novedades)
# ============================================================
# Para completar una fuente:
#   1. Crear su cargador en services/data_loader.py (por ejemplo, load_escrutinio).
#   2. Escribir sus reglas con la función _regla(), como en evaluar_reglas().
#   3. Escribir sus hallazgos con la misma estructura de registrar_hallazgos().
#   4. Crear render_escrutinio() o render_novedades() siguiendo render_preconteo().
#   5. En render(), reemplazar render_pendiente(fuente) por la nueva función.

def render_pendiente(fuente):
    info = FUENTES_PENDIENTES[fuente]
    st.subheader(f"{fuente}: pendiente de completar")
    st.info(
        f"Esta sección está reservada para la auditoría del archivo `{info['archivo']}`. "
        "Sigue la misma estructura del preconteo: reglas, hallazgos, preguntas y linaje."
    )
    st.markdown(f"**Sobre el archivo.** {info['pista']}")

    vista = st.radio("Vista", VISTAS, horizontal=True, key=f"e3_vista_{fuente.lower()}")
    st.divider()

    if vista == "Resumen ejecutivo":
        st.warning(
            "Pendiente. El dictamen se genera automáticamente con la función `dictamen()` "
            "una vez estén definidas las reglas de esta fuente."
        )
    elif vista == "Reglas de calidad":
        st.warning("Pendiente. Reglas sugeridas para empezar:")
        st.dataframe(
            pd.DataFrame(info["reglas_sugeridas"], columns=["Dimensión", "Regla sugerida"]),
            hide_index=True, use_container_width=True,
        )
    elif vista == "Registro de hallazgos":
        st.warning(
            "Pendiente. Cada hallazgo debe tener ID, hallazgo, severidad, evidencia, impacto, "
            "decisión y estado (Abierto, Mitigado o Cerrado)."
        )
    elif vista == "Preguntas de la plantilla":
        apartado = st.selectbox("Apartado", SECCIONES, key=f"e3_apartado_{fuente.lower()}")
        clave = apartado.split(" ")[0]
        st.markdown(f"#### {apartado} · {fuente}")
        st.warning("Pendiente. Agrega aquí los indicadores, gráficos y respuestas de esta sección.")
        with st.expander("Preguntas que debe responder esta sección", expanded=True):
            st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(PREGUNTAS[clave], start=1)))
    else:
        st.warning(
            "Pendiente. Registra la huella SHA-256 del archivo y el recorrido de las "
            "transformaciones, como en el preconteo."
        )