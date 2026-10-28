"""Entrega 2: entendimiento de datos electorales, para observatorio/app.py."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from services.data_loader import load_preconteo, load_resumen_mesas, PRECONTEO_PATH

COLUMNAS_MESA = ["departamento", "municipio", "zona", "puesto", "mesa"]
CLAVE = COLUMNAS_MESA + ["jal", "comunicado", "circunscripcion", "lista_partido", "candidato"]


def _grafico(fig):
    fig.update_layout(height=370, margin=dict(l=15, r=15, t=60, b=25))
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.header("Entrega 2 · Entendimiento de los datos")
    st.caption("CRISP-DM · Análisis exploratorio del archivo de preconteo presidencial 2026")
    st.markdown("Los resultados están organizados por **departamento, municipio, zona, puesto y mesa**. "
                "El objetivo es describir la fuente y su estructura antes de preparar los datos.")

    if not st.session_state.get("e2_activa", False):
        if st.button("Cargar Entrega 2", type="primary", key="e2_cargar"):
            st.session_state.e2_activa = True
            st.rerun()
        st.info("La lectura se realiza al solicitarla para no bloquear otras secciones.")
        return

    seccion = st.selectbox("Apartado", [
        "1.1 · Resumen general", "1.2 · Calidad de los datos",
        "1.3 · Tipos de variables", "1.4 · Relación entre variables",
    ], key="e2_seccion")

    with st.spinner("Cargando indicadores..."):
        df = load_preconteo()

    if seccion.startswith("1.1"):
        st.subheader("1.1 · Resumen general")
        ruta = Path(PRECONTEO_PATH)
        mb = ruta.stat().st_size / (1024 ** 2) if ruta.exists() else None
        a, b, c, d = st.columns(4)
        a.metric("Registros", f"{len(df):,}")
        b.metric("Variables", len(df.columns))
        c.metric("Archivos de datos analizados", "1")
        d.metric("Tamaño del archivo", f"{mb:,.1f} MiB" if mb is not None else "No disponible")
        st.markdown("**Respuesta académica.** La unidad de análisis es el registro de votos "
                    "para una opción electoral en una mesa. La fuente es un archivo de texto "
                    "plano de **ancho fijo (38 caracteres por registro)**; no es un CSV. "
                    "Su tamaño se calcula directamente del archivo local.")
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)
        st.caption("Vista previa de registros; los identificadores conservan sus ceros iniciales.")

    elif seccion.startswith("1.2"):
        st.subheader("1.2 · Calidad de los datos")
        nulos = int(df.isna().sum().sum())
        vacios = sum(int(df[c].str.strip().eq("").fillna(False).sum()) for c in df.columns
                     if pd.api.types.is_string_dtype(df[c]))
        duplicados = int(df.duplicated().sum())
        dup_clave = int(df.duplicated(subset=CLAVE).sum())
        a,b,c,d = st.columns(4)
        a.metric("Nulos", nulos)
        b.metric("Cadenas vacías", vacios)
        c.metric("Duplicados exactos", duplicados)
        d.metric("Duplicados por clave", dup_clave)
        st.markdown("**Respuesta académica.** La lectura del archivo permite verificar los tipos, "
                    "los valores vacíos y los duplicados. Los valores cero en `votos` son datos "
                    "válidos, no valores faltantes. La ausencia de filas para ciertas opciones "
                    "por mesa se analiza en la Entrega 3 porque puede depender del formato de transmisión. "
                    "La carga exitosa no equivale por sí sola a una auditoría de integridad de origen.")
        resumen = pd.DataFrame({"Variable":df.columns,
                                "Nulos":[int(df[c].isna().sum()) for c in df.columns],
                                "Valores únicos":[int(df[c].nunique()) for c in df.columns]})
        st.dataframe(resumen, hide_index=True, use_container_width=True)

    elif seccion.startswith("1.3"):
        st.subheader("1.3 · Tipos de variables")
        a,b,c = st.columns(3)
        a.metric("Variables", len(df.columns))
        b.metric("Códigos / identificadores", len(df.columns)-1)
        c.metric("Variable cuantitativa", "votos")
        st.markdown("**Respuesta académica.** Los identificadores territoriales y electorales "
                    "son **categóricos**, aunque contengan dígitos. `votos` es una variable "
                    "**cuantitativa discreta** (conteo). Este archivo no define por sí solo "
                    "una etiqueta supervisada: la variable objetivo depende de la pregunta "
                    "de investigación y no debe confundirse automáticamente con `votos`.")
        opciones = (df.groupby(["lista_partido", "candidato"], dropna=False)["votos"]
                      .sum().reset_index().sort_values("votos", ascending=False))
        opciones["Opción"] = opciones["lista_partido"] + " / " + opciones["candidato"]
        _grafico(px.bar(opciones, x="Opción", y="votos", title="Votos agregados por opción electoral"))
        st.caption("Los códigos de opción no se traducen a nombres de candidaturas sin un catálogo verificado.")

    else:
        st.subheader("1.4 · Relación entre variables")
        st.markdown("**Respuesta académica.** La relación más relevante aquí es jerárquica: "
                    "departamentos contienen municipios, que contienen zonas, puestos y mesas. "
                    "No es apropiado calcular correlaciones de Pearson entre códigos geográficos "
                    "como si fueran magnitudes continuas. Es más informativo comparar distribuciones "
                    "de votos y cobertura por agrupaciones territoriales.")
        dep = (df.groupby("departamento", dropna=False)
                 .agg(registros=("votos", "size"), votos=("votos", "sum"))
                 .reset_index().sort_values("votos", ascending=False))
        _grafico(px.bar(dep, x="departamento", y="votos", title="Votos registrados por código de departamento"))
        st.dataframe(dep, hide_index=True, use_container_width=True)
        st.info("Los totales absolutos no son tasas de participación; no se dispone aquí "
                "del censo electoral por territorio para normalizarlos.")

    st.divider()
    st.caption("Fuente: archivo local de preconteo. Análisis descriptivo, no auditoría electoral.")
