
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Observatorio Electoral de Colombia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ENCABEZADO
# ============================================================

st.title("Observatorio Electoral de Colombia")

st.caption(
    "Proyecto académico de Machine Learning | "
    "Universidad Nacional de Colombia"
)

# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.title("Observatorio Electoral")

    st.markdown(
        "Análisis de datos electorales de Colombia, "
        "organizado por fases del proyecto."
    )

    st.divider()

    seccion = st.radio(
        "Navegación",
        [
            "Inicio",
            "Entrega 2",
            "Entrega 3",
          #  "Entrega 4",
           # "Entrega 5",
           # "Entrega 6",
        ],
        key="navegacion",
    )

    st.divider()

    st.caption(
        "Universidad Nacional de Colombia\n\n"
        "Maestría en Ingeniería de Sistemas y Computación"
    )

# ============================================================
# FUNCIÓN PARA EJECUTAR MÓDULOS
# ============================================================

def ejecutar_modulo(nombre, funcion):
    """
    Ejecuta un módulo y muestra errores sin
    interrumpir toda la aplicación.
    """
    try:
        funcion()
    except Exception as error:
        st.error(f"Error al cargar {nombre}")
        st.exception(error)


# ============================================================
# INICIO
# ============================================================

if seccion == "Inicio":

    st.header("Presentación del observatorio")

    st.markdown("""
    Este observatorio permite explorar los resultados
    del proyecto académico de Machine Learning
    aplicado a datos electorales de Colombia.

    Su desarrollo sigue progresivamente las fases
    metodológicas del proyecto, desde el entendimiento
    de los datos hasta la evaluación de modelos.

    ### Objetivos del observatorio

    - Visualizar características de los datos electorales.
    - Evaluar la calidad y consistencia de los registros.
    - Explorar distribuciones y patrones territoriales.
    - Documentar transformaciones de datos.
    - Presentar resultados de modelado y evaluación.
    """)

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**Entrega 2**\n\n"
            "Entendimiento de los datos"
        )

    with c2:
        st.info(
            "**Entrega 3**\n\n"
            "Preparación de los datos"
        )

    with c3:
        st.info(
            "**Entrega 4**\n\n"
            "Análisis exploratorio"
        )

    st.caption(
        "Los módulos se incorporan progresivamente "
        "conforme avanzan las entregas."
    )

# ============================================================
# ENTREGA 2
# ============================================================

elif seccion == "Entrega 2":

    from pages.entrega_2 import render

    ejecutar_modulo(
        "Entrega 2 — Entendimiento de datos",
        render,
    )

# ============================================================
# ENTREGA 3
# ============================================================

elif seccion == "Entrega 3":

    from pages.entrega_3 import render

    ejecutar_modulo(
        "Entrega 3 — Preparación de datos",
        render,
    )

# ============================================================
# ENTREGA 4
# ============================================================

elif seccion == "Entrega 4":

    from pages.entrega_4 import render

    ejecutar_modulo(
        "Entrega 4 — Análisis exploratorio",
        render,
    )

# ============================================================
# ENTREGA 5
# ============================================================

elif seccion == "Entrega 5":

    from pages.entrega_5 import render

    ejecutar_modulo(
        "Entrega 5 — Modelado",
        render,
    )

# ============================================================
# ENTREGA 6
# ============================================================

elif seccion == "Entrega 6":

    st.header("Entrega 6 — Evaluación")

    st.info(
        "Este módulo se implementará "
        "cuando estén disponibles los resultados "
        "de evaluación de los modelos."
    )

# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Observatorio Electoral de Colombia | "
    "Proyecto académico de Machine Learning | 2026"
)
