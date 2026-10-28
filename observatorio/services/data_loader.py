
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRECONTEO_PATH = (
    PROJECT_ROOT
    / "data/raw/elecciones_2026/extracted"
    / "MMV_Presidente1V_2026"
    / "MMV_Presidente1V_2026"
    / "PRE1v_MMV_9999_PRECONTEO.txt"
)

COLUMNAS = [
    "departamento",
    "municipio",
    "zona",
    "puesto",
    "mesa",
    "jal",
    "comunicado",
    "circunscripcion",
    "lista_partido",
    "candidato",
    "votos",
]

LONGITUDES = [2, 3, 2, 2, 6, 2, 4, 1, 5, 3, 8]


@st.cache_data(show_spinner="Leyendo preconteo electoral...")
def load_preconteo():
    if not PRECONTEO_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el preconteo: {PRECONTEO_PATH}"
        )

    limites = [0]
    for longitud in LONGITUDES:
        limites.append(limites[-1] + longitud)

    colspecs = list(zip(limites[:-1], limites[1:]))

    df = pd.read_fwf(
        PRECONTEO_PATH,
        colspecs=colspecs,
        names=COLUMNAS,
        dtype="string",
        header=None,
        keep_default_na=False,
    )

    df["votos"] = pd.to_numeric(
        df["votos"], errors="raise"
    ).astype("int64")

    return df


@st.cache_data(show_spinner="Calculando indicadores por mesa...")
def load_resumen_mesas():
    df = load_preconteo()

    columnas_mesa = [
        "departamento",
        "municipio",
        "zona",
        "puesto",
        "mesa",
    ]

    resumen = (
        df.groupby(columnas_mesa, dropna=False)
        .agg(
            registros=("votos", "size"),
            total_votos=("votos", "sum"),
            registros_cero=("votos", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    return resumen
