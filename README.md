# ml-final-project — Análisis de resultados electorales presidenciales Colombia 2026 (primera vuelta)

Análisis exploratorio, geoespacial y de aprendizaje automático sobre los resultados de la primera vuelta de las elecciones presidenciales de Colombia 2026, integrados con la división político-administrativa del país (departamentos y municipios).

## Descripción

Este proyecto académico estudia los resultados electorales de la primera vuelta presidencial de 2026 a partir del archivo `MMV_Presidente1V_2026`, y los articula con información geográfica de referencia (límites departamentales y municipales de Colombia). El propósito es caracterizar la distribución territorial del voto, identificar patrones espaciales y explorar, en etapas posteriores, técnicas de aprendizaje automático (por ejemplo, agrupamiento de territorios con comportamientos electorales similares).

La premisa del trabajo es que el resultado agregado nacional oculta una heterogeneidad territorial considerable. El análisis desagregado —por departamento, municipio y, cuando el dato lo permita, por unidades más finas— permite describir con mayor precisión la estructura del comportamiento electoral, la participación y su relación con el territorio.

El proyecto sigue la filosofía CRISP-DM, priorizando el entendimiento del negocio y del dato antes de avanzar hacia el modelado. El enfoque es descriptivo y analítico; no tiene fines de pronóstico electoral ni de valoración de candidaturas.

## Objetivos

### Objetivo general

Caracterizar los resultados de la primera vuelta presidencial de Colombia 2026 a nivel territorial e integrar esta información con capas geográficas oficiales, de modo que sirva como base reproducible para análisis exploratorio, visualización geoespacial y modelado con técnicas de aprendizaje automático.

### Objetivos específicos

1. Comprender el contexto del proceso electoral y definir las preguntas analíticas del proyecto.
2. Revisar la estructura, granularidad y variables del archivo de resultados `MMV_Presidente1V_2026`.
3. Integrar los resultados electorales con las capas geográficas de departamentos y municipios mediante claves territoriales consistentes.
4. Evaluar la calidad de los datos: valores faltantes, inconsistencias de codificación territorial y registros duplicados.
5. Describir la distribución territorial de votos, participación y votos especiales (blancos, nulos, no marcados), según disponibilidad.
6. Producir visualizaciones geoespaciales (mapas coropléticos) y tablas resumen reproducibles.
7. Explorar modelos de aprendizaje automático no supervisado y/o supervisado sobre las variables territoriales construidas.

## Fuente de datos

### Resultados electorales

- Archivo: `MMV_Presidente1V_2026.zip`, ubicado en `data/raw/elecciones_2026/`.
- Contenido: resultados de la primera vuelta presidencial 2026. La versión descomprimida se encuentra en `data/raw/elecciones_2026/extracted/`.
- Fuente institucional: _por documentar (p. ej., Registraduría Nacional del Estado Civil), incluyendo URL y fecha de descarga._

### Información geográfica

| Archivo | Formato | Nivel territorial |
|---|---|---|
| `colombia_departamentos.geojson` | GeoJSON | Departamentos |
| `colombia-municipios.topojson` | TopoJSON | Municipios |

Ambos archivos se encuentran en `data/external/geo/`. _Fuente y licencia por documentar._

### Distinción entre datos electorales y geográficos

1. **Datos electorales (tabulares)**
   - Conteos de votos por candidatura y unidad territorial.
   - Variables de identificación territorial (códigos de departamento y municipio, entre otras).

2. **Datos geográficos (vectoriales)**
   - Geometrías poligonales de departamentos y municipios.
   - Códigos y nombres que permiten el cruce con los datos electorales.

La integración de ambos niveles depende de una clave territorial común. Dado que las codificaciones electorales y las codificaciones geográficas (p. ej., DIVIPOLA) no siempre coinciden, la armonización de códigos es una tarea central de la preparación de datos.

## Metodología

El proyecto sigue CRISP-DM (Cross-Industry Standard Process for Data Mining).

1. **Entendimiento del negocio** — Contexto del proceso electoral, preguntas de investigación, alcance y criterios de éxito.
2. **Entendimiento de los datos** — Inspección del archivo de resultados, granularidad, variables, calidad y cobertura territorial.
3. **Preparación de los datos** — Limpieza, tipado, armonización de códigos territoriales, agregación por niveles y unión con las capas geográficas.
4. **Análisis exploratorio** — Distribuciones, participación, comparaciones territoriales y visualización geoespacial.
5. **Modelado** — Aplicación de técnicas de aprendizaje automático sobre las variables territoriales construidas.
6. **Evaluación** — Valoración de los modelos con métricas apropiadas e interpretación de resultados.

### Estado actual

| Fase | Estado |
|---|---|
| 1. Entendimiento del negocio | Notebook disponible |
| 2. Entendimiento de los datos | Notebook disponible |
| 3. Preparación de los datos | Pendiente |
| 4. Análisis exploratorio | Pendiente |
| 5. Modelado | Pendiente |
| 6. Evaluación | Pendiente |

## Estructura del repositorio

```text
ml-final-project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── app/
│   └── app.py
├── configs/
├── data/
│   ├── external/
│   │   └── geo/
│   │       ├── colombia-municipios.topojson
│   │       └── colombia_departamentos.geojson
│   ├── interim/
│   ├── processed/
│   └── raw/
│       └── elecciones_2026/
│           ├── MMV_Presidente1V_2026.zip
│           └── extracted/
├── docs/
│   └── data_dictionary/
├── experiments/
├── models/
│   ├── checkpoints/
│   └── final/
├── notebooks/
│   ├── 01_business_understanding/
│   ├── 02_data_understanding/
│   ├── 03_data_preparation/
│   ├── 04_exploratory_analysis/
│   ├── 05_modeling/
│   └── 06_evaluation/
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── reports/
│   └── tables/
├── scripts/
├── src/
│   └── ml_project/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       └── visualization/
└── tests/
```

### Propósito de las principales carpetas

| Directorio | Propósito |
|---|---|
| `app/` | Aplicación de presentación de resultados (`app.py`). |
| `configs/` | Archivos de configuración (rutas, parámetros de experimentos). |
| `data/raw/` | Datos originales sin modificar. No versionados. |
| `data/external/` | Datos de terceros, como las capas geográficas. No versionados. |
| `data/interim/` | Datos intermedios de la preparación. |
| `data/processed/` | Datos limpios e integrados, listos para análisis y modelado. |
| `docs/data_dictionary/` | Diccionario de datos con la descripción de variables. |
| `experiments/` | Registros y configuraciones de experimentos de modelado. |
| `models/` | Modelos entrenados: `checkpoints/` intermedios y `final/`. |
| `notebooks/` | Notebooks organizados por fase de CRISP-DM. |
| `results/` | Figuras, métricas, reportes y tablas generados. |
| `scripts/` | Scripts ejecutables de apoyo (descarga, procesamiento, utilidades). |
| `src/ml_project/` | Paquete Python instalable con el código reutilizable del proyecto. |
| `tests/` | Pruebas automatizadas con `pytest`. |

### Paquete `ml_project`

| Módulo | Responsabilidad prevista |
|---|---|
| `ml_project.data` | Carga, extracción y limpieza de datos electorales y geográficos. |
| `ml_project.features` | Construcción de variables (proporciones de voto, participación, indicadores territoriales). |
| `ml_project.models` | Definición y entrenamiento de modelos. |
| `ml_project.evaluation` | Métricas y validación de modelos. |
| `ml_project.visualization` | Gráficos y mapas. |

## Notebooks

| Notebook | Fase | Propósito |
|---|---|---|
| `01_business_understanding/BSYG_M2U2_fase_1_Entendimiento_del_negocio.ipynb` | Entendimiento del negocio | Contexto, preguntas analíticas, alcance y objetivos. |
| `02_data_understanding/Copy_of_M2U3_fase_2_Entendimiento_de_los_datos.ipynb` | Entendimiento de los datos | Exploración inicial del archivo de resultados y de su calidad. |

## Flujo de análisis de datos

```text
MMV_Presidente1V_2026.zip
→ extracción del archivo
→ carga y tipado de resultados
→ validación de calidad (faltantes, duplicados, códigos)
→ armonización de códigos territoriales
→ agregación por municipio y departamento
→ unión con capas geográficas (GeoJSON / TopoJSON)
→ construcción de variables (proporciones, participación)
→ análisis descriptivo y mapas coropléticos
→ modelado (aprendizaje automático)
→ evaluación e interpretación
```

## Resultados preliminares

_Sección por completar a medida que avancen las fases 2 a 6._

| Aspecto | Valor | Observación |
|---|---:|---|
| Registros en el archivo de resultados | _por documentar_ | |
| Nivel de granularidad | _por documentar_ | |
| Departamentos cubiertos | _por documentar_ | |
| Municipios cubiertos | _por documentar_ | |
| Registros sin correspondencia geográfica | _por documentar_ | |

Los hallazgos que se reporten serán descriptivos y no constituyen pronósticos electorales ni valoraciones sobre candidaturas.

## Análisis geoespacial

Las capas geográficas se cargan con `geopandas`. Las geometrías de departamentos (GeoJSON) y municipios (TopoJSON) se unen a los resultados agregados mediante el código territorial correspondiente.

Una medida básica del análisis es la proporción de votos de una candidatura $c$ en una unidad territorial $i$:

$$
p_{c,i} = \frac{v_{c,i}}{V_i}
$$

donde $v_{c,i}$ es el número de votos de la candidatura $c$ en la unidad $i$ y $V_i$ el total de votos válidos de esa unidad. De forma análoga, la participación puede expresarse como el cociente entre votos totales y potencial electoral, si este último está disponible en los datos.

Cuando se construyan mapas coropléticos, se recomienda usar proporciones y no conteos absolutos, para que las diferencias de población entre territorios no dominen la lectura del mapa.

## Gestión de datos

El repositorio separa explícitamente:

- `data/raw`: datos originales, sin modificar;
- `data/external`: datos de terceros (capas geográficas);
- `data/interim`: artefactos intermedios;
- `data/processed`: datos finales para análisis;
- `results/`: figuras, métricas, reportes y tablas;
- `models/`: modelos entrenados.

El archivo `.gitignore` excluye del control de versiones el contenido de `data/raw`, `data/interim`, `data/processed`, `data/external`, `models/` y `results/`, así como entornos virtuales y archivos `.env`. En consecuencia, quien clone el repositorio deberá obtener por su cuenta el archivo de resultados y las capas geográficas y ubicarlos en las rutas indicadas arriba.

## Instalación

Se recomienda usar un entorno virtual aislado. El proyecto requiere Python 3.10 o superior.

### macOS / Linux

```bash
git clone git@github.com:byepesg/ml-final-project.git
cd ml-final-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Windows (PowerShell)

```powershell
git clone git@github.com:byepesg/ml-final-project.git
cd ml-final-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

El comando `pip install -e .` instala el paquete `ml_project` en modo editable, de modo que puede importarse desde notebooks y scripts:

```python
from ml_project import data, features, visualization
```

### Variables de entorno

Si el proyecto requiere credenciales o rutas locales, deben definirse en un archivo `.env` a partir de la plantilla `.env.example`. El archivo `.env` no se versiona.

### Verificación del entorno

```bash
python --version
pip list
pytest
```

Para iniciar Jupyter:

```bash
jupyter lab
```

## Ejecución

Los notebooks se ejecutan en el orden de las fases de CRISP-DM:

1. `notebooks/01_business_understanding/`
2. `notebooks/02_data_understanding/`
3. `notebooks/03_data_preparation/`
4. `notebooks/04_exploratory_analysis/`
5. `notebooks/05_modeling/`
6. `notebooks/06_evaluation/`

La aplicación de resultados se ejecuta con:

```bash
python app/app.py
```

_Ajustar este comando según el framework que se use en `app.py`._

## Dependencias principales

| Librería | Rol principal |
|---|---|
| `pandas` | Manipulación y análisis de datos tabulares. |
| `numpy` | Operaciones numéricas vectorizadas. |
| `geopandas` | Lectura y operaciones con datos geoespaciales. |
| `shapely`, `pyproj`, `pyogrio` | Geometrías, proyecciones cartográficas y lectura de formatos vectoriales. |
| `matplotlib`, `seaborn` | Visualización estática. |
| `plotly` | Visualización interactiva y mapas. |
| `scipy` | Estadística y cálculo científico. |
| `scikit-learn` | Modelado con aprendizaje automático. |
| `pytest` | Pruebas automatizadas. |
| `jupyterlab`, `ipykernel` | Ejecución de notebooks. |

## Reproducibilidad

La reproducibilidad del proyecto se apoya en:

- entorno virtual Python y dependencias fijadas en `requirements.txt`;
- código reutilizable empaquetado en `src/ml_project` e instalado en modo editable;
- separación estricta entre datos originales, intermedios y procesados;
- notebooks organizados por fase de CRISP-DM;
- pruebas automatizadas en `tests/`;
- documentación de variables en `docs/data_dictionary/`.

## Equipo

_Por documentar: integrantes, roles y afiliación institucional._ FALTA AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

## Referencias

- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). _CRISP-DM 1.0: Step-by-step data mining guide_. SPSS Inc.
- _Fuente oficial de los resultados electorales: por documentar._
- _Fuente de las capas geográficas de Colombia: por documentar._

## Licencia y uso de datos

El repositorio no incluye todavía un archivo `LICENSE`, por lo que no puede afirmarse que el código se distribuya bajo una licencia específica. El uso de los resultados electorales y de las capas geográficas debe respetar los términos de sus fuentes originales; la licencia del código debe mantenerse separada de la licencia de los datos.

---

Este README refleja el estado actual del proyecto: un trabajo académico en las fases de entendimiento del negocio y de los datos, con la estructura preparada para la preparación, el modelado y la evaluación.
