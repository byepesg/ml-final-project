## Observatorio Electoral (Streamlit)

El observatorio es una aplicación de Streamlit que presenta los resultados del proyecto por
entregas. Su código está en `observatorio/` (`app.py`, `pages/` y `services/`).

### Requisitos

1. Python 3.12 y las dependencias del proyecto:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Los datos de la Registraduría extraídos en
   `data/raw/elecciones_2026/extracted/MMV_Presidente1V_2026/`, la misma ubicación que usan los
   notebooks. Los ZIP internos (novedades y archivos básicos) no hace falta descomprimirlos: el
   observatorio los lee directamente.

### Ejecución

Desde la raíz del repositorio, con el script:

```bash
./scripts/run_observatorio.sh
```

El script se ubica en la raíz del repositorio, activa `.venv` si existe y lanza la aplicación.
La primera vez puede ser necesario darle permisos de ejecución:

```bash
chmod +x scripts/run_observatorio.sh
```

O directamente con Streamlit (también funciona en Windows):

```bash
python -m streamlit run observatorio/app.py
```

La aplicación se abre en <http://localhost:8501>. Para usar otro puerto, agrega la opción al
final del comando, por ejemplo `./scripts/run_observatorio.sh --server.port 8600`. Para
detenerla, presiona `Ctrl+C` en la terminal.

### Notas

- Las auditorías de la Entrega 3 leen los archivos completos solo cuando se pulsa el botón de
  cada fuente. La primera ejecución puede tardar alrededor de un minuto; después los resultados
  quedan en caché mientras la aplicación siga abierta.
- Si el escrutinio o las novedades están en otra carpeta, define sus rutas en
  `observatorio/services/data_loader.py` con `ESCRUTINIO_PATH` y `NOVEDADES_PATH`.


## Observatorio Electoral (Streamlit)

El observatorio es una aplicación de Streamlit que presenta los resultados del proyecto por
entregas. Su código está en `observatorio/` (`app.py`, `pages/` y `services/`).

### Requisitos

1. Python 3.12 y las dependencias del proyecto:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Los datos de la Registraduría extraídos en
   `data/raw/elecciones_2026/extracted/MMV_Presidente1V_2026/`, la misma ubicación que usan los
   notebooks. Los ZIP internos (novedades y archivos básicos) no hace falta descomprimirlos: el
   observatorio los lee directamente.

### Ejecución

Desde la raíz del repositorio, con el script de tu sistema operativo:

| Sistema | Comando |
|---|---|
| macOS y Linux (Ubuntu) | `./scripts/run_observatorio.sh` |
| Windows (CMD o PowerShell) | `scripts\run_observatorio.bat` |
| Windows (Git Bash o WSL) | `./scripts/run_observatorio.sh` |

Los scripts se ubican en la raíz del repositorio, activan `.venv` si existe y lanzan la
aplicación. En macOS y Linux, la primera vez puede ser necesario darle permisos de ejecución:

```bash
chmod +x scripts/run_observatorio.sh
```

También se puede ejecutar directamente con Streamlit, en cualquier sistema:

```bash
python -m streamlit run observatorio/app.py
```

La aplicación se abre en <http://localhost:8501>. Para usar otro puerto, agrega la opción al
final del comando, por ejemplo `./scripts/run_observatorio.sh --server.port 8600`. Para
detenerla, presiona `Ctrl+C` en la terminal.

### Notas

- Las auditorías de la Entrega 3 leen los archivos completos solo cuando se pulsa el botón de
  cada fuente. La primera ejecución puede tardar alrededor de un minuto; después los resultados
  quedan en caché mientras la aplicación siga abierta.
- Si el escrutinio o las novedades están en otra carpeta, define sus rutas en
  `observatorio/services/data_loader.py` con `ESCRUTINIO_PATH` y `NOVEDADES_PATH`.