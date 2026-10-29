#!/usr/bin/env bash
# Ejecuta el Observatorio Electoral (Streamlit) desde la raíz del repositorio.
#
# Uso:
#   ./scripts/run_observatorio.sh                    # puerto por defecto (8501)
#   ./scripts/run_observatorio.sh --server.port 8600 # cualquier opción de streamlit run
set -euo pipefail

# Ir a la raíz del repositorio, sin importar desde dónde se llame el script.
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

# Activar el entorno virtual del proyecto si existe y no hay otro activo.
# En Mac y Linux el script está en .venv/bin; en Windows (Git Bash), en .venv/Scripts.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    for activar in ".venv/bin/activate" ".venv/Scripts/activate"; do
        if [[ -f "$activar" ]]; then
            # shellcheck disable=SC1090
            source "$activar"
            break
        fi
    done
fi

PYTHON="$(command -v python || command -v python3)"

if ! "$PYTHON" -c "import streamlit" 2>/dev/null; then
    echo "Streamlit no está instalado en este entorno."
    echo "Instala las dependencias con:  pip install -r requirements.txt"
    exit 1
fi

echo "Iniciando el observatorio (Ctrl+C para detenerlo)..."
exec "$PYTHON" -m streamlit run observatorio/app.py "$@"
