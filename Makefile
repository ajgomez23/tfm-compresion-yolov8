# Makefile del proyecto.
#
# Contiene objetivos de instalación, comprobación, formato y tests. En Windows hace falta GNU Make
# (por ejemplo el de Git for Windows o `conda install make`); el entorno de referencia es Linux.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
SRC_DIRS := src scripts tests

.DEFAULT_GOAL := help
.PHONY: help install install-dev check format format-check lint test test-cov clean

help:  ## Muestra esta ayuda
	@echo "Objetivos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Instala el paquete y sus dependencias ancladas
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:  ## Instala además las herramientas de desarrollo
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]"

check:  ## Comprueba el entorno e informa de qué componentes hay disponibles
	$(PYTHON) scripts/00_setup_check.py

format:  ## Formatea el código con black e isort
	$(PYTHON) -m isort $(SRC_DIRS)
	$(PYTHON) -m black $(SRC_DIRS)

format-check:  ## Verifica el formato sin modificar ficheros (para CI)
	$(PYTHON) -m isort --check-only --diff $(SRC_DIRS)
	$(PYTHON) -m black --check --diff $(SRC_DIRS)

lint: format-check  ## Alias de format-check

test:  ## Ejecuta la batería de tests
	$(PYTEST) tests/ -v

test-cov:  ## Ejecuta los tests con informe de cobertura
	$(PYTEST) tests/ -v --cov=tfm_edge --cov-report=term-missing

clean:  ## Elimina artefactos temporales de Python y de los tests
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ['.pytest_cache', '.coverage', 'htmlcov', 'build', 'dist']]"
	@echo "Limpieza completada. NO se ha tocado data/, models/ ni results/."
