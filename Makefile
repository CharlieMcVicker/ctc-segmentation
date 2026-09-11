# Makefile for CTC segmentation
# Original author: Ludwig Kürzinger, 2021

.PHONY: all build dist clean test install install-dev upload

PYTHON ?= python

all: build

build:
	$(PYTHON) setup.py build_ext --inplace

dist:
	$(PYTHON) setup.py sdist bdist_wheel

test:
	pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.so" -o -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
	rm -f ctc_segmentation/ctc_segmentation_dyn.c

install:
	pip install .

install-dev:
	pip install -e ".[test]"

upload:
	twine upload dist/*

