#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fcl_psp
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
SRC = src

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install the package (editable) with dev extras
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev]"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, isort, and black (matches the CI lint stage)
.PHONY: lint
lint:
	flake8 $(SRC) tests
	isort --check --diff --profile black $(SRC) tests
	black --check $(SRC) tests

## Format source code with black and isort
.PHONY: format
format:
	isort --profile black $(SRC) tests
	black $(SRC) tests

## Run the test suite
.PHONY: test
test:
	pytest

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with: conda activate $(PROJECT_NAME)"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
