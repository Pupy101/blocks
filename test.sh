#!/bin/bash
# run in active python enviroment
pytest . --disable-pytest-warnings
pylint --rcfile=pyproject.toml blocks
mypy --config-file=pyproject.toml blocks
