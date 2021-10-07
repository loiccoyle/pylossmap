PKGNAME=pylossmap

default: install

install:
	pip install .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

clean:
	python setup.py clean --all

test:
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) tests

test-cov:
	py.test --cov ./pylossmap --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage tests

test-loop:
	py.test tests
	ptw --ext=.py,.pyx --ignore=doc tests

docstyle:
	py.test --docstyle

setup.py:
	poetry2setup > setup.py

requirements:
	poetry export > requirements.txt
	poetry export --dev > requirements-dev.txt

format:
	isort . && black .


.PHONY: clean install install-dev test test-cov test-loop docstyle setup.py requirements format
