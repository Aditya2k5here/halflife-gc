PY ?= python3

.PHONY: help test results report figures clean all

help:
	@echo "make test      run the test suite (~1s)"
	@echo "make results   run the full experiment -> results/results.json (~3min)"
	@echo "make report    rebuild results/report.html and results/summary.md"
	@echo "make figures   re-render the README figures (needs playwright)"
	@echo "make all       results + report"

test:
	PYTHONPATH=.:tests $(PY) -m pytest tests -q

results:
	PYTHONPATH=. $(PY) -m halflife bench

report:
	PYTHONPATH=. $(PY) -m halflife report

figures:
	PYTHONPATH=. $(PY) tools/render_figures.py

all: results report

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
