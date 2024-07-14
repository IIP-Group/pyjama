docs: FORCE
	cd docs && make html

install: FORCE
	pip install .

lint:
	pylint sionna/

FORCE:
