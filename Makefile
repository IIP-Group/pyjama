docs: FORCE
	cd docs && ./build_docs.sh

install: FORCE
	pip install .

lint:
	pylint sionna/

FORCE:
