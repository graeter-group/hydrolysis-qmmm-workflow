
preview:
	quarto preview

fix:
	chmod -R +rw docs

sync:
	./scripts/sync-hamilton-data.sh

analysis:
	./setup.sh && python ./scripts/run-analysis.py

example:
	./scripts/sync-example-data.sh

.PHONY: tests docs analysis
tests:
	./setup.sh && python -m pytest src

docs:
	./setup.sh && quartodoc build && quartodoc interlinks && quarto render
