
preview:
	quarto preview thesis.qmd

fix:
	chmod -R +rw docs

sync:
	./scripts/sync-hamilton-data.sh

example:
	./scripts/sync-example-data.sh

.PHONY: tests docs
tests:
	./setup.sh && python -m pytest src

docs:
	./setup.sh && quartodoc build && quartodoc interlinks && quarto render
