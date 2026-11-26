PYTHON := python
UV := uv
SRC := src
ARGS ?= index 

export UV_CACHE_DIR := /tmp/$(USER)-rag/uv-cache
export UV_PROJECT_ENVIRONMENT := /tmp/$(USER)-rag/venv

.PHONY: install run debug clean lint

install:
	mkdir -p "$(UV_CACHE_DIR)" "$(UV_PROJECT_ENVIRONMENT)"
	$(UV) sync
	ln -sfnT "$(UV_PROJECT_ENVIRONMENT)" .venv

run:
	$(UV) run $(PYTHON) -m $(SRC) $(ARGS)

debug:
	$(UV) run $(PYTHON) -m pdb -m $(SRC) $(ARGS)

lint:
	$(UV) run flake8 $(SRC)
	$(UV) run mypy $(SRC) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs 

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete