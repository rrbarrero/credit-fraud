.PHONY: test

test:
	uv run pytest -vv

cache_clean:
	rm -f cache/*.joblib

run: cache_clean
	uv run --directory src python -m main
