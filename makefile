.PHONY: test

test:
	uv run pytest -vv

cache_clean:
	rm cache/*.joblib

xgboost: cache_clean
	uv run --directory src python -m models.xgboost_model

run: xgboost
