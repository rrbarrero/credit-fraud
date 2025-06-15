from factory import create_models_pipeline

models_pipeline = create_models_pipeline()

print(models_pipeline.evaluate())
