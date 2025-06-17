# Kaggle Credit Fraud Detection Challenge

This repository contains my solution approach for the [Kaggle Credit Fraud Detection challenge](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The primary goal is to build a model that accurately identifies fraudulent credit card transactions.

## Getting Started

To run this project, follow these steps:

1.  Download the dataset from the Kaggle competition page.
2.  Place the dataset files inside a `dataset/` folder at the root of the project.
3.  Execute the entire pipeline using the following command (*you will need uv*):
    ```sh
    make run
    ```

## Iteration 1: XGBoost

In this first iteration, I have implemented a solution using an **XGBoost** classifier.

### Methodology & Architecture

The project has been developed following **Clean Code** principles and is supported by a comprehensive suite of **tests**. This architectural approach was chosen to ensure flexibility and maintainability. The key advantages of this design are:

* **Easy Feature Management**: The structure allows for simple addition or removal of features from the model pipeline.
* **Modular Balancing**: Different data balancing techniques (e.g., SMOTE, RandomUnderSampler) can be easily swapped and tested.
* **Extensible Pipeline**: New models can be seamlessly integrated into the pipeline for comparative analysis against the baseline.

### Learnings & Technical Experience

> [!IMPORTANT] 
> I initially attempted to use **Polars** for data manipulation, as I find its 
> syntax highly efficient and expressive. However, I encountered challenges with 
> its >integration into the broader ecosystem, particularly with libraries like 
> **scikit-learn**. This lack of seamless compatibility made it less practical for > this project.

### Current Status & Future Work

While the current implementation is functional, there is still some degree of coupling in the codebase. I plan to refactor this over time to further improve modularity and adhere to SOLID principles.

My immediate future plans include:

* **Feature Engineering**: Explore new features to enhance model accuracy.
* **Model Comparison**: Add other classification models (such as LightGBM, CatBoost, or Logistic Regression) to the pipeline and compare their performance.
* **Code Refactoring**: Decouple components to improve maintainability.

### Current Results

The evaluation metrics for the initial model, `xgboost_v1`, are detailed below.

<pre>
EvaluationResult(
    model_name='xgboost_v1',
    accuracy=0.9995418125425879,
    pr_auc=0.834454585983577,
    per_class={
        '0': {
            'precision': 0.999658803459027,
            'recall': 0.9998823197138016,
            'f1-score': 0.9997705490936689,
            'support': 84976.0
        },
        '1': {
            'precision': 0.9186991869918699,
            'recall': 0.795774647887324,
            'f1-score': 0.8528301886792453,
            'support': 142.0
        }
    },
    macro_avg={
        'precision': 0.9591789952254485,
        'recall': 0.8978284838005628,
        'f1-score': 0.9263003688864571,
        'support': 85118.0
    },
    weighted_avg={
        'precision': 0.999523740775008,
        'recall': 0.9995418125425879,
        'f1-score': 0.9995254125634538,
        'support': 85118.0})
</pre>

### Iteration n2

With parameters from grid_search:

<pre>
[EvaluationResult(
    model_name='xgboost_v1', 
    per_class={
        '0': {
            'precision': 0.9996705378469648, 
            'recall': 0.9997999435134626, 
            'f1-score': 0.9997352364926484, 
            'support': 84976.0}, 
        '1': {
            'precision': 0.8702290076335878, 
            'recall': 0.8028169014084507, 
            'f1-score': 0.8351648351648352, 
            'support': 142.0}
        }, 
    accuracy=0.9994713221645245, 
    macro_avg={
        'precision': 0.9349497727402762, 
        'recall': 0.9013084224609567, 
        'f1-score': 0.9174500358287418, 
        'support': 85118.0
    }, 
    weighted_avg={
        'precision': 0.9994545941301212, 
        'recall': 0.9994713221645245, 
        'f1-score': 0.9994606882538676, 
        'support': 85118.0}, 
    pr_auc=np.float64(0.8354554023593456))]
</pre>

> [!NOTE]
> Given that in fraud detection, recall (catching all fraud instances) is often prioritized, a small gain in recall might be considered valuable, > even if it comes with a minor trade-off in precision. The fact that PR-AUC also improved, albeit marginally, suggests the optimized model is 
> indeed slightly better at balancing precision and recall across various thresholds."""