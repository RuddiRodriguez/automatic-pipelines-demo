"""Build, compare, and record classification pipelines from a YAML file."""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import estimator_html_repr

from custom_transformers import DropColumns, RawDataTransformations
from mlflow_helper import MLFlowLogger

ROOT = Path(__file__).resolve().parents[1]


def NumericPreprocessor():
    return Pipeline([("fill_missing", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def CategoricalPreprocessor():
    return Pipeline([("fill_missing", SimpleImputer(strategy="most_frequent")),
                     ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])


STEPS = {"NumericPreprocessor": NumericPreprocessor,
         "CategoricalPreprocessor": CategoricalPreprocessor,
         "DropColumns": DropColumns, "StandardScaler": StandardScaler,
         "SimpleImputer": SimpleImputer, "OneHotEncoder": OneHotEncoder}
MODELS = {"DummyClassifier": DummyClassifier,
          "LogisticRegression": LogisticRegression,
          "RandomForestClassifier": RandomForestClassifier}


def load_config(config_path):
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def apply_filters(df, pipeline, filters, filters_parameters):
    for step in filters:
        df = df.pipe(getattr(pipeline, step), **filters_parameters.get(step, {}))
    return df


def choose_selector(cols):
    if cols == "numeric":
        return make_column_selector(dtype_include=np.number)
    if cols == "categorical":
        return make_column_selector(dtype_include=object)
    return cols


def create_pipeline(config):
    transformers, general_steps = [], []
    details = zip(config["pipeline_steps_names"], config["pipeline_steps"],
                  config["pipeline_steps_columns"], config["pipeline_steps_params"],
                  config["pipeline_steps_type"], strict=True)
    for name, step_name, cols, params, step_type in details:
        step = STEPS[step_name](**params)
        if step_type == "transformer":
            transformers.append((name, step, choose_selector(cols)))
        else:
            general_steps.append((name, step))
    if transformers:
        general_steps.append(("preprocessor", ColumnTransformer(transformers, remainder="drop")))
    return Pipeline(general_steps)


def run_experiment(config_path=ROOT / "config/config.yml", output_dir=ROOT / "results",
                   tracking_dir=ROOT / "mlruns"):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    config = load_config(config_path)
    data_path = ROOT / config["data_parameters"][0]["path"]
    target = config["target"][0]
    preparation = RawDataTransformations(str(data_path))
    df = apply_filters(preparation.load_data(), preparation,
                       config.get("filters", []), config.get("filters_params", {}))
    numeric_columns = df.drop(columns=target).select_dtypes("number").columns
    df[numeric_columns] = df[numeric_columns].astype(float)
    X_train, X_test, y_train, y_test = preparation.split_train_test(df, target)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = MLFlowLogger(tracking_dir)
    mlflow.set_experiment("Automatic_Pipelines_Demo")
    scores = []
    with mlflow.start_run(run_name=data_path.name):
        mlflow.set_tags({"data_kind": config.get("data_kind", "unspecified"), "purpose": "workflow demonstration"})
        logger.log_params({"training_rows": len(X_train), "test_rows": len(X_test), "seed": 42})
        logger.log_artifacts([config_path])
        for name, params in zip(config["models"], config["models_params"], strict=True):
            with mlflow.start_run(run_name=name, nested=True):
                clf = Pipeline([("preprocessing", create_pipeline(config)),
                                ("model", MODELS[name](**params))])
                cv = cross_validate(clf, X_train, y_train,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                    scoring=["accuracy", "f1", "roc_auc"], error_score="raise")
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                probabilities = clf.predict_proba(X_test)[:, 1]
                metrics = {f"cv_{key[5:]}": float(value.mean())
                           for key, value in cv.items() if key.startswith("test_")}
                metrics.update(test_accuracy=accuracy_score(y_test, predictions),
                               test_f1=f1_score(y_test, predictions, zero_division=0),
                               test_roc_auc=roc_auc_score(y_test, probabilities))
                logger.log_params({"model": name, **params})
                logger.log_metrics(metrics)
                mlflow.sklearn.log_model(clf, "model", signature=infer_signature(X_train, clf.predict(X_train)),
                                         pip_requirements=str(ROOT / "requirements.txt"))
                diagram = output_dir / f"{name}_pipeline.html"
                diagram.write_text(estimator_html_repr(clf))
                logger.log_artifacts([diagram])
                scores.append({"model": name, **metrics})
        summary = pd.DataFrame(scores)
        summary.to_csv(output_dir / "comparison.csv", index=False)
        logger.log_artifacts([output_dir / "comparison.csv"])
    print(summary.round(3).to_string(index=False))
    print(f"\nResults: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yml")
    args = parser.parse_args()
    run_experiment(args.config)
