"""Check reproducibility, missing-value handling, and saved-model predictions."""
import sys
import tempfile
import unittest
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_demo_data import generate_data
from model_selection import ROOT, run_experiment


class DemoTest(unittest.TestCase):
    def test_reproducible_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = generate_data(Path(directory) / "first.csv")
            second = generate_data(Path(directory) / "second.csv")
            pd.testing.assert_frame_equal(first, second)
            self.assertTrue(first.isna().any().any())
            self.assertEqual(set(first.would_recommend), {0, 1})

    def test_saved_models_and_training_only_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scores = run_experiment(output_dir=directory / "results", tracking_dir=directory / "mlruns")
            self.assertEqual(len(scores), 3)
            self.assertTrue(np.isfinite(scores.select_dtypes("number")).all().all())
            runs = mlflow.search_runs(experiment_names=["Automatic_Pipelines_Demo"])
            for name in ["DummyClassifier", "LogisticRegression", "RandomForestClassifier"]:
                run = runs[runs["tags.mlflow.runName"] == name].iloc[0]
                model = mlflow.sklearn.load_model(f"runs:/{run.run_id}/model")
                data = pd.read_csv(ROOT / "examples/synthetic_survey.csv")
                X = data.drop(columns="would_recommend")
                self.assertEqual(len(model.predict(X.head(8))), 8)
                numeric = model.named_steps["preprocessing"].named_steps["preprocessor"].named_transformers_["numeric"]
                from sklearn.model_selection import train_test_split
                train, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data.would_recommend)
                expected = train.select_dtypes("number").drop(columns="would_recommend").median().to_numpy()
                np.testing.assert_allclose(numeric.named_steps["fill_missing"].statistics_, expected)
                sample = X.head(2).copy()
                sample["channel"] = "new_channel"
                numeric_columns = sample.select_dtypes("number").columns
                sample[numeric_columns] = sample[numeric_columns].astype(float)
                sample["age"] = np.nan
                sample["satisfaction"] = np.nan
                self.assertEqual(len(model.predict(sample)), 2)
                served_model = mlflow.pyfunc.load_model(f"runs:/{run.run_id}/model")
                self.assertEqual(len(served_model.predict(sample)), 2)


if __name__ == "__main__":
    unittest.main()
