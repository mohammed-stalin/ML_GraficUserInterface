import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

from mlstudio.core import classification, clustering, dataset, regression
from mlstudio.core.features import infer_features
from mlstudio.core.report import build_report

ROOT = Path(__file__).resolve().parents[1]


def iris():
    return load_iris(as_frame=True).frame.rename(columns={"target": "label"})


class DatasetTests(unittest.TestCase):
    def test_fill_missing_handles_numeric_and_text(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0], "c": ["a", None, "a"]})
        out = dataset.fill_missing(df.copy(), "mean")
        self.assertEqual(out.loc[1, "x"], 2.0)
        self.assertEqual(out.loc[1, "c"], "a")

    def test_apply_undo_reset(self):
        ds = dataset.Dataset(pd.DataFrame({"c": ["a", "b", "a"], "y": [1, 2, 3]}))
        ds.apply("encode", lambda df: dataset.one_hot_encode(df, ["c"]))
        self.assertIn("c_a", ds.df)
        ds.apply("drop", lambda df: dataset.drop_columns(df, ["y"]))
        ds.undo()
        self.assertIn("y", ds.df)
        self.assertEqual(ds.log, ["encode"])
        ds.reset()
        self.assertEqual(list(ds.df.columns), ["c", "y"])
        self.assertFalse(ds.can_undo)

    def test_failed_operation_is_not_committed(self):
        ds = dataset.Dataset(pd.DataFrame({"x": [1, 2]}))
        with self.assertRaises(ValueError):
            ds.apply("drop all", lambda df: dataset.drop_columns(df, ["x"]))
        self.assertEqual(ds.log, [])

    def test_outliers_and_duplicates(self):
        df = pd.DataFrame({"x": [1, 2, 2, 3, 2, 100]})
        self.assertNotIn(100, dataset.remove_outliers_iqr(df.copy())["x"].tolist())
        self.assertEqual(len(dataset.drop_duplicates(df.copy())), 4)

    def test_sample_file_loads(self):
        ds = dataset.Dataset.load(ROOT / "Car Price.csv")
        self.assertEqual(ds.summary()["rows"], 4340)


class FeatureTests(unittest.TestCase):
    def test_identifiers_and_constants_are_skipped(self):
        df = pd.DataFrame({"id": range(50), "name": [f"n{i}" for i in range(50)], "const": 1,
                           "x": np.random.default_rng(0).normal(size=50), "c": ["a", "b"] * 25})
        spec = infer_features(df)
        self.assertEqual(spec.numeric, ["x"])
        self.assertEqual(spec.categorical, ["c"])
        self.assertEqual(set(spec.dropped), {"id", "name", "const"})


class ClassificationTests(unittest.TestCase):
    def test_every_model_trains_on_iris(self):
        for name in classification.CLASSIFIERS:
            result = classification.train_classifier(iris(), "label", name)
            self.assertGreater(result.accuracy, 0.8, name)
            self.assertEqual(result.test_rows, 30)  # 80/20 split, not the old 20/80

    def test_text_features_and_missing_values_are_handled(self):
        df = iris()
        df["colour"] = np.where(df.index % 2, "red", "blue")
        df.iloc[0, 0] = np.nan
        result = classification.train_classifier(df, "label", "Logistic Regression")
        self.assertIn("colour_red", result.feature_names)

    def test_continuous_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Regression"):
            classification.train_classifier(iris(), "sepal length (cm)", "Naive Bayes")

    def test_scaling_beats_the_original_setup_on_wine(self):
        wine = load_wine(as_frame=True).frame
        result = classification.train_classifier(wine, "target", "SVM (linear)")
        self.assertGreater(result.cv_accuracy, 0.95)

    def test_compare_and_importance(self):
        table = classification.compare_classifiers(iris(), "label", ["Naive Bayes", "Decision Tree (ID3)"])
        self.assertEqual(len(table), 2)
        result = classification.train_classifier(iris(), "label", "Random Forest")
        imp = classification.feature_importance(result)
        self.assertTrue(imp.index[0].startswith("petal"))
        coords, _ = classification.projection(result)
        self.assertEqual(coords.shape, (30, 2))


class ClusteringTests(unittest.TestCase):
    def test_sweep_and_run(self):
        table = clustering.sweep_k(iris(), 2, 5, exclude=["label"])
        self.assertIn(clustering.best_k(table), range(2, 6))
        result = clustering.run_kmeans(iris(), 3, exclude=["label"])
        self.assertEqual(len(set(result.labels)), 3)
        self.assertEqual(result.profile["Size"].sum(), 150)

    def test_deterministic(self):
        a = clustering.run_kmeans(iris(), 3, exclude=["label"]).labels
        b = clustering.run_kmeans(iris(), 3, exclude=["label"]).labels
        np.testing.assert_array_equal(a, b)


class RegressionTests(unittest.TestCase):
    def test_car_prices_beat_baseline(self):
        df = pd.read_csv(ROOT / "Car Price.csv").sample(1200, random_state=0)
        report = regression.compare_regressors(df, "Selling_Price", model_names=["Median baseline", "Ridge (tuned)"])
        self.assertTrue(report.log_target)
        self.assertEqual(report.best_model, "Ridge (tuned)")
        self.assertGreater(report.results.iloc[0]["Test R2"], 0.5)

    def test_text_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not numeric"):
            regression.compare_regressors(pd.read_csv(ROOT / "Car Price.csv"), "Fuel")


class ReportTests(unittest.TestCase):
    def test_pdf(self):
        import tempfile
        ds = dataset.Dataset(iris())
        result = classification.train_classifier(ds.df, "label", "Naive Bayes")
        with tempfile.TemporaryDirectory() as folder:
            path = build_report(Path(folder) / "r.pdf", ds, classification=result)
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
