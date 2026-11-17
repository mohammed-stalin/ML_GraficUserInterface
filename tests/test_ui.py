"""End-to-end smoke test of the CustomTkinter interface (needs a display)."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sklearn.datasets import load_iris

from mlstudio.ui.app import StudioApp


def wait(app, timeout=120):
    deadline = time.time() + timeout
    app.update()
    while app.tasks.busy:
        if time.time() > deadline:
            raise TimeoutError("background task did not finish")
        app.update()
        time.sleep(0.02)
    app.update()


class UiWorkflowTest(unittest.TestCase):
    def test_full_workflow(self):
        errors = []
        app = StudioApp("Light")
        app.report_callback_exception = lambda *exc: errors.append(exc)
        app.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                csv = Path(folder) / "iris.csv"
                data = load_iris(as_frame=True).frame.rename(columns={"target": "label"})
                data["species"] = data["label"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
                data.loc[3, "sepal width (cm)"] = None
                data.to_csv(csv, index=False)

                for key in app.pages:  # empty states render without data
                    app.show_page(key)
                app.open_dataset(str(csv))
                wait(app)
                self.assertEqual(len(app.dataset.df), 150)
                self.assertEqual(app.current, "data")

                prepare = app.pages["prepare"]
                app.show_page("prepare")
                prepare.picker.select(["label"])
                prepare.run("drop")
                self.assertNotIn("label", app.dataset.df)
                prepare.undo()
                self.assertIn("label", app.dataset.df)
                prepare.picker.select([])
                prepare.run("fill")
                self.assertEqual(int(app.dataset.df.isna().sum().sum()), 0)

                vis = app.pages["visualize"]
                app.show_page("visualize")
                for name in ("Histogram", "Box plot", "Scatter"):
                    vis.chart.set(name)
                    vis._chart_changed(1)
                for name in ("Category counts", "Mean by category", "Correlation"):
                    vis.chart2.set(name)
                    vis._chart_changed(2)
                app.update()

                cls = app.pages["classify"]
                app.show_page("classify")
                cls.target.set("species")
                cls.model.set("Decision Tree (ID3)")
                cls.train()
                wait(app)
                self.assertIsNotNone(app.results["classification"])
                cls.tabs.set("Feature importance")
                cls._tab_changed()
                wait(app)
                cls.compare()
                wait(app)

                clu = app.pages["cluster"]
                app.show_page("cluster")
                clu.exclude.set("species")
                clu.sweep()
                wait(app)
                clu.run()
                wait(app)
                clu.save_labels()
                self.assertIn("Cluster", app.dataset.df)

                reg = app.pages["regress"]
                app.show_page("regress")
                reg.target.set("petal width (cm)")
                reg.run()
                wait(app)
                self.assertIsNotNone(app.results["regression"])

                app.show_page("export")
                pdf = Path(folder) / "report.pdf"
                with patch("mlstudio.ui.pages.export.filedialog.asksaveasfilename", return_value=str(pdf)):
                    app.pages["export"].export_pdf()
                    wait(app)
                self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
                out = Path(folder) / "clean.csv"
                with patch("mlstudio.ui.pages.export.filedialog.asksaveasfilename", return_value=str(out)):
                    app.pages["export"].export_data()
                self.assertTrue(out.exists())

                app.set_appearance("Dark")
                app._restyle()
                app.update()
                self.assertEqual(errors, [])
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
