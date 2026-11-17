# ML Studio (MLusingTKinterface)

A CustomTkinter desktop application for exploring tabular data, preparing features and training machine learning models without writing code.

Developed by **Mohammed AACHABI** and **Abdelmajid BENJELLOUN**, supervised by **Sanae KHALI ISSA**, for the Master AI & Data Science programme (2023–2024).

![Application welcome artwork](home1.png)

## Features

- **Data**: load CSV, TSV or Excel files (separator auto-detected). Inspect a preview, a per-column profile (type, missing values, unique values) and numeric statistics.
- **Prepare**: select columns, then:
  - fill missing values (mean, median, mode or zero; text columns fall back to the mode)
  - drop incomplete rows
  - one-hot or ordinal encode
  - drop columns
  - remove duplicates or IQR outliers
  - standardize values

  Every step is logged and can be **undone**, or the dataset reset to the original file.
- **Visualize**: histograms, box plots, scatter plots, category counts, mean-by-category point plots and a correlation heatmap. You choose the columns, charts are embedded in the window, and you can save them as PNG, SVG or PDF.
- **Classify**: 8 models (Logistic Regression, Naive Bayes, RBF and linear SVM, ID3 decision tree, KNN, Random Forest, Gradient Boosting).
  - Results: accuracy, balanced accuracy, macro F1 and k-fold CV accuracy.
  - Plots: confusion matrix, per-class report, permutation feature importance, a PCA projection that highlights misclassified rows, and a decision tree diagram.
  - One click compares all models.
- **Cluster**: K-Means on scaled features. An elbow and silhouette sweep picks *k*, and the page shows a PCA scatter and per-cluster profiles. You can add the labels to the dataset.
- **Regression**: cross-validated comparison of six regressors with automatic log-scaling of skewed targets. Shows predicted-vs-actual and residual plots.
- **Export**: a PDF report (overview, profile, statistics, preparation steps and the latest model results), or save the prepared dataset as CSV or Excel.
- Light and dark mode, a status bar for feedback, and slow work runs in the background so the window never freezes.

## Setup and launch

Install Python 3.10+ with Tcl/Tk support, then from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

On macOS and Linux use `.venv/bin/python`. On Linux, install your distribution's Tkinter package if `python -m tkinter` fails.

Optional arguments: `python main.py "Car Price.csv" --theme Dark` opens a dataset on start-up with the chosen theme (`Light`, `Dark` or `System`).

## Usage

1. **Home**: click **Open dataset…**, or **Try sample: car prices**.
2. **Data**: check the column types and missing values.
3. **Prepare**: tick columns, then apply operations. Use **Undo** if needed.
4. **Visualize**: explore distributions and relationships.
5. **Classify**, **Cluster** or **Regression**: pick a target and train.
6. **Export**: save a PDF report or the prepared data.

See the [user manual](Manuel_of_use.md) for details.

## Algorithm improvements in version 2

| Area | Before | Now |
| --- | --- | --- |
| Train/test split | `test_size=0.8`: models trained on only **20%** of the rows | Stratified 80/20 split (adjustable 10–40%) |
| Scaling | None: SVM/logistic regression ran on raw scales (and logistic regression did not converge) | `StandardScaler` inside every pipeline |
| Text features | Crashed unless one-hot encoded by hand | Imputed and one-hot encoded automatically |
| Missing values | Filled on the full dataset before splitting (leakage) | Imputed inside the pipeline, fitted on training rows only |
| Evaluation | One noisy split, accuracy only | Stratified k-fold CV ± std, balanced accuracy, macro F1 |
| Models | 4 classifiers, each retrained again just for plotting | 8 classifiers, trained once, plus a side-by-side comparison |
| Target check | Any column accepted (e.g. prices as 445 "classes") | Continuous targets are redirected to Regression |
| Identifier columns | Used as features | Row IDs, unique text and constant columns are skipped automatically |
| K-Means | Unscaled, so large-range columns dominated; *k* guessed | Scaled features, silhouette/elbow sweep, cluster profiles |
| Regression | Fixed-alpha Ridge, raw target | Tuned `RidgeCV`, log target for skewed data, gradient boosting |

Measured effect, on the same data:

- **Wine dataset**, linear SVM: 0.86 → 0.97 test accuracy. Logistic regression: 0.90 → 0.97.
- **Car Price data**, best regression test MAE: 146,717 → **109,879** (−25%). R² 0.55 → **0.74**.

## Command-line regression benchmark

```powershell
python -m mlstudio.core.regression "Car Price.csv" --target Selling_Price --output benchmark_results.json
```

## Tests

```powershell
python -m unittest discover -s tests -t .
```

`tests/test_core.py` covers the data and ML logic. `tests/test_ui.py` drives the full interface end to end (it needs a display).

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | Entry point (`python -m mlstudio` also works) |
| `mlstudio/core/dataset.py` | Loading, profiling, cleaning operations and undo history |
| `mlstudio/core/features.py` | Feature detection and leakage-free preprocessing |
| `mlstudio/core/classification.py` | Classifiers, evaluation, comparison, importance |
| `mlstudio/core/clustering.py` | K-Means, k selection, cluster profiles |
| `mlstudio/core/regression.py` | Regression comparison (also a CLI) |
| `mlstudio/core/report.py` | PDF report |
| `mlstudio/ui/app.py` | Main window, sidebar, status bar, background tasks |
| `mlstudio/ui/widgets.py` | Cards, metric tiles, data tables, plot panels, column picker |
| `mlstudio/ui/pages/` | One module per page |
| `tests/` | Unit and UI tests |

## Dependencies

Listed in [requirements.txt](requirements.txt):

- **customtkinter**: interface
- **pandas, numpy**: data handling
- **scikit-learn**: models
- **matplotlib, seaborn**: charts
- **Pillow**: images
- **reportlab**: PDF reports
- **openpyxl, xlrd**: Excel files
