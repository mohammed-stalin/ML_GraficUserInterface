# User manual: ML Studio

## Before you start

Follow the [setup instructions](README.md#setup-and-launch), then run `python main.py`. The window has a sidebar on the left:

- Workspace: **Home**, **Data**, **Prepare**, **Visualize**
- Models: **Classify**, **Cluster**, **Regression**
- Output: **Export**

The bottom of the sidebar shows the loaded dataset and a **Light / Dark / System** theme switch. The status bar at the bottom of the window shows confirmations, warnings and errors. A progress bar appears there while a model trains in the background.

Prepare a CSV, TSV or Excel file with a header row and one observation per row. The CSV separator (comma, semicolon, tab…) is detected automatically.

Changes affect the dataset in memory only. Use **Export → Save as CSV or Excel** to keep them.

## 1. Home

- **Open dataset…** opens a file picker.
- **Try sample: car prices** loads the bundled `Car Price.csv`.
- **User manual** opens this document.

## 2. Data

After loading, the **Data** page shows four tiles: rows, columns, missing cells and duplicate rows. It has three tabs:

- **Preview**: the first 500 rows.
- **Columns**: type, missing count and percentage, number of unique values and example values for each column.
- **Statistics**: count, mean, standard deviation, min, quartiles and max of numeric columns.

## 3. Prepare

1. Tick columns in the **Columns** list. Use the filter box, **All** or **None**.
2. Click an operation.

| Operation | What it does | Columns used |
| --- | --- | --- |
| Fill missing values | Uses the chosen strategy: mean, median, most frequent or zero. Text columns always use the most frequent value. | Selected, or all |
| Drop rows with missing values | Removes rows with a gap in the chosen columns | Selected, or all |
| One-hot encode | Replaces a text column with 0/1 indicator columns (max 50 categories) | Selected (required) |
| Ordinal encode | Replaces categories with integer codes | Selected (required) |
| Drop selected columns | Removes columns such as identifiers | Selected (required) |
| Remove duplicate rows | Keeps the first copy of identical rows | — |
| Remove outliers (1.5 × IQR) | Drops rows outside Q1 − 1.5·IQR … Q3 + 1.5·IQR | Selected numeric, or all numeric |
| Standardize | Rescales to mean 0, standard deviation 1 | Selected numeric, or all numeric |

The **History** card lists every step:

- **Undo** reverts the last step.
- **Reset to original** restores the file as loaded.

An operation that fails, or that would remove every row, is not applied, and the status bar explains why.

You do **not** need to encode or impute before modelling: the model pages do this automatically and without data leakage. Use Prepare to clean the data itself, for example to remove duplicates, identifiers or impossible values.

## 4. Visualize

Choose a chart, then its columns. The chart redraws on every change.

| Chart | Settings |
| --- | --- |
| Histogram | One numeric column, or a grid of all of them; optional colour split |
| Box plot | Value column (or all); optional grouping column |
| Scatter | Two numeric columns; optional colour column (5,000-row sample for large data) |
| Category counts | The 20 most frequent values of a column |
| Mean by category | Average of a numeric column per category with 95% confidence intervals |
| Correlation | Pearson correlation of numeric columns |

Use the toolbar under the chart to zoom and pan. Click **Save chart as image…** to export it as PNG, SVG or PDF.

## 5. Classify

1. Choose the **Target column**, the category to predict. The hint below it lists the classes. It also warns if the column looks continuous, in which case use **Regression** instead.
2. Choose a **Model**.
3. Optionally change the **Test set** size (default 20%) and whether text columns are used.
4. Click **Train model**, or **Compare all models**.

The remaining columns become features. The app ignores row identifiers, unique text values and constant columns, and lists them in the left panel. Numeric features are imputed with the median and standardized. Text features are imputed with the most frequent value and one-hot encoded. All of this is learned from the training rows only.

### Read the results

**Metric tiles**

| Tile | What it measures |
| --- | --- |
| Accuracy | Share of correctly classified test rows |
| Balanced accuracy | Average recall per class. More honest when classes are imbalanced. |
| Macro F1 | Average F1 score across classes |
| CV accuracy | Mean ± standard deviation of k-fold cross-validation on the training rows. Large gaps between it and test accuracy suggest an unlucky split or overfitting. |

**Result tabs**

| Tab | What it shows |
| --- | --- |
| Confusion matrix | Actual classes in rows, predictions in columns. Shows counts and the row percentage. |
| Report | Precision, recall, F1 and support for each class |
| Feature importance | How much test accuracy drops when a column is shuffled. Computed when you open the tab. |
| Projection | Test rows in two dimensions (PCA). Misclassified rows are circled in red. |
| Decision tree | The first levels of the tree (Decision Tree (ID3) only) |
| Comparison | Every model ranked by cross-validated macro F1, with test accuracy and training time |

## 6. Cluster

1. Optionally choose a column to **exclude**, such as a known label.
2. Click **Find best k**. The app compares k = 2…10 using the elbow curve and the silhouette score, then sets the slider to the best k.
3. Click **Run K-Means**.

The page shows:

- **Clusters**: rows plotted on two principal components.
- **Profiles**: the size of each cluster, its mean numeric values and its most common category values.
- **Silhouette tile**: separation quality, from −1 to 1; higher is better.

Features are standardized, so columns with large values do not dominate. Click **Add cluster labels to dataset** to store the result as a `Cluster` column. This step can be undone on the Prepare page.

## 7. Regression

1. Choose a numeric **Target column**, for example `Selling_Price`.
2. Choose the **Log-scale target** option. Auto applies it to positive, skewed targets such as prices.
3. Click **Compare models**.

The app then:

- removes duplicate rows;
- splits the data 80/20;
- ranks six models by three-fold cross-validated MAE on the training rows: median baseline, tuned Ridge, decision tree, random forest, extra trees and gradient boosting;
- evaluates every model on the test rows.

| Result | Meaning |
| --- | --- |
| MAE, RMSE | Errors in the target's units; lower is better. RMSE penalises large errors more. |
| R² | 1 is perfect; 0 equals always predicting the mean. |
| vs baseline | How much lower the error is than always predicting the median |

The **Predicted vs actual** and **Residuals** tabs show where the best model makes errors.

## 8. Export

- **Export PDF report…** writes a report with:
  - the dataset overview, column profile and statistics;
  - your preparation steps;
  - the latest classification, regression and clustering results.
- **Save as CSV or Excel…** saves the prepared dataset.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` at start-up | Install `requirements.txt` into the same Python that launches the app. |
| Tkinter is unavailable | Install Python with Tcl/Tk support. `python -m tkinter` checks it. |
| Excel import fails | Install `openpyxl` (`.xlsx`) or `xlrd` (`.xls`). |
| "looks continuous … use the Regression page" | The target has many numeric values. Predict it with Regression, or choose a categorical column. |
| "class(es) with a single example were ignored" | These classes are too rare to split into training and test rows. |
| "No usable feature columns" | Every other column was skipped (identifiers, constants, text with text columns turned off). |
| One-hot encoding refused | The column has more than 50 categories. Drop it, ordinal-encode it, or let the model pages encode it automatically. |
| Results differ from version 1 | Version 1 trained on 20% of the rows without scaling. The new numbers are more reliable. |
