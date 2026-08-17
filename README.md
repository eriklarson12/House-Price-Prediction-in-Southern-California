<div align="center">

# House Price Prediction in Southern California

Predicting 2021 home sale prices from seven property features, comparing Linear Regression, Ridge, and XGBoost across 15,474 Southern California listings.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0-337AB7)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)

**[Read the full project report](index.md)**

</div>

---

This project predicts residential sale prices in Southern California from a 2021 Kaggle listing dataset. Three regression models train on the same seven features and are evaluated on an identical 80/20 split, so the comparison isolates model choice rather than data handling.

Gradient boosting wins clearly. XGBoost explains 64% of price variance against 35% for both linear baselines, cutting typical absolute error from $228K to $145K. The gap is the finding: Southern California pricing turns on location and nonlinear interactions that a linear equation cannot represent.

<!-- Screenshot: add a model comparison plot to assets/ and reference it here with alt text. -->

## Table of Contents

- [Results](#results)
- [Dataset](#dataset)
- [Methods](#methods)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Team](#team)
- [Limitations](#limitations)

## Results

Test set metrics across 3,095 held out samples. The average sale price in the test set is $707,084.

| Model | RMSE | MAE | R² | RMSE as % of average price |
|---|---|---|---|---|
| Linear Regression | $309,313 | $228,129 | 0.351 | 43.7% |
| Ridge, alpha 1.0 | $309,293 | $228,133 | 0.351 | 43.7% |
| XGBoost | $230,669 | $144,856 | 0.639 | 32.6% |

Five fold cross validation puts XGBoost at a mean RMSE of $233,579 with a standard deviation of $6,307, so the advantage holds across splits rather than resting on one lucky partition. XGBoost reaches an R² of 0.931 on training against 0.639 on test, which is real overfitting. The linear models barely move between the two, 0.353 on training and 0.351 on test, underfitting consistently.

Ridge lands within $21 of plain Linear Regression on RMSE. L2 regularization stabilizes the coefficients but cannot add the expressiveness the linear form is missing.

XGBoost ranks bathroom count (0.30), city (0.18), and square footage (0.16) as the three strongest price drivers.

## Dataset

[House Prices and Images, SoCal](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal) on Kaggle, scraped in 2021. A seven column subset is committed at `data/socal2_7_columns.csv`. The companion image set from the original upload is not used.

| Property | Value |
|---|---|
| Rows | 15,474 |
| Distinct cities | 415 |
| Modeled columns | `bed`, `bath`, `sqft`, `citi`, `price` |
| Price range | $195,000 to $2,000,000 |
| Median price | $639,000 |
| Missing values | none |

`image_id` and `street` are dropped before modeling.

## Methods

- **Imputation:** median for numeric columns, most frequent for categorical, through `SimpleImputer`. The dataset has no missing values, so the step is a no op here and exists for reproducibility on dirtier inputs.
- **Feature engineering:** four derived columns, `price_per_sqft`, `sqft_per_room`, `total_rooms`, and `bath_to_bed_ratio`. Bedroom counts of zero are floored to one so the ratio features cannot divide by zero.
- **Leakage guard:** `price_per_sqft` is computed for exploratory analysis but excluded from the model feature set, because it is derived from the target.
- **Encoding:** city names are label encoded into 415 integer IDs.
- **Outliers:** IQR bounds are computed and reported but nothing is dropped, which keeps the full spread of the market in the training data.
- **Split:** 80/20 train and test, 12,379 and 3,095 samples, `random_state=42`.
- **Models:** `LinearRegression`, `Ridge(alpha=1.0)`, and `XGBRegressor` at 500 trees, max depth 8, learning rate 0.1, subsample and colsample 0.8.

Final feature set: `bed`, `bath`, `sqft`, `citi_encoded`, `sqft_per_room`, `total_rooms`, `bath_to_bed_ratio`.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| Modeling | scikit-learn for Linear Regression and Ridge, XGBoost for gradient boosting |
| Plots | Matplotlib, seaborn |
| Environment | Jupyter notebooks |
| Report | Markdown in `index.md`, laid out for GitHub Pages |

## Repository Structure

| Path | Contents |
|---|---|
| `index.md` | the full report: background, methods, every figure, and per model analysis |
| `data/socal2_7_columns.csv` | the seven column dataset, 15,474 rows |
| `midterm_checkpoint/notebooks/` | Linear Regression only, as submitted at the midterm |
| `final_checkpoint/notebooks/` | Linear Regression, Ridge, and XGBoost |
| `images/` | the 20 figures embedded in `index.md` |

## Getting Started

### Prerequisites

- Python 3.12 or newer
- Roughly 600 MB of disk for the virtual environment
- On macOS, the OpenMP runtime that XGBoost links against: `brew install libomp`. Without it the `import xgboost` cell raises `XGBoostError`. Linux and Windows wheels bundle it.

### Installation

```bash
git clone https://github.com/eriklarson12/House-Price-Prediction-in-Southern-California.git
cd House-Price-Prediction-in-Southern-California
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Usage

```bash
.venv/bin/jupyter notebook final_checkpoint/notebooks/ML_House_Price_Prediction_final.ipynb
```

Run all cells. The notebook trains all three models and regenerates every figure in under 30 seconds. It reads the dataset by a path relative to its own directory, so leave the directory layout intact. Running it writes `socal_processed.csv` and a set of PNG files next to the notebook; both are gitignored.

`midterm_checkpoint/notebooks/ML_House_Price_Prediction_midterm.ipynb` runs the same way and covers the Linear Regression baseline on its own.

## Team

Group 118, five contributors.

| Name | Contribution |
|---|---|
| Aditya Sriram | data visualization, feature processing, XGBoost analysis, repository formatting |
| Soham Pati | implementation and coding, Ridge regression analysis, writeup |
| Blake Alford | feature reduction, feature processing and analysis, project schedule |
| Erik Larson | data cleaning, XGBoost model improvement, graph creation, quantitative analysis |
| Eric Joseph | results evaluation, model comparison, writeup |

Work ran in two graded checkpoints. The midterm delivered the preprocessing pipeline and the Linear Regression baseline; the final added Ridge, XGBoost, and the comparative analysis.

## Limitations

- **Seven features, and only one of them is geographic.** There is no lot size, year built, school district, or latitude and longitude, so no model can see the block level variation that drives Southern California prices.
- **Prices are capped at $2,000,000.** The source dataset truncates there, which compresses the top of the market and pushes every model to understate expensive homes.

---

Built by Group 118 at Georgia Tech, fall 2025.
