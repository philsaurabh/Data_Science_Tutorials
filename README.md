# Data Science Tutorials

Hands-on Jupyter notebooks I prepared and used for teaching **Python, statistics, machine learning, time-series forecasting and image processing**. They were written for B.Tech and BS/BSE classes. Each notebook is self-contained, pairs a short explanation with code, and works on a small dataset that is included in the repository in most cases.

Most notebooks are designed for **Google Colab**: open one, upload the CSV it needs (or mount Google Drive), and run the cells top to bottom.

---

## Contents

| Folder | Topics | Notebooks |
|---|---|---|
| [`Python_fundamentals/`](#1-python-fundamentals) | Python basics, NumPy, TensorFlow tensors, pandas, k-NN | 4 |
| [`Tutorials-main/`](#2-statistics-regression-and-pca) | pandas, linear and multiple regression, hypothesis testing, PCA | 7 |
| [`MLDS_Demos-main/`](#3-machine-learning-classification-and-clustering) | Classification (logistic regression, LDA, SVM, k-NN, Naive Bayes, decision trees, XGBoost) and clustering | 9 |
| [`Machine-Learning-Tutorials-main/`](#3-machine-learning-classification-and-clustering) | Same notebooks as `MLDS_Demos-main/`, plus a Titanic assignment | 10 |
| [`Knowings-main/`](#4-time-series-analysis-and-forecasting) | Time-series analysis and forecasting (statistical, ML and deep learning), plus extras | 18 |
| [`Image-Processing-tutorials-main/`](#5-image-processing) | NumPy and image processing with Pillow, OpenCV and Keras | 2 |
| [`Working-with-images-main/`](#6-working-with-video) | Video format conversion, pdf merger and key-frame extraction | 3 |

## Getting started

**In Colab (recommended).** Open a notebook on GitHub and use its *Open in Colab* badge, or go to [colab.research.google.com](https://colab.research.google.com) → *GitHub* and paste the notebook's URL. Then upload the CSV it reads from the same folder.

**Locally**

```bash
git clone https://github.com/philsaurabh/Data_Science_Tutorials.git
cd Data_Science_Tutorials
python -m venv .venv && source .venv/bin/activate
pip install jupyter numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
jupyter notebook
```

Some notebooks need extra packages, listed under [Requirements](#requirements).

---

## 1. Python fundamentals

`Python_fundamentals/`

| Notebook | What it covers |
|---|---|
| `FundamentalsOfPython.ipynb` | Data types (strings, lists, tuples, sets, dictionaries), conditionals and exceptions, loops, a switch-case emulation, functions, file I/O, JSON and pickle |
| `NumpyTensorflow.ipynb` | Array dimensions, indexing, data types, slicing, copy vs view, reshaping, side by side in NumPy and TensorFlow |
| `Pandas.ipynb` | Why pandas; `loc` vs `iloc`; filtering rows on one or more conditions; selecting row ranges |
| `Copy_of_KNN.ipynb` | k-Nearest Neighbours with scikit-learn: distance metrics, the curse of dimensionality, choosing *k* (diabetes dataset) |

## 2. Statistics, regression and PCA

`Tutorials-main/` — datasets: `Advertising.csv`, `Credit.csv`

| Notebook | What it covers |
|---|---|
| `Pandas_Tutorial.ipynb` | Loading data, exploring a DataFrame, selection and summary statistics (Advertising data) |
| `Pandas_Tutorial_2.ipynb` | More pandas tasks on the Credit data: age groups, filtering and aggregation |
| `Testing_tutorial.ipynb` | Hypothesis testing with SciPy and statsmodels: PPF, CDF, critical values, two-sample *t*-test, one-way ANOVA, chi-square test |
| `Linear_Regression.ipynb` | Simple linear regression with OLS (statsmodels), an ANOVA table, and model evaluation |
| `Multiple_Linear_Regression.ipynb` | Multiple linear regression with scikit-learn, with visualisation |
| `Principal_Component_Analysis.ipynb` | PCA from scratch: standardisation, covariance, eigen-decomposition, projection |
| `PCA_Sklearn.ipynb` | PCA with scikit-learn: `components_`, `explained_variance_` |

## 3. Machine learning: classification and clustering

`MLDS_Demos-main/` (B.Tech class) and `Machine-Learning-Tutorials-main/` (BS/BSE class) contain the same notebooks and datasets. The second folder also has the Titanic assignment.

| Notebook | Dataset | What it covers |
|---|---|---|
| `Machine_Learning_Tutorial_1.ipynb` | `Telco_Customer_Churn.csv` | End-to-end churn prediction: cleaning, encoding, train/test split, several classifiers including XGBoost |
| `Credit_Logistic_regression.ipynb` | `Credit.csv` | Logistic regression, an ANOVA table, extra tasks |
| `Credit_LDA_SVM.ipynb` | `Credit.csv` | Naive Bayes and SVM classifiers |
| `LDA_IRIS.ipynb` | `iris.csv` | Linear discriminant analysis and SVM on Iris |
| `LDA_Glass.ipynb` | `glass.csv` | LDA, SVM, and SVM on LDA-reduced features for glass types |
| `Stockmarket_LDA.ipynb` | `Weekly.csv` | LDA to predict market direction from lagged weekly returns (1,089 weeks over 21 years) |
| `Stockmarket_Classifiers.ipynb` | `Weekly.csv` | Comparing classifiers on the same task |
| `Stockmarket_SVM_KNN_NB_DT_Classifiers.ipynb` | `Weekly.csv` | SVM, k-NN, Gaussian Naive Bayes and decision trees |
| `Clustering.ipynb` | `CLV.csv` | Customer segmentation with k-means (customer lifetime value data) |
| `Assignment.ipynb` *(ML-Tutorials only)* | `titanic.csv` | Exercise: preprocess the Titanic data, split it, and compare classifiers |

`diabetes.csv` and `heart.csv` are also included for practice.

## 4. Time-series analysis and forecasting

`Knowings-main/` — datasets: `AirPassengers.csv`, `Apple_stock.csv`, `Facebook_stock.csv`, `Google_stock.csv`, `ibm_stock.csv`, `covid_19_india.csv`

| Notebook | What it covers |
|---|---|
| `Time Series (1).ipynb` | NumPy refresher for time-series work |
| `TS_matplotlib.ipynb` | Plotting with matplotlib: markers, line styles, colours, titles, subplots |
| `TS.ipynb` | Exercise: extract year, month, week and hour features from timestamps |
| `Time series_Lab_exercise.ipynb` · `Time Series (3).ipynb` | Lab: load and visualise AirPassengers, rolling statistics, error metrics |
| `Time Series Tutorial.ipynb` | Stationarity, decomposition, ACF/PACF, ARIMA and Prophet, then vanilla, stacked and bidirectional LSTMs |
| `Project tutorial.ipynb` | Time-series transforms: additive and multiplicative decomposition, and forecasting basics |
| `Project Forecasting_models.ipynb` · `app.ipynb` | A catalogue of forecasting methods: SES, Holt-Winters, AR, MA, ARIMA/SARIMA, Prophet, XGBoost and LightGBM regressors, LSTM, DeepAR (GluonTS) |
| `apple.ipynb` · `google.ipynb` · `facebook.ipynb` · `ibm.ipynb` | That pipeline applied to each company's stock prices |
| `Time Series (4).ipynb` | COVID-19 testing in India: missing-value imputation, one-way ANOVA |
| `Kmedoids Demonstration.ipynb` | k-medoids clustering implemented from scratch |
| `Plot wave.ipynb` | Heart-sound `.wav` files: spectrograms and classifiers (SVM, Keras) on the PASCAL heartbeat data |
| `supervised-contrastive-learning` | Keras example: supervised contrastive learning vs a baseline classifier on CIFAR-10 (a notebook file without the `.ipynb` extension) |
| `Upload files.ipynb` | Helper for uploading files to Colab |

## 5. Image processing

`Image-Processing-tutorials-main/`

| Notebook | What it covers |
|---|---|
| `Numpy_and_basics_tutorial.ipynb` | Python lists, tuples, sets and dictionaries, then NumPy arrays and operations |
| `Image_processing_tutorial_numpy.ipynb` | Images as NumPy arrays: opening and inspecting images with Pillow, colour channels and conversion with OpenCV, cropping, resizing, and loading image data for Keras |

## 6. Working with video to image and pdf

`Working-with-images-main/`

| Notebook | What it covers |
|---|---|
| `Folderwise_Video_format_conversion.ipynb` | Batch-convert every video in a folder to another format (moviepy) |
| `Keyframe_Extraction.ipynb` | Extract key frames from every video in a folder with Katna |
| `pdfmerger.ipynb` | Merging pdfs |
---

## Requirements

| Needed for | Packages |
|---|---|
| All notebooks | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `statsmodels` |
| Gradient boosting | `xgboost`, `lightgbm` |
| Hypothesis-testing extras | `bioinfokit` |
| Deep learning | `tensorflow` (includes Keras), `tensorflow-addons` for the contrastive-learning example |
| Forecasting | `prophet` (imported as `fbprophet` in these notebooks), `pmdarima`, `gluonts`, `pandas-datareader`, `shap` |
| Images and video | `pillow`, `opencv-python`, `moviepy`, `katna` |

These notebooks were written in 2020–2021. `fbprophet` has since been renamed `prophet`: install `prophet` and change `from fbprophet import Prophet` to `from prophet import Prophet`.

## Datasets not included

Some notebooks load files that are not in the repository. Download them from the source named in the notebook, or swap in your own data:

- `Knowings-main/`: `cancer.csv` (k-medoids), `baseball.csv` (plotting), `StatewiseTestingDetails.csv` and `population_india_census2011.csv` (COVID-19 India, Kaggle), `daily-Minimum-temperature.csv`, `monthly-robberies.csv` and `yearly-water-usage.csv` (classic forecasting datasets), and `CSV.csv` (forecasting project input).
- `Python_fundamentals/`: `PlayersAll.csv`.
- `Plot wave.ipynb` reads heart-sound `.wav` files from Google Drive ([PASCAL Classifying Heart Sounds Challenge](http://www.peterjbentley.com/heartchallenge/)).

## Acknowledgements

Some notebooks build on outside material, credited in the notebooks themselves:

- `supervised-contrastive-learning` is adapted from the [Keras code example](https://keras.io/examples/vision/supervised-contrastive-learning/).
- `Time Series (3).ipynb` follows a CS575 lab exercise.
- Several `Python_fundamentals` notebooks originate from the DANeS-Lab practice repository.
- The datasets are public: ISLR (Advertising, Credit, Weekly), UCI (Iris, Glass, Heart, Diabetes), Kaggle (Telco churn, Titanic, AirPassengers, COVID-19 India), and stock prices from public market data.
