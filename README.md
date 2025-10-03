# ML Project Proposal: House Price Prediction in Southern California 

### Group 118  
**Aditya Sriram, Soham Pati, Blake Alford, Erik Larson, Eric Joseph**

---

## 1. Introduction / Background 

The real estate market in Southern California is always changing and one of the most expensive markets in the country. Home prices vary because of high demand and low supply. Predicting those prices becomes difficult, as there are so many factors influencing these values such as location, number of bedrooms and bathrooms, lot size, square footage, and proximity to amenities and major employment centers. 

### Dataset Description 

Our dataset contains over 15000 house prices and data. For each house we keep track of the address and city. As well as the number of rooms, bathrooms and square feet.

LINK: https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal/code

### Literature Review

---

## 2. Problem Definition 


**Problem:** Are we able to utilize historical property data from 2021 to then create machine learning models to predict the price of a home based on property features?

**Motivation:**  Housing affordability is one of the biggest issues in all of Southern California. Creating accurate price prediction models could help buyers figure out fair market values, help realtors with price strategies and also support lawmakers in creating proper legislation using market trends.

---

## 3. Methods  

### Data Preprocessing 

1. **Missing value imputation:** sklearn.impute.SimpleImputer(strategy="median") for numeric, SimpleImputer(strategy="most_frequent") for categorical. This will ensure that data that is missing values or data that is skewed is not lost.
2. **Feature Engineering:** We can add features for half baths, and also for other important metrics like price per square foot, sqaure foot per room, etc. This will also allow us to convert our data so it more sense and can be understood better.
3. **Categorical encoding:** sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'). For features which are categorical(city, etc.) we will transform it into a numerical representation. This will alow for models like linear regression to treat cities like different entities without a ranking or order.
4. **Pipelining:** sklearn.pipeline.Pipeline to combine transforms + models for reproducibility and grid search. We can chain together the above preprocessing steps into one pipeline for easy accessibility and convenient runs.

### Machine Learning Models 

**Regression (Supervised)**
**Model/Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)  
A simple model that predicts housing prices as a weighted combination of input features, assuming a linear relationship.

**Ensemble Learning (Supervised)**
**Model/Algorithm:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)  
An ensemble of decision trees that averages their predictions to capture nonlinear relationships and reduce overfitting.

**Gradient Boosting (Supervised)**
**Model/Algorithm:** XGBoost Regressor (`xgboost.XGBRegressor`)  
A gradient boosting model that builds trees sequentially to correct previous errors, often achieving high accuracy on structured data.

**Clustering (Unsupervised)**
**Model/Algorithm:** K-Means (`sklearn.cluster.KMeans`)  
Groups houses into clusters based on similarity of features, helping identify natural market segments.

**Density-Based Clustering (Unsupervised)**
**Model/Algorithm:** DBSCAN (`sklearn.cluster.DBSCAN`)  
Finds clusters of homes based on density, capturing irregularly shaped neighborhoods or sparse vs. dense regions.

**Dimensionality Reduction (Unsupervised)**
**Model/Algorithm:** Principal Component Analysis (PCA) (`sklearn.decomposition.PCA`)  
Reduces the number of features while retaining most of the variance, simplifying the dataset and highlighting key patterns.


---

## 4. Potential Results and Discussion 

### Quantitative Metrics

**Supervised Metrics:**
1. **Root Mean Squared Error (RMSE)** [1]: Measures the square root of the average squared difference between predicted and actual prices. RMSE penalizes larger errors more heavily, making it useful for identifying significant mispredictions in housing prices.


2. **Mean Absolute Error (MAE):** Calculates the average absolute difference between predicted and actual prices. MAE is more interpretable and less sensitive to outliers, providing a complementary perspective to RMSE.


3. **R² Score:** Represents the proportion of variance in the target variable explained by the model. Higher values indicate better model fit and predictive power.

**Unsupervised Learning Metrics:**

1. **Silhouette Score:** Evaluates clustering quality by measuring how similar each data point is to its own cluster compared to other clusters. Higher scores indicate well-separated, distinct clusters.

2. **Explained Variance Ratio (for PCA):** Quantifies how much of the dataset’s total variance is captured by the selected principal components, helping assess dimensionality reduction effectiveness.


### Project Goals 

1. Achieve RMSE within 10–15% of the average house price in the dataset, indicating practical predictive accuracy.


2. Demonstrate that unsupervised features enhance supervised regression performance by providing additional structured information.


3. Assess model interpretability, identifying key drivers of price such as sqft, bedrooms, and neighborhood cluster segments.


4. Consider ethical implications, ensuring predictions do not reinforce biases, particularly when sensitive attributes (e.g., demographics) could influence results.

### Expected Results

1. Gradient Boosting and Random Forest models are expected to outperform linear regression due to their ability to capture nonlinear interactions and complex feature relationships.


2. Adding unsupervised features from PCA and clustering is expected to improve predictive accuracy relative to using only raw property features.


3. Outlier detection and handling will likely reduce RMSE by removing mislabeled or extreme-value records that could skew model predictions.

---

## 5. References 

- [1] Marcinrutecki, “Regression models evaluation metrics,” Kaggle, https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-metrics/notebook (accessed Oct. 2, 2025).  
- [2] A. Ihre and I. Engström, *Predicting House Prices with Machine Learning Methods*, Examensarbete inom teknik, Grundnivå, 15 HP, School of Electrical Engineering and Computer Science, KTH, Stockholm, Sweden, 2019.  
- [3] Q. Truong, M. Nguyen, H. Dang, and B. Mei, “Housing Price Prediction via Improved Machine Learning Techniques,” *Procedia Computer Science*, vol. 174, pp. 433–442, 2020, doi: 10.1016/j.procs.2020.06.111.

---

## Gantt Chart & Contributions

https://docs.google.com/spreadsheets/d/1sEn3y5obKaQtLnEJNnsmjzddWluIJcmOrhaOsC1w4q8/edit?usp=sharing

| Name      | Proposal Contributions |
|-----------|-------------------------|
| Aditya Sriram   | Problem Definition, GitHub Page  |
| Soham Pati      | Methods, Dataset |
| Blake Alford    | Literature Review, Gantt Chart |
| Erik Larson     | Youtube Video, Results |
| Eric Joseph     | Youtube Video, Methods |

