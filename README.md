# Retail-Sales-Forcasting-ML-project


Project Overview
  - This project analyzes a retail dataset to forecast Units Sold and generate actionable insights for inventory planning, promotions, and customer engagement. Using Exploratory Data Analysis (EDA) and machine learning models (Random Forest, Gradient Boosting, XGBoost), the study identifies the key drivers of sales and evaluates predictive performance.

Objectives
- Forecast retail sales with high accuracy.
- Identify top contributing features driving sales.
- Provide business recommendations for inventory and promotion strategies.

Dataset
Size: 5,000 records.

Features: 
  - Category, Region, Store_Type, Loyalty_Status, Discount, Season, Payment_Method etc.

Target Variable: Units_Sold.

Data Preparation
  - Dropped non-informative columns (Transaction_Id, Product_ID).
  - Handled 1250 null values using imputation:
        - Numeric → mean/median .
        - Categorical → mode (most frequent).

Encoded low-cardinality categorical features (Region, Store_Type etc).

Applied log transformation to Price, Customer_Income variable for variance stabilization.

Exploratory Data Analysis (EDA)
  - Units Sold Distribution: Concentrated between 200–450 units, right-skewed with occasional bulk purchases.
  - Category Trends: Essentials and electronics show consistently higher demand.
  - Seasonality: Peaks during holidays and promotional periods.
  - Regional Differences: Certain regions and store types outperform others.

Modeling

Baseline Model: Random Forest Regressor.

Hyperparameter Tuning: GridSearchCV with MAE optimization.

Evaluation Metrics: R², MAE, MSE.

Feature Importance: Discounts, Loyalty_Status, Seasonality, Category, Region.

Comparisons: Linear Regression, Random Forest, Decision Tree

Key Drivers of Sales: Discounts, Loyalty programs, Seasonality, Category, Region.

Insights & Interpretation
  - Discounts & Promotions: Strong positive impact on units sold.
  - Loyalty Programs: Higher tiers correlate with repeat purchases and larger baskets.
  - Seasonality: Prime demand during festive seasons and promotions.
  - Category & Region: Certain product types and regions dominate sales.

Business Implications
  - Inventory Planning: Stock primarily in the 200–450 units range.
  - Promotion Strategy: Optimize discounts and loyalty programs to balance demand vs. margin.
  - Regional Strategy: Tailor campaigns by store type and region.
  - Seasonal Strategy: Align inventory with holiday peaks and promotional events.

Technologies Used
  - Python: pandas, numpy, scikit-learn, seaborn, matplotlib

Machine Learning: Random Forest, Linear Regression,Decision Tree

EDA Tools: Visualization libraries

Conclusion
  - The dataset highlights how discounts, loyalty programs, seasonality, and category/region differences drive retail sales. While Random Forest achieved strong explanatory power (R² ≥ 0.75), MAE reduction requires further feature engineering and model comparison. The project provides actionable insights for inventory planning, promotions, and customer engagement.
