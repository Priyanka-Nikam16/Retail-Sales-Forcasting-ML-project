import numpy as np
import pandas as pd

# Number of rows
n = 5000
np.random.seed(42)

# Define categories and other attributes
categories = ['Electronics','Clothing','Grocery','Home','Sports']
regions = ['North','South','East','West']
genders = ['Male','Female']
payment_methods = ['Credit Card','Debit Card','Cash','UPI','Wallet']
store_types = ['Online','Offline']
loyalty_status = ['Bronze','Silver','Gold','Platinum']

# Generate base dataset
data = {
    "Transaction_ID": [f"T{100000+i}" for i in range(n)],
    "Product_ID": [f"P{1000+i}" for i in range(n)],
    "Category": np.random.choice(categories, n, p=[0.2,0.25,0.35,0.1,0.1]),  # skewed
    "Price": np.round(np.clip(np.random.lognormal(mean=4, sigma=0.7, size=n), 5, 2000), 2),
    "Discount": np.round(np.random.uniform(0, 50, n), 2),
    "Promotion_Flag": np.random.choice([0,1], n),
    "Customer_Age": np.random.randint(18, 70, n),
    "Gender": np.random.choice(genders, n),
    "Region": np.random.choice(regions, n, p=[0.5,0.2,0.2,0.1]),  # skewed
    "Month": np.random.randint(1, 13, n),
    "Holiday_Flag": np.random.choice([0,1], n, p=[0.8,0.2]),
    "Weekend_Flag": np.random.choice([0,1], n, p=[0.7,0.3]),
    "Payment_Method": np.random.choice(payment_methods, n),
    "Store_Type": np.random.choice(store_types, n),
    "Loyalty_Status": np.random.choice(loyalty_status, n),
    "Customer_Satisfaction": np.random.randint(1, 6, n),
    "Delivery_Time_Days": np.random.randint(1, 10, n),
    "Return_Flag": np.random.choice([0,1], n, p=[0.9,0.1]),
    "Shipping_Cost": np.round(np.random.uniform(0, 50, n), 2),
    "Stock_Availability": np.random.choice([0,1], n, p=[0.95,0.05]),
    "Season": np.random.choice(['Spring','Summer','Autumn','Winter'], n),
    "Customer_Income": np.random.randint(20000, 150000, n),
    "Review_Score": np.random.randint(1, 6, n)
}

df = pd.DataFrame(data)

# --- Category-specific baseline demand ---
category_effect = df["Category"].map({
    "Grocery": 50,
    "Clothing": 30,
    "Home": 25,
    "Sports": 20,
    "Electronics": 10
})

# --- Units_Sold formula with structured logic ---
df["Units_Sold"] = (
    category_effect
    + (120 - np.log1p(df["Price"]))        # smoother price penalty
    + (df["Discount"]*3)                   # balanced discount effect
    + (df["Promotion_Flag"]*40)            # promotions strong
    + (df["Holiday_Flag"]*70)              # holidays strong
    + (df["Weekend_Flag"]*25)              # weekends moderate
    + (df["Loyalty_Status"].map({'Bronze':10,'Silver':20,'Gold':40,'Platinum':60}))
    + (df["Discount"] * df["Promotion_Flag"] * 2)    # interaction effect
    + (df["Holiday_Flag"] * df["Weekend_Flag"] * 10) # holiday-weekend synergy
    + np.random.normal(0, 1, n)            # very small noise
).astype(int)

# Ensure no negative sales
df["Units_Sold"] = df["Units_Sold"].clip(lower=0)

# --- Seasonality effect ---
seasonal_effect = df['Season'].map({
    'Winter': 15,
    'Summer': 10,
    'Spring': 5,
    'Autumn': 0
})
df['Units_Sold'] += seasonal_effect

# --- Customer behavior ---
df['Return_Flag'] = np.where(
    df['Category'].isin(['Electronics','Clothing']),
    np.random.choice([0,1], n, p=[0.85,0.15]),
    np.random.choice([0,1], n, p=[0.95,0.05])
)
df['Loyalty_Points'] = (df['Units_Sold'] * 0.1).astype(int)

# --- Business metrics ---
df['Revenue'] = (df['Units_Sold'] * (df['Price'] - df['Discount'])).round(2)
margin_map = {'Grocery':0.05,'Clothing':0.2,'Home':0.15,'Sports':0.1,'Electronics':0.25}
df['Profit'] = (df['Revenue'] * df['Category'].map(margin_map)).round(2)

# --- Introduce Outliers (moderate count, controlled) ---
outlier_indices_price = np.random.choice(df.index, 50, replace=False)
df.loc[outlier_indices_price, "Price"] = np.random.uniform(2000, 5000, 50)

outlier_indices_income = np.random.choice(df.index, 50, replace=False)
df.loc[outlier_indices_income, "Customer_Income"] = np.random.randint(500000, 2000000, 50)

outlier_indices_units = np.random.choice(df.index, 50, replace=False)
df.loc[outlier_indices_units, "Units_Sold"] = np.random.randint(500, 1000, 50)  # controlled range

# --- Introduce Missing Values ---
for col in ["Customer_Age","Customer_Income","Review_Score","Discount","Shipping_Cost"]:
    df.loc[np.random.choice(df.index, int(0.05*n), replace=False), col] = np.nan  # 5% missing

# Preview first 10 rows
print(df.head(10))

# Save to CSV
df.to_csv("synthetic_retail_sales_final3.csv", index=False)
print("Dataset saved as synthetic_retail_sales_final3.csv")
