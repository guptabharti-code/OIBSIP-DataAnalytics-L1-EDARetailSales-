# ============================================================
# TASK 1: EXPLORATORY DATA ANALYSIS ON RETAIL SALES DATA
# ============================================================

# Objective:
# To perform a thorough Exploratory Data Analysis on a retail
# sales dataset to identify sales patterns, customer behaviour,
# product performance and actionable business insights.

# Technology:
# Python, Pandas, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    r"C:\Users\DELL\.matplotlib\eda.py\Vietnam_Electronics_Retail_Sales_and_Profit_Dataset.csv"
)

# Display first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# -----------------------------
# DATA CLEANING
# -----------------------------

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Convert Unit COGS from string to numeric
df["Unit COGS (VND)"] = (
    df["Unit COGS (VND)"]
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Convert Unit Selling Price from string to numeric
df["Unit Selling Price (VND)"] = (
    df["Unit Selling Price (VND)"]
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Convert Discount Rate from percentage to decimal
df["Discount Rate"] = (
    df["Discount Rate"]
    .str.replace("%", "", regex=False)
    .astype(float) / 100
)

# Fill missing categorical values
df["Sales Channel"] = df["Sales Channel"].fillna("Unknown")
df["Customer Type"] = df["Customer Type"].fillna("Unknown")

# Fill missing discount rate with 0
df["Discount Rate"] = df["Discount Rate"].fillna(0)

print("\nData cleaning completed!")
print(df.isnull().sum())
# -----------------------------
# CREATE NEW FEATURES
# -----------------------------

# Gross Revenue before discount
df["Gross Revenue"] = (
    df["Quantity Sold"] * df["Unit Selling Price (VND)"]
)

# Discount amount
df["Discount Amount"] = (
    df["Gross Revenue"] * df["Discount Rate"]
)

# Net Sales after discount
df["Net Sales"] = (
    df["Gross Revenue"] - df["Discount Amount"]
)

# Total Cost
df["Total Cost"] = (
    df["Quantity Sold"] * df["Unit COGS (VND)"]
)

# Profit
df["Profit"] = (
    df["Net Sales"] - df["Total Cost"]
)

print("\nNew columns created:")
print(df[[
    "Quantity Sold",
    "Gross Revenue",
    "Discount Amount",
    "Net Sales",
    "Total Cost",
    "Profit"
]].head())
print("\nFinal Data Types:")
print(df.dtypes)

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:")
print(df.shape)
# -----------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------

numerical_columns = [
    "Quantity Sold",
    "Unit COGS (VND)",
    "Unit Selling Price (VND)",
    "Discount Rate",
    "Gross Revenue",
    "Discount Amount",
    "Net Sales",
    "Total Cost",
    "Profit"
]

print("\nMEAN:")
print(df[numerical_columns].mean())

print("\nMEDIAN:")
print(df[numerical_columns].median())

print("\nMODE:")
print(df[numerical_columns].mode().iloc[0])

print("\nSTANDARD DEVIATION:")
print(df[numerical_columns].std())
print("\nComplete Descriptive Statistics:")
print(df[numerical_columns].describe())
print(df.dtypes)
print(df.isnull().sum())
print(df[numerical_columns].describe())
# -----------------------------------
# MONTHLY SALES TREND
# -----------------------------------

# Extract month from Order Date
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# Calculate total net sales for each month
monthly_sales = df.groupby("Month")["Net Sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# Plot monthly sales
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Net Sales Trend", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Net Sales (VND)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# Observation:
# The monthly sales trend shows fluctuations in net sales across different
# months. Higher-sales months indicate periods of stronger customer demand.
# These trends can help the business plan inventory and promotional campaigns.
# -----------------------------------
# QUARTERLY SALES TREND
# -----------------------------------

# Extract quarter from Order Date
df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)

# Calculate quarterly net sales
quarterly_sales = df.groupby("Quarter")["Net Sales"].sum()

print("\nQuarterly Sales:")
print(quarterly_sales)

# Plot quarterly sales
plt.figure(figsize=(10, 6))

plt.plot(
    quarterly_sales.index,
    quarterly_sales.values,
    marker="o",
    linewidth=2
)

plt.title("Quarterly Net Sales Trend", fontsize=16)
plt.xlabel("Quarter")
plt.ylabel("Net Sales (VND)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# Observation:
# Quarterly sales provide a broader view of business performance.
# The highest-performing quarter indicates a period of stronger demand,
# which can help the company plan inventory and marketing activities.
# -----------------------------------
# CUSTOMER TYPE ANALYSIS
# -----------------------------------

customer_type = df["Customer Type"].value_counts()

print("\nCustomer Type Distribution:")
print(customer_type)

plt.figure(figsize=(8, 6))

customer_type.plot(
    kind="bar"
)

plt.title("Customer Type Distribution", fontsize=16)
plt.xlabel("Customer Type")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
# Observation:
# The customer type distribution shows the number of orders generated
# by different customer segments. This information can help the company
# develop separate marketing and pricing strategies for different
# customer groups.

# Dataset Limitation:
# The dataset does not contain customer Age or Gender information.
# Therefore, age-group and gender analysis could not be performed.
# Customer behaviour was analyzed using the available Customer Type variable.

# -----------------------------------
# TOP 10 BEST-SELLING PRODUCTS
# -----------------------------------

top_products = (
    df.groupby("Product Name")["Quantity Sold"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Best-Selling Products:")
print(top_products)

# Plot
plt.figure(figsize=(12, 7))

top_products.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Best-Selling Products", fontsize=16)
plt.xlabel("Quantity Sold")
plt.ylabel("Product Name")

plt.tight_layout()
plt.show()
# Observation:
# The top 10 products are the products with the highest sales volume.
# These products should receive priority in inventory planning to
# reduce stockout risk and maintain customer satisfaction.
# -----------------------------------
# REVENUE BY PRODUCT CATEGORY
# -----------------------------------

category_revenue = (
    df.groupby("Product Category")["Net Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by Product Category:")
print(category_revenue)

# Plot
plt.figure(figsize=(10, 6))

category_revenue.plot(kind="bar")

plt.title("Revenue by Product Category", fontsize=16)
plt.xlabel("Product Category")
plt.ylabel("Net Sales (VND)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Observation:
# Product categories contribute different amounts of revenue.
# The highest-revenue category is the strongest contributor to overall
# sales and should receive appropriate attention in inventory and marketing.
# -----------------------------------
# CORRELATION HEATMAP
# -----------------------------------

correlation_columns = [
    "Quantity Sold",
    "Unit COGS (VND)",
    "Unit Selling Price (VND)",
    "Discount Rate",
    "Gross Revenue",
    "Discount Amount",
    "Net Sales",
    "Total Cost",
    "Profit"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Plot heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix of Numerical Variables", fontsize=16)

plt.tight_layout()
plt.show()

# Observation:
# The correlation heatmap shows relationships between numerical variables.
# Strong positive correlations indicate variables that tend to increase
# together, while negative correlations indicate an inverse relationship.
# The relationships between quantity, revenue, cost and profit provide
# useful information about factors affecting business performance.
# -----------------------------------
# ADDITIONAL VISUALIZATION:
# PROFIT BY REGION
# -----------------------------------

region_profit = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit by Region:")
print(region_profit)

# Plot
plt.figure(figsize=(10, 6))

region_profit.plot(kind="bar")

plt.title("Profit by Region", fontsize=16)
plt.xlabel("Region")
plt.ylabel("Profit (VND)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Observation:
# Profitability differs across regions. A region generating high sales
# does not necessarily generate the highest profit. This comparison can
# help management identify regions that are more profitable and evaluate
# pricing, costs and discount strategies.
# -----------------------------------
# SALES BY CHANNEL
# -----------------------------------

channel_sales = (
    df.groupby("Sales Channel")["Net Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Channel:")
print(channel_sales)

plt.figure(figsize=(9, 6))

channel_sales.plot(kind="bar")

plt.title("Net Sales by Sales Channel", fontsize=16)
plt.xlabel("Sales Channel")
plt.ylabel("Net Sales (VND)")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.show()

# Observation:
# The sales channel analysis identifies which channels contribute the
# most to net sales. The company can prioritize high-performing channels
# while improving the performance of weaker channels.
# ==================================================
# BUSINESS RECOMMENDATIONS
# ==================================================

print("\n" + "=" * 60)
print("BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("""
1. Focus on high-demand products:
   Samsung Galaxy S24 Ultra and other top-selling products should be
   prioritized in inventory planning to reduce stockout risk.

2. Prepare for Q4 demand:
   Q4 generated the highest quarterly net sales. The company should
   increase inventory and marketing activities before the fourth quarter.

3. Improve low-performing periods:
   May recorded the lowest monthly net sales. Targeted discounts,
   promotional campaigns and seasonal offers can be used to increase
   demand during weaker months.

4. Strengthen profitable regions:
   Regional profit analysis should be used to identify high-profit
   markets and allocate marketing and distribution resources accordingly.

5. Optimize sales channels:
   The company should invest more in high-performing sales channels
   while identifying reasons for weaker channel performance.
""")
# ==================================================
# CONCLUSION
# ==================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
The exploratory data analysis of the Vietnam Electronics Retail Sales
dataset revealed important patterns in sales, products, customers,
regions and sales channels.

Monthly analysis showed significant fluctuations in net sales, with
October being the strongest month and May being the weakest. Quarterly
analysis showed that Q4 generated the highest net sales.

Product analysis identified the best-selling products based on quantity
sold, while category analysis showed differences in revenue contribution.
Customer analysis showed that individual customers generated more orders
than corporate clients.

The correlation analysis provided insights into relationships between
quantity, costs, revenue and profit. Regional profit analysis also helped
identify differences in profitability across markets.

These findings can support better inventory planning, seasonal marketing,
regional strategy, sales-channel optimization and overall data-driven
business decision-making.

Note: The dataset does not contain customer age or gender information,
so age-group and gender analysis could not be performed.
""")