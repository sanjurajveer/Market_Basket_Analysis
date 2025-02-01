!pip install kagglehub
import kagglehub

# Download latest version
path = kagglehub.dataset_download("aslanahmedov/market-basket-analysis")
print("Path to dataset files:", path)

!pip install mlxtend
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Importing make_subplots
from mlxtend.frequent_patterns import apriori,association_rules
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(os.path.join(path,'Assignment-1_Data.csv'),sep=";")
df.head()
#Get Summary
print(df.shape)
print(df.info())

# Converting the date column to date format
df["Date"] = pd.to_datetime(df["Date"], format='%d.%m.%Y %H:%M')

df["Year/Month"] = df["Date"].dt.to_period("M")

# We notice that 'Price' column has commas in the numeric values, let's replace them and convert it to float
df["Price"] = df["Price"].str.replace(",",".").astype(float)

# Dropping Non-product data.
df=df.loc[(df['Itemname']!='POSTAGE')&(df['Itemname']!='DOTCOM POSTAGE')&(df['Itemname']!='Adjust bad debt')&(df['Itemname']!='Manual')]

missing = df.isnull().sum()

print(missing)

#Statistics
df.describe()


df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
#drop rows without item
df = df[df["Itemname"].notnull()]
# Filling missing customer IDs
df = df.fillna('#NV')
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

#EDA
df.info()
import matplotlib.ticker as ticker
# Grouping the data by month and year, and calculating the total sum of sales
monthly_sales = df.groupby('Year/Month')['TotalPrice'].sum()

plt.figure(figsize=(15,6))
monthly_sales.plot(kind='line', marker='o', color='b')
plt.title('Total Sales per Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')

formatter = ticker.FormatStrFormatter('€%.2f')
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(True)
plt.show()

#Unique item analysis
monthly_item = df.groupby("Year/Month")["Itemname"].nunique()

plt.figure(figsize=(15,6))
monthly_item.plot(kind="line",marker="o",color="b")
plt.title("Sum of Unique Items per Month")
plt.xlabel("Month")
plt.ylabel("Sum of Items")
plt.grid(True)
plt.show()

# prompt: select top sold item from dataframe df

# Assuming 'df' is your DataFrame and it has a column named 'Itemname' representing the item and a column named 'Quantity' representing the number of items sold.
top_sold_item = df.groupby('Itemname')['Quantity'].sum().idxmax()

print(f"The top sold item is: {top_sold_item}")

# Preprocessing the data
data_processed = df[['BillNo', 'Itemname']]
data_encoded = pd.get_dummies(data_processed, columns=['Itemname'])
data_encoded.columns = data_encoded.columns.str.replace("Itemname_", "")
basket = data_encoded.groupby('BillNo').sum()

basket.head()
basket[basket > 0] = 1

requent_itemsets = apriori(basket,min_support=0.02,use_colnames=True)


# Confidence represents the likelihood that the consequent item (item bought after) is purchased given the antecedent item(s) (item bought before).
# A confidence of 0.6 implies that the consequent item is purchased in 60% of transactions where the antecedent item(s) are also present, indicating a strong positive relationship.
rules_confidenz = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
print(rules_confidenz.shape)
rules_c = rules_confidenz.round(3)
rules_c.head()

#Lift Analysis
# Lift measures how much more likely the consequent item(s) are purchased when the antecedent item(s) are present compared to when they are not.
# A lift value of 1.0 indicates that the items in the consequent are bought together as often as would be expected by chance.
rules_lift = association_rules(frequent_itemsets, metric="lift", min_threshold= 2.5)
print(rules_lift.shape)
rules_l = rules_lift.round(3)
rules_l.head()

#Suppport Analysis
# Support quantifies the frequency with which a rule occurs in the dataset.
# A support of 0.05 means that the rule occurs in at least 5% of transactions, signifying its significance in the dataset.
rules_support = association_rules(frequent_itemsets, metric="support", min_threshold=0.03)
print(rules_support.shape)
rules_s = rules_support.round(3)
rules_s.head()



