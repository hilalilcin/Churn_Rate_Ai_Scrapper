import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv('dataset.csv')

def clean_churn(churn):
    if isinstance(churn,str) and '%' in churn:
        return float(churn.replace('%',"").strip())
    elif isinstance(churn,(int,float)):
        return churn
    return np.none

df["Churn Rate"] = df["Churn Rate"].apply(clean_churn)

def parse_money(value):
    if isinstance(value,str):
        value = value.upper()
        value = re.sub(r"[^\d\.MBK]", "", value)
        try:
            if 'M' in value:
                return float(value.replace("M", "")) * 1e6
            elif 'B' in value:
                return float(value.replace("B", "")) * 1e9
            elif 'K' in value:
                return float(value.replace("K", "")) * 1e3
            else:
                return float(value)
        except:
            return np.nan
        
    elif isinstance(value,(int,float)):
        return value
    return np.nan

money_cols = ["Revenue", "Customer Lifetime Value (CLV)", "Customer Acquisition Cost (CAC)"]
for col in money_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_money)

def parse_employees(val):
    if isinstance(val, str):
        val = val.replace("+", "")
        nums = re.findall(r"\d+", val)
        if nums:
            return int(nums[0])
    elif isinstance(val, (int, float)):
        return val
    return np.nan

if "Number of Employees" in df.columns:
    df["Number of Employees"] = df["Number of Employees"].apply(parse_employees)


# Converting Percentage Columns to Numeric Values
def parse_percentage(x):
    if isinstance(x, str) and "%" in x:
        return float(x.replace("%", "").strip())
    elif isinstance(x, (int, float)):
        return x
    return np.nan

percent_cols = ["Revenue Growth Rate (YoY)", "Profit Margin"]
for col in percent_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_percentage)


features = [
    "Number of Employees", "Revenue", "Sector",
    "Customer Acquisition Cost (CAC)", "Customer Lifetime Value (CLV)",
    "Revenue Growth Rate (YoY)", "Profit Margin"
]

# Label Encoding only for categorical features
df_all = df.copy()
label_encoders = {}

for col in features:
    if col in df_all.columns and df_all[col].dtype == object:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))
        label_encoders[col] = le


# Train test Split
df_train = df_all[df_all["Churn Rate"].notna()]
df_test = df_all[df_all["Churn Rate"].isna()]
usable_features = [col for col in features if col in df_all.columns and df_train[col].notna().sum() > 0 and df_test[col].notna().sum() > 0]


X_train = df_train[usable_features]
y_train = df_train["Churn Rate"]
X_test = df_test[usable_features]


# Filling missing values 
imputer = SimpleImputer(strategy="mean")
X_train_array = imputer.fit_transform(X_train)
X_test_array = imputer.transform(X_test)

X_train = pd.DataFrame(X_train_array, columns=usable_features)
X_test = pd.DataFrame(X_test_array, columns=usable_features)

# Training Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Making Prediction 
predicted = model.predict(X_test)
df.loc[df_test.index, "Churn Rate"] = predicted.round(2) 
df["Churn Rate"] = df["Churn Rate"].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "")



df.to_csv("dataset_updated.csv", index=False)
print("Predictions completed. > File: dataset_updated.csv")