import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# 🔹 Load datasets
customer_features = pd.read_excel('customer_features.xlsx')
customer_dish = pd.read_excel('customer_dish.xlsx')
cuisine_features = pd.read_excel('cuisine_features.xlsx')
cuisine_dish = pd.read_excel('cuisine_dish.xlsx')

# 🔹 Merge datasets
df = customer_features.merge(customer_dish, left_on="customer_id", right_on="Customer_ID", how="left")
df = df.merge(cuisine_features, on="Preferred Cuisine", how="left")
df = df.merge(cuisine_dish, on="Preferred Cuisine", how="left")

# 🔹 Encoding categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_data = encoder.fit_transform(df.select_dtypes(include=['object']))
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())

df = df.drop(columns=df.select_dtypes(include=['object']).columns)
df = pd.concat([df, encoded_df], axis=1)

# 🔹 Save Encoder
joblib.dump(encoder, "encoder.pkl")

# 🔹 Train Model
X = df.drop(columns=["Dish"])
y = df["Dish"]

model = XGBClassifier()
model.fit(X, y)

# 🔹 Save Model
joblib.dump(model, "xgb_model_dining.pkl")
