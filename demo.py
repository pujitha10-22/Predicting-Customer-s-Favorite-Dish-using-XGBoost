import os
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder

# ✅ Load required datasets safely
def load_dataset(file_path, default_columns):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=default_columns)
        df.to_excel(file_path, index=False)
    return pd.read_excel(file_path)

customer_dish = load_dataset("customer_dish.xlsx", ["Customer_ID", "Dish", "Cuisine", "Price", "Qty"])
cuisine_features = load_dataset("cuisine_features.xlsx", ["Preferred Cuisine", "Feature1"])
customer_features = load_dataset("customer_features.xlsx", ["customer_id", "Feature1"])
cuisine_dish = load_dataset("cuisine_dish.xlsx", ["Preferred Cuisine", "Dish"])

# ✅ Load models safely
model_files = ["encoder.pkl", "label_encoder.pkl", "xgb_model_dining.pkl"]
if not all(os.path.exists(f) for f in model_files):
    st.error("❌ Missing model files! Ensure all required models exist.")
    st.stop()

encoder = joblib.load('encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model = joblib.load('xgb_model_dining.pkl')

# 🔹 MongoDB Connection
client = MongoClient("mongodb+srv://pujithamodem55:ggXrpjf3z4sWV9sy@cluster0.tzmzm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["hotel_guests"]
new_bookings_collection = db["new_bookings"]

# 🔹 Streamlit UI
st.title("🏨 Hotel Booking Form")

# 🔹 Customer ID Handling
has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))
customer_id = st.text_input("Enter your Customer ID") if has_customer_id == "Yes" else random.randint(10001, 99999)
if has_customer_id == "No":
    st.write(f"Your generated Customer ID: {customer_id}")

# 🔹 User Inputs
name = st.text_input("Enter your name", "")
checkin_date = st.date_input("Check-in Date", min_value=date.today())
checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
stayers = st.number_input("How many stayers in total?", min_value=1, max_value=3, step=1)
preferred_cuisine = st.selectbox("Preferred Cuisine", ["South Indian", "North Indian", "Multi"])
preferred_booking = st.selectbox("Do you want to book through points?", ["Yes", "No"])
special_requests = st.text_area("Any Special Requests? (Optional)", "")

# Submit Button
if st.button("Submit Booking"):
    if name and customer_id:
        # Compute missing features
        stay_duration = (checkout_date - checkin_date).days
        check_in_day = checkin_date.day
        check_out_day = checkout_date.day
        check_in_month = checkin_date.month
        check_out_month = checkout_date.month

        # Default values for missing numerical features
        total_orders = 0
        avg_spend = 0
        total_qty = 0
        avg_stay = 0

        new_data = {
            'customer_id': int(customer_id),
            'Preferred Cuisine': preferred_cuisine,
            'age': age,
            'check_in_date': datetime.combine(checkin_date, datetime.min.time()),
            'check_out_date': datetime.combine(checkout_date, datetime.min.time()),
            'booked_through_points': 1 if preferred_booking == 'Yes' else 0,
            'number_of_stayers': stayers,
            'stay_duration': stay_duration,
            'check_in_day': check_in_day,
            'check_out_day': check_out_day,
            'check_in_month': check_in_month,
            'check_out_month': check_out_month,
            'total_orders': total_orders,
            'avg_spend': avg_spend,
            'total_qty': total_qty,
            'avg_stay': avg_stay
        }

        # Insert into MongoDB
        new_bookings_collection.insert_one(new_data)

        # Convert to DataFrame
        new_df = pd.DataFrame([new_data])

        # Load datasets
        customer_features = pd.read_excel('customer_features.xlsx')
        cuisine_features = pd.read_excel('cuisine_features.xlsx')

        # Merge datasets
        new_df = new_df.merge(customer_features, on="customer_id", how="left")
        new_df = new_df.merge(cuisine_features, on="Preferred Cuisine", how="left")

        # Drop duplicate columns
        new_df = new_df.loc[:, ~new_df.columns.duplicated()]

        # Load encoder
        encoder = joblib.load('encoder.pkl')
        expected_categorical_features = list(encoder.feature_names_in_)

        # Ensure categorical features exist before encoding
        new_df = new_df.reindex(columns=expected_categorical_features, fill_value=0)
        encoded_test = encoder.transform(new_df)
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())

        # Merge Encoded Data
        new_df = new_df.drop(columns=expected_categorical_features, errors="ignore")
        new_df = pd.concat([new_df, encoded_test_df], axis=1)

        # Load expected features list
        expected_features = list(pd.read_excel('features.xlsx')[0])

        # Ensure all required features exist
        for feature in expected_features:
            if feature not in new_df.columns:
                new_df[feature] = 0  # Fill missing features with default values

        # Check for missing features before prediction
        missing_features = [col for col in expected_features if col not in new_df.columns]

        if missing_features:
            st.error(f"❌ ERROR: Missing features before prediction: {missing_features}")
            st.stop()

        # Load model and predict
        label_encoder = joblib.load('label_encoder.pkl')
        model = joblib.load('xgb_model_dining.pkl')

        new_df = new_df[expected_features]
        y_pred_prob = model.predict_proba(new_df)
        dish_names = label_encoder.classes_
        top_3_dishes = dish_names[np.argsort(-y_pred_prob, axis=1)[:, :3]]

        # Display results
        st.success(f"✅ Booking Confirmed for {name} (Customer ID: {customer_id})!")
        st.write(f"**Check-in:** {checkin_date} | **Check-out:** {checkout_date} | **Age:** {age}")
        st.write(f"**Preferred Cuisine:** {preferred_cuisine}")
        st.write(f"🍽 Recommended Dishes: {', '.join(top_3_dishes[0])}")

    else:
        st.warning("⚠️ Please enter your name and Customer ID to proceed!")
