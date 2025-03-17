import os
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from pymongo import MongoClient

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
email = st.text_input("Enter your email", "")  # Added email input
checkin_date = st.date_input("Check-in Date", min_value=date.today())
checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
stayers = st.number_input("How many stayers in total?", min_value=1, max_value=3, step=1)
preferred_cuisine = st.selectbox("Preferred Cuisine", ["South Indian", "North Indian", "Multi"])
preferred_booking = st.selectbox("Do you want to book through points?", ["Yes", "No"])
special_requests = st.text_area("Any Special Requests? (Optional)", "")

# 🔹 Discount Mapping
discount_map = {
    "South Indian": 0.80,  # 20% OFF
    "North Indian": 0.85,  # 15% OFF
    "Multi": 1.00          # No discount
}
discount_percent = discount_map.get(preferred_cuisine, 1.0)

# Submit Button
if st.button("Submit Booking"):
    if name and email and customer_id:
        stay_duration = (checkout_date - checkin_date).days
        new_data = {
            'customer_id': int(customer_id),
            'name': name,
            'email': email,  # Store email
            'Preferred Cuisine': preferred_cuisine,
            'age': age,
            'booked_through_points': 1 if preferred_booking == 'Yes' else 0,
            'number_of_stayers': stayers,
            'stay_duration': stay_duration,
            'check_in_day': checkin_date.day,
            'check_out_day': checkout_date.day,
            'check_in_month': checkin_date.month,
            'check_out_month': checkout_date.month,
            'total_orders': 0,
            'avg_spend': 0,
            'total_qty': 0,
            'avg_stay': 0
        }

        # Insert into MongoDB
        new_bookings_collection.insert_one(new_data)
        new_df = pd.DataFrame([new_data])

        # Merge datasets
        new_df = new_df.merge(customer_features, on="customer_id", how="left")
        new_df = new_df.merge(cuisine_features, on="Preferred Cuisine", how="left")
        new_df = new_df.loc[:, ~new_df.columns.duplicated()]

        # Ensure all required features exist
        expected_features = list(pd.read_excel('features.xlsx')[0])
        for feature in expected_features:
            if feature not in new_df.columns:
                new_df[feature] = 0
        new_df = new_df[expected_features]

        # Predict preferred dishes
        y_pred_prob = model.predict_proba(new_df)
        dish_names = label_encoder.classes_
        top_3_dishes = dish_names[np.argsort(-y_pred_prob, axis=1)[:, :3]]

        # ✅ Filter and apply discounts based on selected cuisine
        discounted_prices = {}

        if "Cuisine" in customer_dish.columns and "Dish" in customer_dish.columns:
            customer_dish["Dish"] = customer_dish["Dish"].astype(str).str.lower()
            customer_dish["Cuisine"] = customer_dish["Cuisine"].astype(str).str.lower()
        else:
            st.error("❌ 'Dish' or 'Cuisine' column missing in customer_dish.xlsx! Please check the file.")
            st.stop()

        for dish in top_3_dishes[0]:  
            dish_lower = dish.lower()
            dish_info = customer_dish[customer_dish["Dish"] == dish_lower]

            if not dish_info.empty:
                cuisine = dish_info["Cuisine"].values[0]
                original_price = dish_info["Price"].values[0]

                # ✅ Apply discount only if dish matches preferred cuisine
                if preferred_cuisine.lower() in cuisine:
                    discounted_prices[dish_lower] = (original_price, round(original_price * discount_percent, 2))

        # 🎉 Display Booking Confirmation
        st.success(f"✅ Booking Confirmed for {name} (Customer ID: {customer_id})!")
        st.write(f"📩 **A confirmation email will be sent to:** {email}")
        st.write(f"🛏️ **Check-in:** {checkin_date} | **Check-out:** {checkout_date}")
        st.write(f"🍽️ **Preferred Cuisine:** {preferred_cuisine}")

        # 🎉 Special Discounts Section
        if discounted_prices:
            discount_value = int((1 - discount_percent) * 100)
            st.write("\n### 🎉 Special Discounts on Your Favorite Dishes! 🎉")
            st.write(f"🍛 **{discount_value}% OFF** on your preferred {preferred_cuisine} dishes!")

            # 🏷️ Display Discounted Prices
            st.write("\n#### 🏷️ Discounted Prices:")
            for dish, (original, discounted) in discounted_prices.items():
                st.write(f"✅ {dish.title()}: ₹{discounted} (Original: ₹{original})")
        else:
            st.write(f"❌ No discounted dishes available for your preferred cuisine: {preferred_cuisine}.")

        # 📩 Notify User
        st.write("\n📩 **Check your email for exclusive coupons! 🎁**")

    else:
        st.warning("⚠️ Please enter your name, email, and Customer ID to proceed!")










