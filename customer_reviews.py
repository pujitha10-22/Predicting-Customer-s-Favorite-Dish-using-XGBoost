import streamlit as st
import pandas as pd
from datetime import datetime

# Define the storage file
storage_file = "customer_reviews_data.xlsx"

# Load existing data or create a new one
def load_data():
    try:
        return pd.read_excel(storage_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Customer ID", "Review", "Date", "Rating", "Currently Staying"])

# Save the new review
def save_review(customer_id, review, rating, staying):
    df = load_data()
    new_data = pd.DataFrame(
        [[customer_id, review, datetime.now().strftime("%Y-%m-%d"), rating, staying]],
        columns=["Customer ID", "Review", "Date", "Rating", "Currently Staying"]
    )
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_excel(storage_file, index=False)

# Streamlit UI
st.title("Customer Reviews Submission")

# User Input Fields
customer_id = st.text_input("Enter Customer ID:", placeholder="Enter your unique customer ID")

review_text = st.text_area("Enter your review:", placeholder="Write your experience...")

rating = st.selectbox("Select Rating (1-10 Stars)", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

staying = st.radio("Are you currently staying in the hotel?", ["Yes", "No"])

# Submit Button
if st.button("Submit Review"):
    if not customer_id:
        st.error("❌ Please enter your Customer ID.")
    elif not review_text.strip():
        st.error("❌ Please enter a review before submitting.")
    else:
        save_review(customer_id, review_text, rating, staying)
        st.success("✅ Your review has been submitted successfully!")

# Display Recent Reviews
st.subheader(" Updated Customer Reviews")
df = load_data()

if not df.empty:
    st.dataframe(df.tail(10))  # Show last 10 reviews
else:
    st.write("No reviews submitted yet.")
