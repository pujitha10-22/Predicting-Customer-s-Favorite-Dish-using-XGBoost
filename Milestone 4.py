HOTEL SENTIMENT ANALYSIS
import streamlit as st
import pandas as pd
import together

# 🔑 Set API Key for Together AI
TOGETHER_AI_KEY = "4ca156fb9884e7153537ed8e6f47e8fce6f7e0032d56806618e635edac065961"  # Replace with actual API key

# Initialize Together AI
together.api_key = TOGETHER_AI_KEY

# Load Excel file
file_path = "reviews_data.xlsx"  # Ensure the file is correctly placed
df = pd.read_excel(file_path)

# Convert column names to lowercase for consistency
df.columns = df.columns.str.lower().str.strip()

# Rename columns if needed
column_mapping = {"review_date": "date"}
df.rename(columns=column_mapping, inplace=True)

# Check required columns
required_columns = {"review", "date", "rating"}
if not required_columns.issubset(df.columns):
    st.error(f"❌ Missing required columns! Found: {list(df.columns)}. Expected: Review, Date, and Rating.")
    st.stop()

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# ✅ User Inputs
st.title("📊 Hotel Sentiment Analysis")
st.write("Analyze customer reviews based on search query, rating, and sentiment.")

query = st.text_input("🔎 Enter a query about customer review (Required):", "")

if not query:
    st.warning("⚠️ Please enter a query before analyzing sentiment.")

start_date = st.date_input("📅 Start Date:", df["date"].min())
end_date = st.date_input("📅 End Date:", df["date"].max())

rating_range = st.slider("⭐ Select Rating Range (1-10):", 1.0, 10.0, (1.0, 10.0))
 # ✅ Apply Filters
filtered_df = df[
    (df["date"] >= pd.Timestamp(start_date)) &
    (df["date"] <= pd.Timestamp(end_date)) &
    (df["rating"].between(*rating_range))
]

# ✅ Apply Query Filter (Ensure Partial Match & Case-Insensitive)
import re  # Import regex module

# Ensure exact word match
query_pattern = r"\b" + re.escape(query) + r"\b"
filtered_df = filtered_df[filtered_df["review"].str.contains(query_pattern, case=False, na=False, regex=True)]


# Debugging Statements (Optional)
st.write(f"🔎 Debug: Total Reviews After Date & Rating Filter: {filtered_df.shape[0]}")
if query:
    st.write(f"🔎 Debug: Total Reviews After Query Filter: {filtered_df.shape[0]}")


# ✅ Sentiment Analysis Function
def get_sentiment(review, rating):
    try:
        response = together.Embeddings.create(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            input=[review]
        )
        embedding = response["data"][0]["embedding"]
        sentiment_score = sum(embedding) / len(embedding)

        # Adjust sentiment based on rating
        if rating >= 7:
            return "Positive 😊" if sentiment_score > -0.1 else "Neutral 😐"
        elif rating <= 4:
            return "Negative 😡"
        else:
            if sentiment_score > 0.02:
                return "Positive 😊"
            elif sentiment_score < -0.02:
                return "Negative 😡"
            else:
                return "Neutral 😐"
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Analyze Sentiment (Only if Query is Provided)
if st.button("🚀 Analyze Sentiment"):
    if not query:
        st.error("❌ Please enter a query before analyzing sentiment.")
    elif filtered_df.empty:
        st.write("❌ No matching reviews found based on the entered query and filters.")
    else:
        st.subheader("📄 Sentiment Summary")
        for _, row in filtered_df.iterrows():
            sentiment = get_sentiment(row["review"], row["rating"])
            st.write(f"📅 **Date:** {row['date'].date()}  |  ⭐ **Rating:** {row['rating']}")
            st.write(f"💬 **Review:** {row['review']}")
            st.write(f"🟢 **Sentiment:** {sentiment}")
            st.write("---")


CUSTOMER REVIEW 
       

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







