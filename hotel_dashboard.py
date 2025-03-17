import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# ✅ Load Datasets
try:
    hotel_df = pd.read_excel("hotel_bookings.xlsx")
    dining_df = pd.read_excel("dining_info.xlsx")
    reviews_df = pd.read_excel("reviews_data.xlsx")
    print("✅ Files loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ File NOT found! Check your filenames: {e}")
    exit()

# ✅ Debugging: Print available columns
print("Hotel Data Columns:", hotel_df.columns)
print("Dining Data Columns:", dining_df.columns)
print("Reviews Data Columns:", reviews_df.columns)

# ✅ Ensure Date Columns are in Correct Format
for col in ["check_in_date", "check_out_date"]:
    if col in hotel_df.columns:
        hotel_df[col] = pd.to_datetime(hotel_df[col], dayfirst=True, errors="coerce")

if "review_date" in reviews_df.columns:
    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"], dayfirst=True, errors="coerce")

if "check_in_date" in dining_df.columns:
    dining_df["check_in_date"] = pd.to_datetime(dining_df["check_in_date"], dayfirst=True, errors="coerce")

# ✅ Initialize Dash App
app = dash.Dash(__name__)

# 📈 **Dashboard 1: Hotel Booking Insights**
fig1 = px.histogram(hotel_df, x="check_in_date", title="📊 Bookings Trend Over Time")

# Fixing the column name for cuisine analysis
fig2 = px.pie(hotel_df, names="Preferred Cusine", title="🍽️ Preferred Cuisine Analysis")

# Fixing resampling method for monthly aggregation
hotel_df_monthly = hotel_df.resample('M', on='check_in_date').size().reset_index(name='count')
fig3 = px.bar(hotel_df_monthly, x="check_in_date", y="count", title="📅 Average Length of Stay (Monthly)")

# 🍽️ **Dashboard 2: Dining Insights**
if "price_for_1" in dining_df.columns and "Preferred Cusine" in dining_df.columns:
    fig4 = px.pie(dining_df, values="price_for_1", names="Preferred Cusine", title="🍕 Average Dining Cost by Cuisine")
else:
    fig4 = None

if "customer_id" in dining_df.columns and "check_in_date" in dining_df.columns:
    fig5 = px.line(dining_df, x="check_in_date", y="customer_id", title="👥 Customer Count Over Time")
else:
    fig5 = None

# 📝 **Dashboard 3: Reviews Analysis**
# ✅ Categorize Review Sentiments
def categorize_sentiment(review):
    review = str(review).lower()  # Convert to lowercase for consistency
    if any(word in review for word in ["good", "excellent", "great", "amazing", "positive", "love"]):
        return "Positive"
    elif any(word in review for word in ["bad", "poor", "terrible", "negative", "worst"]):
        return "Negative"
    else:
        return "Neutral"

# ✅ Apply function to categorize reviews
reviews_df["Sentiment"] = reviews_df["Review"].apply(categorize_sentiment)

# ✅ Debugging: Print unique sentiment categories
print("Unique Sentiments:", reviews_df["Sentiment"].unique())

# ✅ Create Sentiment Analysis Pie Chart
fig6 = px.pie(reviews_df, names="Sentiment", title="📝 Sentiment Analysis", hole=0.3)


if "Rating" in reviews_df.columns:
    fig7 = px.histogram(reviews_df, x="Rating", title="⭐ Rating Distribution")
else:
    fig7 = None

# ✅ Layout with Tabs
app.layout = html.Div([
    html.H1("🏨 Hotel & Dining Dashboard"),
    
    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="🏨 Hotel Insights", value="tab1"),
        dcc.Tab(label="🍽️ Dining Insights", value="tab2"),
        dcc.Tab(label="📝 Reviews Analysis", value="tab3"),
    ]),
    
    html.Div(id="tab-content")
])

# ✅ Callback to Update Tabs
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value")]
)
def update_tab(selected_tab):
    if selected_tab == "tab1":
        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3)
        ])
    elif selected_tab == "tab2":
        return html.Div([
            dcc.Graph(figure=fig4) if fig4 else html.Div("⚠️ No dining cost data available"),
            dcc.Graph(figure=fig5) if fig5 else html.Div("⚠️ No customer count data available")
        ])
    elif selected_tab == "tab3":
        return html.Div([
            dcc.Graph(figure=fig6) if fig6 else html.Div("⚠️ No sentiment analysis data available"),
            dcc.Graph(figure=fig7) if fig7 else html.Div("⚠️ No rating distribution data available")
        ])
    return html.Div("📊 Select a tab to view data")

# ✅ Run the Dash App
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

