{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e50ccaba-aa3f-440b-a19f-44775b7e2684",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pymongo import MongoClient\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n",
    "import xgboost as xgb\n",
    "from sklearn.metrics import accuracy_score, log_loss\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "59682208-5d34-47eb-8c55-34543b630388",
   "metadata": {},
   "outputs": [],
   "source": [
    "# MongoDB Connection\n",
    "client = MongoClient(\"mongodb+srv://pujithamodem55:ggXrpjf3z4sWV9sy@cluster0.tzmzm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0\")\n",
    "db = client[\"hotel_guests\"]\n",
    "collection = db[\"dining_info\"]\n",
    "df_from_mongo = pd.DataFrame(list(collection.find()))\n",
    "df = df_from_mongo.copy()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9148ba34-e752-4567-b303-ff4e3a288722",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data preprocessing\n",
    "df['check_in_date'] = pd.to_datetime(df['check_in_date'])\n",
    "df['check_out_date'] = pd.to_datetime(df['check_out_date'])\n",
    "df['order_time'] = pd.to_datetime(df['order_time'])\n",
    "df['check_in_day'] = df['check_in_date'].dt.dayofweek\n",
    "df['check_out_day'] = df['check_out_date'].dt.dayofweek\n",
    "df['check_in_month'] = df['check_in_date'].dt.month\n",
    "df['check_out_month'] = df['check_out_date'].dt.month\n",
    "df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7bf5ebdb-ac00-4ef4-a029-3961bba16b73",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Split data into features, train and test sets\n",
    "features_df = df[df['order_time'] < '2024-01-01']\n",
    "train_df = df[(df['order_time'] >= '2024-01-01') & (df['order_time'] <= '2024-10-01')]\n",
    "test_df = df[df['order_time'] > '2024-10-01']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "06c4567b-03d6-4ea2-9d3c-6be6fd19f7a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Engineering\n",
    "# Customer level features\n",
    "customer_features = features_df.groupby('customer_id').agg(\n",
    "    avg_spend=('price_for_1', 'mean'),\n",
    "    avg_stay_duration=('stay_duration', 'mean'),\n",
    "    avg_qty=('Qty', 'mean')\n",
    ").reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "407338ab-1e26-4dff-ac2e-827a0fc15c16",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Customer cuisine preferences\n",
    "customer_fav_cuisine = features_df.groupby('customer_id')['Preferred Cusine'].agg(lambda x: x.mode()[0]).reset_index()\n",
    "customer_frequent_cuisine = features_df.groupby('customer_id')['Preferred Cusine'].agg(lambda x: x.value_counts().idxmax()).reset_index()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "71b8cec9-6f3c-4efe-b391-ed45b995c2ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Customer dish preferences\n",
    "customer_fav_dish = features_df.groupby('customer_id')['dish'].agg(lambda x: x.mode()[0]).reset_index()\n",
    "customer_fav_dish.rename(columns={'dish': 'customer_fav_dish'}, inplace=True)\n",
    "customer_frequent_dish = features_df.groupby('customer_id')['dish'].agg(lambda x: x.value_counts().idxmax()).reset_index()\n",
    "customer_frequent_dish.rename(columns={'dish': 'customer_frequent_dish'}, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "09c4c1f0-ce2c-4859-91ad-1eb058cb2043",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cuisine level features\n",
    "cuisine_features = features_df.groupby('Preferred Cusine').agg(\n",
    "    total_orders_per_cuisine=('transaction_id', 'count'),\n",
    "    avg_spend_per_cuisine=('price_for_1', 'mean'),\n",
    "    avg_qty_per_cuisine=('Qty', 'mean'),\n",
    "    avg_stay_duration_per_cuisine=('stay_duration', 'mean')\n",
    ").reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "30cb897e-8580-4ac7-a9dc-d5fc08a8645d",
   "metadata": {},
   "outputs": [],
   "source": [
    "cuisine_popular_dish = features_df.groupby('Preferred Cusine')['dish'].agg(lambda x: x.mode()[0]).reset_index()\n",
    "cuisine_popular_dish.rename(columns={'dish': 'cuisine_popular_dish'}, inplace=True)\n",
    "cuisine_frequent_dish = features_df.groupby('Preferred Cusine')['dish'].agg(lambda x: x.value_counts().idxmax()).reset_index()\n",
    "cuisine_frequent_dish.rename(columns={'dish': 'cuisine_frequent_dish'}, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6f88107c-5bc3-4f93-82a4-d7c3979586c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merge features with train data\n",
    "def merge_features(df):\n",
    "    df = df.merge(customer_features, on='customer_id', how='left')\n",
    "    df = df.merge(customer_fav_cuisine, on='customer_id', how='left')\n",
    "    df = df.merge(customer_frequent_cuisine, on='customer_id', how='left')\n",
    "    df = df.merge(customer_fav_dish, on='customer_id', how='left')\n",
    "    df = df.merge(customer_frequent_dish, on='customer_id', how='left')\n",
    "    \n",
    "    df = df.merge(cuisine_features, on='Preferred Cusine', how='left')\n",
    "    df = df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')\n",
    "    df = df.merge(cuisine_frequent_dish, on='Preferred Cusine', how='left')\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7e7c18a6-f370-493d-bb57-0de2443028f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "train_df = merge_features(train_df)\n",
    "test_df = merge_features(test_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5c76951c-18e8-4f3b-9e61-63a90fa5e672",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop unnecessary columns\n",
    "columns_to_drop = ['_id', 'transaction_id', 'customer_id', 'price_for_1',\n",
    "                  'Qty', 'order_time', 'check_in_date', 'check_out_date']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3e5746ad-0312-450e-9fa5-168bd84fed74",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df.drop(columns_to_drop, axis=1, inplace=True)\n",
    "test_df.drop(columns_to_drop, axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b368f58e-d8bc-4183-b251-765aa24c9f8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert object columns to category\n",
    "for df in [train_df, test_df]:\n",
    "    object_columns = df.select_dtypes(include=['object']).columns\n",
    "    for col in object_columns:\n",
    "        df[col] = df[col].astype('category')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "2836fef2-d91b-4f7e-95bd-9ada7a8bcaea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# One-hot encoding\n",
    "categorical_cols = ['Preferred Cusine', 'customer_fav_dish', 'cuisine_popular_dish']\n",
    "encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "2bd79f30-be70-4041-b839-434c77ef0f46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transform train data\n",
    "encoded_train = encoder.fit_transform(train_df[categorical_cols])\n",
    "encoded_train_df = pd.DataFrame(\n",
    "    encoded_train, \n",
    "    columns=encoder.get_feature_names_out(categorical_cols)\n",
    ")\n",
    "train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_train_df], axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "dddb782e-c203-4d53-bf15-5f3dbc4b1e0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transform test data\n",
    "encoded_test = encoder.transform(test_df[categorical_cols])\n",
    "encoded_test_df = pd.DataFrame(\n",
    "    encoded_test, \n",
    "    columns=encoder.get_feature_names_out(categorical_cols)\n",
    ")\n",
    "test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a74a1d42-c93d-412f-8d39-a3617afb3ab6",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df = train_df.dropna(subset=['dish'])\n",
    "test_df = test_df.dropna(subset=['dish'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fa4af581-776d-4630-8161-1cfc8e3ec99c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare target variable\n",
    "label_encoder = LabelEncoder()\n",
    "train_df['dish'] = label_encoder.fit_transform(train_df['dish'])\n",
    "test_df['dish'] = label_encoder.transform(test_df['dish'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "972fc126-3036-4df8-89e4-732dbb6be50a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split features and target\n",
    "X_train = train_df.drop(columns=['dish'])\n",
    "y_train = train_df['dish']\n",
    "X_test = test_df.drop(columns=['dish'])\n",
    "y_test = test_df['dish']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d8a66d83-da77-4ede-8452-129e67091059",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train XGBoost model\n",
    "xgb_model = xgb.XGBClassifier(\n",
    "    objective=\"multi:softmax\",\n",
    "    eval_metric=\"mlogloss\",\n",
    "    learning_rate=0.1,\n",
    "    max_depth=1,\n",
    "    n_estimators=100,\n",
    "    subsample=1,\n",
    "    colsample_bytree=1,\n",
    "    random_state=42,\n",
    "    enable_categorical=True\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "295dc4e6-7a11-4929-a78f-1546d6456641",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit and evaluate model\n",
    "xgb_model.fit(X_train, y_train)\n",
    "y_pred = xgb_model.predict(X_test)\n",
    "y_pred_prob = xgb_model.predict_proba(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "8315f247-f7d4-4c63-b6ec-c553db29770a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Model Performance:\n",
      "Accuracy: 0.1770\n",
      "Log loss: 2.4865\n"
     ]
    }
   ],
   "source": [
    "# Print metrics\n",
    "print(\"\\nModel Performance:\")\n",
    "print(f\"Accuracy: {accuracy_score(y_test, y_pred):.4f}\")\n",
    "print(f\"Log loss: {log_loss(y_test, y_pred_prob):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "98f11c25-99cf-414d-a7b4-874804ddebde",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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T/U80dHROjc/DgkJ4cMPP2TQoEE0aNCA5ORkJk+enOslDkKIf1+xjeCZmZkxduxYQkJCMDExoWnTpty5c4fz588TGBiIq6sroaGhzJgxg8uXL7NgwQKd44cPH06HDh3w9PTk3r177Nmzh6pVqwJPpjE1Gg0///wzfn5+mJubY2VlxZAhQxgzZoz63+3cuXNJSUl54U09n8fa2prRo0czYsQItFotzZo148GDBxw8eBAbGxv1+sKi4uPjQ+PGjenSpQtz587F09OT27dvs3XrVgICAqhfvz4fffQRQUFB1K9fn6ZNm7J27VrOnz+vM+3cunVrRo4cydatW6lYsSKffvqpzipmT09P+vTpQ2BgIAsWLKBOnTrcuXOHXbt2UbNmTTp2fPFtGwDc3NzYsWMH0dHR2NvbY2tr+9z/THr37s2ECRMYOHAg48eP58aNG8yfP/+553jea6OwHR3fJtv0+KsiPT2dyMhIzoX6vrL/Kf/XY/D29n7hCHm/fv3o169fgY8HstXRaDRMnTpVvb42KwYhRMlRrKtoJ02ahJGREZMnT+b27ds4OzszePBgjI2N+d///seQIUOoWbMmDRo0YPr06XTv3l09NjMzk6FDh/LXX39hY2ND+/btWbhwIQCvv/46U6ZMYdy4cbz33nsEBgayatUqZs+ejVar5d133+Xhw4fUr1+fHTt2PHdFa15MmzYNBwcHZs2axR9//IGdnR1169bl448/fql280Kj0RAZGcmECRN47733uHPnDmXKlKFFixbqbQx69uzJtWvXCAkJ4fHjx3Tt2pUhQ4awY8cOtZ1+/fpx5swZAgMDMTIyYsSIEertZrKEh4czffp0Ro0axa1bt3jttdd444036NSpU577O3DgQPbu3Uv9+vVJSkpiz549eHt751rfysqKn376icGDB1OnTh2qVavGnDlz6Nq1a67HPO+1IYQQQvwXaJR/+wIpUSKEhoayefPmHKe+xYslJiZia2tLQkLCKz+C5+fn98qPfkkMxUtiKBkkhpKhKGLIes/JuhF5Xshn0QohhBBC6BlJ8IpJ9erVsbKyyvFr7dq1xd29f83MmTNzfRye9xmbQgghhMidfBZtMYmMjMz1ztTPfgRQUQgNDSU0NLTIz/MigwcPpkePHjnue96tdIQQQgiRO0nwikn58uWLuwslQunSpV/4cWVCCCGEyB+Z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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot feature importance\n",
    "xgb.plot_importance(xgb_model, max_num_features=10)\n",
    "plt.title(\"Top 10 Most Important Features\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4d44a35-3a87-4fff-8757-db6bc53fe454",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "631f3183-cd68-4f1c-931d-a3f4d37da21e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f36dabc0-ec0d-422e-b99d-9051617c0ca6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3d74968-68b8-45fe-983a-206e2edb7647",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83bcb3f4-7a90-4469-9903-f15828dec0b5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75844e76-171e-435e-982c-f8d81e3b1e35",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f21d32b-70d1-48d1-bd6b-23d2c9b86bff",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4367dc33-1bc0-40ef-b8a3-cd65c0855559",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
