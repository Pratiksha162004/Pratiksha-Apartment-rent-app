# train_model.py

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load CSV with proper delimiter and encoding
df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\ML CAS\apartments_for_rent_classified_10K.csv",
    encoding='latin1',
    delimiter=';',
    low_memory=False
)

# Step 2: Print columns for verification (optional, can be commented after first run)
print("Columns in CSV file:")
print(df.columns)

# Step 3: Select required columns
df = df[['square_feet', 'bedrooms', 'bathrooms', 'price']]

# Step 4: Convert columns to numeric and drop invalid rows
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Drop rows where any of these columns could not be converted (i.e. NaN values)
df = df.dropna(subset=['square_feet', 'bedrooms', 'bathrooms', 'price'])

# Step 5: Rename columns for consistency
df.rename(columns={'square_feet': 'area', 'price': 'rent'}, inplace=True)

# Optional: print data info for verification
print("\nCleaned data sample:")
print(df.head())

# Step 6: Prepare X and y for model
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['rent']

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# Step 10: Save the trained model
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
