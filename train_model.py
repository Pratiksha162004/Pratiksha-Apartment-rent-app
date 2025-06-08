# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    'area': [500, 1000, 1500, 2000],
    'bedrooms': [1, 2, 3, 4],
    'bathrooms': [1, 2, 2, 3],
    'rent': [1500, 2500, 3500, 4500]
}
df = pd.DataFrame(data)

# Features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['rent']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('rent_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as rent_model.pkl")
