import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import joblib

df = pd.read_csv("Data.csv")

df_cleaned = df.dropna()

years = ['2020-21', '2021-22', '2022-23', '2023-24']
records = []

for i, row in df_cleaned.iterrows():
    for j in range(len(years) - 1):
        year = years[j]
        next_year = years[j + 1]
        record = {
            'Soil_Moisture': row[f'Soil_Moisture-{year}'],
            'Nitrogen': row[f'Nitrogen-{year}'],
            'Phosphorus': row[f'Phosphorus-{year}'],
            'Potassium': row[f'Potassium-{year}'],
            'Soil_pH': row[f'Soil_pH-{year}'],
            'Temperature': row[f'Temperature-{year}'],
            'Rainfall': row[f'Rainfall-{year}'],
            'Humidity': row[f'Humidity-{year}'],
            'Area': row[f'Area-{year}'],
            'Prev_Yield': row[f'Yield-{year}'],
            'Next_Yield': row[f'Yield-{next_year}']
        }
        records.append(record)

df_model = pd.DataFrame(records)

features = ['Soil_Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'Soil_pH',
            'Temperature', 'Rainfall', 'Humidity', 'Area', 'Prev_Yield']
target = 'Next_Yield'

X = df_model[features]
y = df_model[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),   # New hidden layer 1
    Dense(64, activation='relu'),    # New hidden layer 2
    Dense(32, activation='relu'),    # New hidden layer 3
    Dense(1)                         # Output layer
])

model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
model.fit(X_train, y_train, epochs=1000, batch_size=8, validation_split=0.1, verbose=1)

model.save("model.h5")
joblib.dump(scaler, "scaler.save")
