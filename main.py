import os
from data_loader import load_sample_data
from eda import run_eda
from model import train_model

os.makedirs('plots', exist_ok=True)

print("Loading data...")
df = load_sample_data()

print("\n--- EDA ---")
run_eda(df)

print("\n--- Model Training ---")
model, scaler = train_model(df)

print("\nDone! Check /plots folder for all charts.")