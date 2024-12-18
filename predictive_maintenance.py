import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline

# Create simulated machine data
def generate_machine_data(start_date, end_date):
    date_rng = pd.date_range(start=start_date, end=end_date, freq='H')
    data = pd.DataFrame(date_rng, columns=['timestamp'])
    data['temperature'] = np.random.normal(75, 5, size=(len(date_rng)))
    data['vibration'] = np.random.normal(1.5, 0.2, size=(len(date_rng)))
    data['operational_hours'] = data.index
    return data

# Create simulated maintenance logs
def generate_maintenance_logs():
    logs = [
        {
            'date': '2023-09-15',
            'issue': 'Overheating',
            'action': 'Reduced load, increased lubrication'
        },
        {
            'date': '2023-10-01',
            'issue': 'Vibration anomaly',
            'action': 'Replaced bearing'
        }
    ]
    return pd.DataFrame(logs)

machine_data = generate_machine_data(start_date='2023-09-01', end_date='2023-10-31')
maintenance_logs = generate_maintenance_logs()

# Preparing data for modeling
def prepare_data(machine_data):
    machine_data['days_to_failure'] = np.random.randint(0, 10, size=len(machine_data))  # Simulated target
    features = ['temperature', 'vibration', 'operational_hours']
    X = machine_data[features][:-10]  # leaving last 10 points for testing
    y = machine_data['days_to_failure'][:-10]
    return X, y

X, y = prepare_data(machine_data)

# Train a simple RandomForest model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# Predict the next maintenance date
def predict_maintenance(data, model):
    X_test = data[['temperature', 'vibration', 'operational_hours']][-10:]
    predictions = model.predict(X_test)
    return predictions

predictions = predict_maintenance(machine_data, model)

# Loading a pre-trained model for text generation (hypothetical)
def generate_maintenance_instructions(issue, action):
    instructions_pipeline = pipeline('text-generation', model='gpt-2')
    input_text = f"Issue: {issue}. Action taken: {action}. Detailed maintenance instructions:"
    generated_text = instructions_pipeline(input_text, max_length=50, num_return_sequences=1)
    return generated_text[0]['generated_text']

# Example usage
log_example = maintenance_logs.iloc[0]
detailed_instructions = generate_maintenance_instructions(log_example['issue'], log_example['action'])
print(detailed_instructions)
