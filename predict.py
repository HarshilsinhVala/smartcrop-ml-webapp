import sys
import pickle
import numpy as np
import os

model_file = "crop_model.pkl"
encoder_file = "label_encoder.pkl"

if not os.path.exists(model_file) or not os.path.exists(encoder_file):
    print("🚨 Error: Model or encoder file is missing! Ensure 'crop_model.pkl' and 'label_encoder.pkl' exist.")
    sys.exit(1)

with open(model_file, "rb") as f:
    model = pickle.load(f)

with open(encoder_file, "rb") as f:
    label_encoder = pickle.load(f)

if len(sys.argv) != 8:  
    print("🚨 Error: Incorrect number of arguments!")
    print("Usage: python predict.py <N> <P> <K> <temperature> <humidity> <ph> <rainfall>")
    sys.exit(1)

try:
    N = float(sys.argv[1])
    P = float(sys.argv[2])
    K = float(sys.argv[3])
    temperature = float(sys.argv[4])
    humidity = float(sys.argv[5])
    ph = float(sys.argv[6])
    rainfall = float(sys.argv[7])

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)[0]
    crop_name = label_encoder.inverse_transform([prediction])[0]

    print(crop_name)

except ValueError:
    print("🚨 Error: Invalid input! Please enter numerical values for all parameters.")
    sys.exit(1)
