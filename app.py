import joblib
from sklearn.ensemble import RandomForestRegressor

model = joblib.load('aqi_model.pkl')

if isinstance(model, RandomForestRegressor):
    print("Model is a RandomForestRegressor")
else:
    print("Model is not a RandomForestRegressor")
