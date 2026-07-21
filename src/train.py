import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    print("Loading dataset...")
    housing = fetch_california_housing()
    
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training RandomForestRegressor...")
    # Reduce size to stay under GitHub's 100MB limit
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    # Safest way across scikit-learn versions
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Metrics on Test Set:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    print("Saving model and scaler...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("Done!")

if __name__ == "__main__":
    main()
