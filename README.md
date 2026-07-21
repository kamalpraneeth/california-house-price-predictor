# Deployed ML Prediction Service

A complete Machine Learning portfolio project featuring data preprocessing, model training, a FastAPI backend, and a web UI. The service predicts California house prices based on various neighborhood features.

## Problem Statement
The goal is to build a regression model to estimate the median house value for California districts, given features like median income, house age, average rooms, and geographical location. This demonstrates an end-to-end ML workflow from data to a deployed REST API.

## Dataset
- **Source**: [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (built-in via `scikit-learn`)
- **Features**: 8 numerical features including `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, and `Longitude`.
- **Target**: Median house value (in units of 100,000).

## Model Performance
A `RandomForestRegressor` was trained and evaluated on a 20% test split:
- **RMSE**: 0.5449
- **R²**: 0.7734

## Running Locally

1. **Install Dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the Model** (optional, model is already saved):
   ```bash
   python src/train.py
   ```

3. **Start the API Server**:
   ```bash
   uvicorn src.app:app --reload
   ```
   Navigate to `http://localhost:8000` to view the demo UI.

## Testing
Run the API tests using `pytest`:
```bash
pytest tests/
```

## Running with Docker
```bash
docker build -t ml-prediction-service .
docker run -p 8000:8000 ml-prediction-service
```
Navigate to `http://localhost:8000` to test the service.

## Deployment to Render
This project is structured to be easily deployed on [Render](https://render.com).
1. Create a new **Web Service** on Render and connect your GitHub repository.
2. Select **Docker** as the Environment.
3. Set the build context to the repository root. Render will automatically build the Docker image using the provided `Dockerfile`.
4. Expose port `8000` or let Render automatically detect it from the Dockerfile.

## Example API Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "MedInc": 8.3252,
           "HouseAge": 41.0,
           "AveRooms": 6.9841,
           "AveBedrms": 1.0238,
           "Population": 322.0,
           "AveOccup": 2.5555,
           "Latitude": 37.88,
           "Longitude": -122.23
         }'
```
**Response**:
```json
{"prediction": 4.1352}
```
