# California House Price Prediction
A machine learning web application for predicting median housing values in California using a Random Forest Regressor, with a FastAPI backend and Streamlit frontend.
## Project Overview
This project provides an interactive interface for estimating California housing values from socioeconomic, housing, population, and geographical features.
The application supports both individual predictions and batch predictions through CSV uploads.

## Key Features
- Single housing value prediction using eight input features
- Batch predictions using CSV files
- Input validation and field-specific error messages
- CSV validation for missing columns, invalid values, and out-of-range coordinates
- Detailed row-level CSV validation errors
- Batch prediction analytics, including average, minimum, and maximum predicted values
- Housing value visualization
- Downloadable prediction results

## Machine Learning Model
**Algorithm:** RandomForestRegressor
**Dataset:** California Housing Dataset from Scikit-learn
**Reported Mean Absolute Error:** Approximately $39,000

### Input Features
| Feature | Description |
|---|---|
| MedInc | Median income in the block group |
| HouseAge | Median housing age |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block group population |
| AveOccup | Average occupants per household |
| Latitude | Geographical latitude |
| Longitude | Geographical longitude |

## Technology Stack
- Python
- Scikit-learn
- Pandas
- FastAPI
- Pydantic
- Streamlit
- Joblib
- Render
- Streamlit Community Cloud
- GitHub

## API Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | API and model information |
| `/predict` | POST | Predict a single housing value |
| `/predict-file` | POST | Generate predictions from a CSV file |
| `/docs` | GET | Interactive Swagger API documentation |

## Live Deployment
**Web Application:** https://californiahousepriceprediction-d8imbmbsfri4z2xncqnhhm.streamlit.app/.
**FastAPI Documentation:** https://california-house-price-prediction-6pre.onrender.com/docs
## Local Installation
Clone the repository and install dependencies using `uv`.
Start the FastAPI backend using:
`uvicorn main:app --reload`
Start the Streamlit frontend in a separate terminal using:
`streamlit run streamlit_app.py`
For local development, the frontend connects to `http://127.0.0.1:8000` unless an `API_URL` environment variable is configured.

## Limitations
The California Housing dataset represents historical median property values for census block groups, rather than individual property sale prices.
Predictions are estimates and should not be treated as current real-estate valuations.

The reported mean absolute error describes average prediction error on the evaluation data; it is not a statistically calibrated confidence interval.
