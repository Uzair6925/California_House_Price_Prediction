import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field

app = FastAPI()

##Loading model
model = joblib.load("house_model.joblib")
features =  joblib.load("house_features.joblib")

#Input schema 
class houseFeatures (BaseModel):
    MedInc:float = Field(gt=0,description = "Media Income of Neighbourhood")    
    HouseAge:float = Field(gt=0,description="Average age of house in the block")
    AveRooms: float = Field(
        gt=0,description="Average number of rooms per household"
    )
    AveBedrms: float = Field(
        gt=0,description="Average number of bedrooms per household"
    )
    Population: float = Field(
        gt=0,description="Total population living in the census block group"
    )
    AveOccup: float = Field(
        gt=0, description="Average number of people per household"
    )
    Latitude: float = Field(
        ge=-90, le=90, description="Geographical latitude of the neighbourhood in degrees"
    )
    Longitude: float = Field(
        ge=-180,le=180, description="Geographical longitude of the neighbourhood in degrees"
    )
    

@app.get("/")
def home():
    return {
        "Message":"California house prediction api",
        "status":"running",
        "endpoint":"send post request to /predict"
    }
    
@app.get("/health")
def health():
    return {
            "status":"running",
            "model":"RandomForestRegressor",
            "features":features,
            "avg error":"$39,000"
    }
    
#prediction
@app.post("/predict")
def predict(house:houseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc":house.MedInc,"HouseAge":house.HouseAge,"AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,"Population":house.Population,
            "AveOccup":house.AveOccup,"Latitude":house.Latitude,"Longitude":house.Longitude
        }])
        
        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000
        
        return{
            "predicted_price":f"${price_usd:,.0f}",
            "predicted_price_short":f"${predicted:,.2f} houndred thousands",
            "fidence range":f"${price_usd - 39000:,.0f} to $ {price_usd + 39000:,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail= f"Prediction failed: {str(e)}"
        )
 
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    
    # Check file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload CSV file only."
        )
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    # Check missing columns
    missing_col = [
        col for col in required_columns
        if col not in df.columns
    ]
    if missing_col:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_col}"
        )
    # Check empty dataset
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file has no rows."
        )

    # Convert required columns to numeric
    numeric_df = df[required_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )
    # Replace infinite values with NaN
    numeric_df = numeric_df.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Features that must be greater than zero
    positive_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup"
    ]

    # Collect all validation errors
    errors = []
    for i in range(len(numeric_df)):
        row = numeric_df.iloc[i]
        row_num = i + 2
        for col in required_columns:
            value = row[col]

            # Check missing or non-numeric values
            if pd.isna(value):
                errors.append(
                    f"Row {row_num}: {col} is required and must be numeric."
                )

            # Check positive features
            elif col in positive_features and value <= 0:
                errors.append(
                    f"Row {row_num}: {col} must be greater than 0."
                )

            # Check Latitude
            elif col == "Latitude" and not (-90 <= value <= 90):
                errors.append(
                    f"Row {row_num}: Latitude must be between -90 and 90."
                )

            # Check Longitude
            elif col == "Longitude" and not (-180 <= value <= 180):
                errors.append(
                    f"Row {row_num}: Longitude must be between -180 and 180."
                )

    # Return validation errors after checking all rows
    if errors:
        displayed_errors = errors[:20]
        if len(errors) > 20:
            displayed_errors.append(
                f"And {len(errors) - 20} more validation errors."
            )
        raise HTTPException(
            status_code=400,
            detail="\n\n".join(displayed_errors)
        )
        
    try:
        predictions = model.predict(numeric_df)
        
        df["predicted_columns_usd"] = predictions * 100000
        df["predicted_columns_usd"] = df["predicted_columns_usd"].apply(lambda x:f"${x:,.0f}") 
        
        output = df.to_csv(index=False)
        
        return StreamingResponse(
            io.StringIO(output),
            media_type= "text/csv",
            headers={
                "content-Disposition":"attachment; filename= predictions.csv"
            }
        )         
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )