import streamlit as st 
from fastapi import requests
import requests
import pandas as pd
import io
import os

#Page configuration
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
# st.write("API URL:", API_URL)
#Header
st.title("🏠 California House Price Predictor")
st.write("Predict California Hoousing Values in seconds with the power of AI")

#Tabs
tab1, tab2 = st.tabs([
    "🏠 Single Prediction",
    "📂 Batch Prediction"
])

#Single prediction tab
with tab1:
    st.subheader("Enter House Features")
    
    with st.form("House_Form"):
        col1, col2 = st.columns(2)
        with col1:
            med_inc = st.number_input(
                "Median Income",
                min_value=0.01,
                value=3.5
            )
            house_age = st.number_input(
                "House Age",
                min_value=1.0,
                max_value=52.0,
                value = 1.0,
                step=1.0
            )
            ave_rooms = st.number_input(
                "Average Rooms",
                min_value=0.01,
                value=5.5
            )
            av_bedrms = st.number_input(
                "Aevrage Bedrooms",
                min_value=0.01,
                value=1.1
            )
        with col2:
            popu= st.number_input(
                "Population",
                min_value=0.01,
                value= 800.0
            )
            ave_occup = st.number_input(
                "Average Occupacy",
                min_value=0.01, 
                value=3.0
            )
            lat = st.number_input(
                "Latitude",
                step=0.01,
                value=37.33,
                help="Allowed range: -90 to 90"
            )
            long = st.number_input(
                "Longitude",
                step=0.01,
                value=-121.89,
                help="Enter longitude between -180 and 180"
            )
            
        sub = st.form_submit_button("Submit data")
    
    if sub:
        if long < -180 or long > 180:
             st.error("Invalid Longitude! Enter a value between -180 and 180.")
        if lat < -90 or lat > 90:
                st.error("Invalid Latitude! Enter a value between -90 and 90.")
        payload = {
            "MedInc" : med_inc,
            "HouseAge" : house_age,
            "AveRooms" : ave_rooms,
            "AveBedrms" : av_bedrms,
            "Population" : popu,
            "AveOccup" : ave_occup,
            "Latitude" : lat,
            "Longitude" : long
        }
        st.json(payload)
        try:
            with st.spinner("Predicting House price...."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=(10,120)
                )
        # st.write(response.status_code)
            if response.status_code ==200:
            
                result = response.json()
                price = result["predicted_price"]
                st.metric(
                    label="Predicted House Price",
                    value=price
                )
                price_range = result.get("fidence range")
                st.success("House Price Predicted Successfully!")
                st.divider()
                st.info(f"Estimated price range: {price_range}")
                st.caption("This is an estimated error band, not a guaranteed price range.")
                
            elif response.status_code == 422:
                st.error("Invalid house features. Please check your inputs.")
            elif response.status_code == 500:
                st.error("Prediction failed due to a backend error.")
            else:
                st.error(f"Request failed with status code: {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Please check if the server is running.")
        except requests.exceptions.RequestException as e:
             st.error(f"API request failed: {e}")
        except (ValueError, KeyError, TypeError):
            st.error("Received an invalid response from the prediction server.")

# Batch prediction tab

with tab2:
    st.subheader("Upload Dousing Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )
    if uploaded_file is not None:
        # CSV Preview
        st.subheader("Dataset Preview")
        preview_df = pd.read_csv(
            io.BytesIO(uploaded_file.getvalue())
        )
        show = st.checkbox("Show All Uploaded Records",key="dataset_preview_checkbox")
        if show:
            st.dataframe(preview_df)
        else:
            st.dataframe(preview_df.head(5))
        st.write("Uploaded File:", uploaded_file.name)
        # Dataset Statistics
        st.subheader("Dataset Overview")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="Total Houses",
                value=preview_df.shape[0],
                border=True
            )

        with metric_col2:
            st.metric(
                label="Total Features",
                value=preview_df.shape[1],
                border=True
            )

        st.divider()
        st.write(uploaded_file.name)
        st.write(f"File size: {uploaded_file.size/1024:,.2f} KB")
        st.success(f"{uploaded_file.name} uploaded successfully")
        
        predict_all = st.button("Predict All Houses")      
        
        if predict_all:
            files = {
                "file":(uploaded_file.name, uploaded_file.getvalue(),"text/csv")
            }
            try:
                st.session_state.pop("batch_csv", None)
                response = requests.post(
                    f"{API_URL}/predict-file",
                    files=files,
                    timeout=(10,120)                  
                )
                # st.write(response.status_code)
                if response.status_code == 200:
                    csv_data = response.content
                    st.session_state["batch_csv"] = csv_data
                    st.success("Batch Prediction successfull")
                elif response.status_code == 400:
                    error = response.json()
                    st.error(error.get("detail", "Invalid CSV file"))
                elif response.status_code == 500:
                    st.error("Prediction failed due to a backend error.")
                else:
                    st.error(f"Batch prediction failed: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Please start the backend server.")
                
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")       
                            
        if "batch_csv" in st.session_state:
            # Predicted Results Preview
            if "batch_csv" in st.session_state:
                st.subheader("Predicted Results Preview")
                result_df = pd.read_csv(
                    io.BytesIO(st.session_state["batch_csv"])
                )
                show_all = st.checkbox("Show All Predictions")
                if show_all:
                    st.dataframe(result_df)
                else:
                    st.dataframe(result_df.head(5))
                st.write(f"Total Predicted Houses: {len(result_df)}")
                # Batch Prediction Analytics
                st.subheader("📊 Prediction Analytics")

                # Convert formatted prices into numeric values
                prices = pd.to_numeric(
                    result_df["predicted_columns_usd"]
                    .str.replace(r"[$,]", "", regex=True)
                )
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="Total Predictions",
                        value=len(prices),
                        border=True
                    )
                with col2:
                    st.metric(
                        label="Average Price",
                        value=f"${prices.mean():,.0f}",
                        border=True
                    )
                with col3:
                    st.metric(
                        label="Minimum Price",
                        value=f"${prices.min():,.0f}",
                        border=True
                    )
                with col4:
                    st.metric(
                        label="Maximum Price",
                        value=f"${prices.max():,.0f}",
                        border=True
                    )

                # Prediction Visualization
                st.divider()
                st.subheader("Predicted Housing Values")
                chart_df = pd.DataFrame({
                    "House": range(1, len(prices) + 1),
                    "Predicted Price (USD)": prices.to_numpy()
                })

                st.bar_chart(
                    chart_df.set_index("House"),
                    y="Predicted Price (USD)"
                )
                st.download_button(
                        label="Download Predictions",
                        data= st.session_state["batch_csv"],
                        file_name=f"Predictions_{uploaded_file.name}",
                        mime="text/csv"
                    )
        
