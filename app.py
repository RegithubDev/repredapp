import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO
# from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging
#%%

# Set up the title and company image
st.image("https://via.placeholder.com/150", width=150)  # Replace with your company's logo URL
st.title("Metal Price Forecasting")

# option = st.sidebar.radio("Choose an option", ("Update Data", "Forecasting"))


# Azure Blob connection details
conn_str = "DefaultEndpointsProtocol=https;AccountName=vehiclescrapeanalysis;AccountKey=/GgjrL9aE9p9oJ2XvswZV4wGeob/btLwFVOk2/0Y/Lx4QwbPSm+FPpkV8cTTjWZlJEbGJv9Kl+5k+AStTsWQrw==;EndpointSuffix=core.windows.net"
container_name = "data"

# Initialize blob service
blob_service_client = BlobServiceClient.from_connection_string(conn_str)

def read_csv_from_blob(blob_name):
    #blob_name must be the file name ex: aluminum.csv, steel.csv
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob().readall().decode('utf-8')
    return pd.read_csv(StringIO(blob_data))

def forecast_price_aluminum(df, date_col='Date', price_col='Aluminum Price'):
    # Convert dates to ordinal for regression
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    st.write("**Aluminum Price for last 7 Days**")
    st.write(df.tail(7))
    df=df.set_index(['Date'])
    #NOTE: ETS raises an error if the prices are 0 or -ve values
    y=df
    model = ExponentialSmoothing(
        y,        
        trend='mul', #add
        seasonal='mul', #add
        seasonal_periods=7  # weekly seasonality
    )

    fit = model.fit()

    forecast = fit.forecast(30) 
    forecast_df=pd.DataFrame(forecast).reset_index()
    forecast_df.columns=['Date','Forecasted Price']
    # forecast_df['Forecasted Price']=forecast_df['Forecasted Price'].apply(lambda x:np.ceil(x))
    forecast_df['Forecasted Price']=forecast_df['Forecasted Price'].apply(lambda x:np.round(x,2))   

    
    return forecast_df

def forecast_price_steel(df, date_col='Date', price_col='Steel Prices'):
    # Convert dates to ordinal for regression
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    st.write("**Steel Price for last 7 Days**")
    st.write(df.tail(7))
    #
    df=df.set_index(['Date'])
    #NOTE: ETS raises an error if the prices are 0 or -ve values
    y=df
    y_log = np.log(y + 1)
    model = ExponentialSmoothing(
        y_log,
        trend='mul', #add
        seasonal='mul', #add
        seasonal_periods=7  # weekly seasonality
    )
    
    fit = model.fit()
    forecast_log = fit.forecast(30)
    forecast = np.exp(forecast_log) - 1    
    forecast_df=pd.DataFrame(forecast).reset_index()
    forecast_df.columns=['Date','Forecasted Price']
    # forecast_df['Forecasted Price']=forecast_df['Forecasted Price'].apply(lambda x:np.ceil(x))
    forecast_df['Forecasted Price']=forecast_df['Forecasted Price'].apply(lambda x:np.round(x,2))  
    
    return forecast_df
#%%
def plot_forecast(data,metal):
    # Plot the forecast
    plt.figure(figsize=(10, 6))
    # plt.plot(data['Date'], data['Forecasted Price'], label=f"Actual {metal_choice} Prices")
    plt.plot(data['Date'], data['Forecasted Price'], label=f"Forecasted {metal_choice} Prices", linestyle="--")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Forecasted Price")
    plt.title(f"{metal} Price Forecast")
    plt.xticks(rotation=90)
    st.pyplot(plt)
#%%
def append_to_blob(blob_name, new_data_df):        
    
    
    # Step 1: Read existing data
    try:
        existing_df = read_csv_from_blob(blob_name)
        #check data validity
        if len(new_data_df)!=2:
            logging.error("Extra columns. It should have only two Columns: Date,Aluminum Price")            
            
        #read existing file        
        if list(new_data_df.columns) == list(existing_df.columns):
            flag=1
        else:
            flag=0
        if flag==0:
            if 'Aluminum Price' in new_data_df.columns:
                logging.error("Column Names are not matching. Correct Column names are: Date,Aluminum Price")   
            else:
                logging.error("Column Names are not matching. Correct Column names are: Date,Steel Prices")                  
        
        
    except Exception as e:
        existing_df = pd.DataFrame()  # If file doesn't exist, start new

    # Step 2: Append new data
    #convert date column from new_df
    new_data_df['Date']=pd.to_datetime(new_data_df['Date'])    
    
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    updated_df.drop_duplicates(inplace=True)
    updated_df = updated_df.sort_values(by='Date')

    # Step 3: Convert updated DataFrame to CSV (in memory)
    csv_buffer = StringIO()
    updated_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Step 4: Upload CSV back to Blob (overwrite)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)

#%% MAIN
option = st.sidebar.radio("**Choose an option**", ("Update Data", "Forecasting"))
if option=="Update Data":
    # append new data
    st.write("**Update Data by uploading new data**")
    option1 = st.radio("**Choose an option**", ("Aluminum", "Steel"))
    if option1=="Aluminum":       
        uploaded_file = st.file_uploader("Upload file for Aluminum Prices", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Check if the file is CSV or Excel
            if uploaded_file.name.endswith(".csv"):
                new_data_df = pd.read_csv(uploaded_file)
                st.write("Preview of the uploaded CSV file:", new_data_df.head())
            elif uploaded_file.name.endswith(".xlsx"):
                new_data_df = pd.read_excel(uploaded_file)
                st.write("Preview of the uploaded Excel file:", new_data_df.head())
            blob_name="Aluminum.csv"
            append_to_blob(blob_name, new_data_df)
            
            
    elif option1=="Steel":
        uploaded_file = st.file_uploader("Upload file for Steel Prices", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Check if the file is CSV or Excel
            if uploaded_file.name.endswith(".csv"):
                new_data_df = pd.read_csv(uploaded_file)
                st.write("Preview of the uploaded CSV file:", new_data_df.head())
            elif uploaded_file.name.endswith(".xlsx"):
                new_data_df = pd.read_excel(uploaded_file)
                st.write("Preview of the uploaded Excel file:", new_data_df.head())
            blob_name="Steel.csv"
            append_to_blob(blob_name, new_data_df)
            
    
# elif option=="Forecasting":
else:
    # Dropdown to select between Aluminum and Steel
    metal_choice = st.selectbox("Select Metal", ("Aluminum", "Steel"))

    if metal_choice=='Aluminum':
        df = read_csv_from_blob("Aluminum.csv")
        forecast = forecast_price_aluminum(df)
        forecast["Date"]=forecast["Date"].apply(lambda x:x.strftime('%Y-%m-%d'))
        plot_forecast(forecast, metal_choice)
        st.write("**Forecast for the next 30 days for Aluminum:**")
        st.write(forecast)    
        
    elif metal_choice=='Steel':
        df = read_csv_from_blob("Steel.csv")
        forecast = forecast_price_steel(df)
        forecast["Date"]=forecast["Date"].apply(lambda x:x.strftime('%Y-%m-%d'))
        plot_forecast(forecast, metal_choice)
        st.write("**Forecast for the next 30 days for Steel:**")
        st.write(forecast)   
        # st.dataframe(forecast)



#%%
