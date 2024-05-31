import streamlit as st
import pandas as pd
import numpy as np
import os 
from automl_regression import H2OModel
import h2o
from h2o.automl import H2OAutoML
import requests
import json
# from tools import encode



# h2o.init()

# info feature
with st.expander(
    "Streamlit app to build a regression model in a few clicks", expanded=False
):
    app_intro = """
                This app allows you to train, evaluate and optimize a Regression model in just a few clicks.
                All you have to do is to upload a regression dataset, and follow the guidelines in the sidebar to:
                * __Prepare data__: Choose the dataset and the target column you want to predict.
                * __Processing__: Once your data is ready, you can start to build your model. It shows the prediction results and what parameter is impacting forecasts the most.
                * __Download__: Download the prediction results.
                * __Reset__: It can reset the data from storage if want to. \n


                Once you are satisfied, click on "Download" to save data locally.
                """
    st.write(app_intro)
    st.write("")

# read saved data from local
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 
    st.image('logo.png', width=200)
    st.title("Regression ML Engine")
    
st.sidebar.title("1. Data")
# input data and target    
with st.sidebar.expander("Dataset", expanded=True):
    file = st.file_uploader(
            "Upload a csv file", type="csv"
        )
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
    
        # y_target = st.selectbox('Choose the Target Column', df.columns)
        
        # hf = h2o.H2OFrame(df)
        
        # automl = H2OModel(df, y_target, hf)
        # automl = H2OModel(df, y_target)
    y_target = st.selectbox('Choose the Target Column', df.columns)
    model_name = st.text_input('Model Name')
    st.warning("Please provide both the Target Column and Model Name.")
    automl = H2OModel(df, y_target)




st.sidebar.title("2. Processing")

with st.sidebar.expander("Metrics", expanded=False):
    metrics = st.multiselect(
        "Select evaluation metrics",
        ["MAE", "MSE", "RMSE", "RMSLE"],
        default=["MAE", "MSE", "RMSE", "RMSLE"]
        # help=readme["tooltips"]["metrics"],
    )
        
if st.sidebar.checkbox(
            "Build model from data", False
        ):
    st.title('Data')
    st.dataframe(df)
    # run model automl
    automl.run_modelling()
    
    st.title('Prediction Results')
    data_pred = automl.get_prediction_result()
    st.dataframe(data_pred.head())
    
    # Display mae 
    mae = automl.get_mae(metrics)
    
    mae_formatted = "{:.2f}".format(mae)
    
    st.write(f'''On average, our predictions may deviate from the actual sales numbers by up to <span style='color: blue;'>{mae_formatted} units </span>.
                Therefore, when using our regression for decision-making, we should be aware that the actual value
                could be below or higher than the predicted values by approximately <span style='color: blue;'>{mae_formatted} units </span>. Understanding this
                potential variability allows us to make more informed decisions and account for potential fluctuations in business.''',
                unsafe_allow_html=True)

    # Show important variables   
    st.title('Important Factors')
    st.write(f'Below are some of the most important features that affect our {y_target} value')
    varimp = automl.get_important_features()
    
    for count, var in enumerate(varimp, start=1):
        st.write(f'{count}. {var}')

    # save predicted values as result
    result = automl.result
    
    # st.markdown("---")

    # satisfaction_question = st.text("Are you satisfied with the result?")

    # if satisfaction_question:
    #     save_button_clicked = st.button('Save final model')
    #     if save_button_clicked:
    #         pass
    #         # Code to save the final model
    
    # st.dataframe(automl.error_df)
    # st.dataframe(automl.importantvar_df)



st.sidebar.title("3. Prediction")
def get_csrf_token(session, url):
    response = session.get(url)
    response.raise_for_status()  # Ensure the request was successful
    st.write(f"Cookies: {response.cookies}")  # Debug print all cookies
    for cookie in response.cookies:
        st.write(f"Cookie Name: {cookie.name}, Value: {cookie.value}")  # Debug print each cookie
    return response.cookies['csrf_token']  # Replace with the correct cookie name if different

if st.sidebar.checkbox("Predict new data", False):
    st.title('Prediction')
    
    value_to_predict = []
    
    for col in df.columns:
        selected_dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(selected_dtype):  # Check for numeric data types
            value = st.number_input(f'Enter the {col}', key=col)
        
        elif pd.api.types.is_datetime64_any_dtype(selected_dtype):  # Check for datetime data types
            value = st.date_input(f'Choose the {col} value', key=col)
            value = value.strftime('%Y-%m-%d')  # Convert date to string format if needed
        
        elif pd.api.types.is_object_dtype(selected_dtype):  # Check for object data types
            unique_values = df[col].unique().tolist()
            value = st.selectbox(f'Choose the {col} value', unique_values, key=col)
        
        value_to_predict.append(value)
    
    # Create DataFrame from inputted values
    custom_pred_df = pd.DataFrame([value_to_predict], columns=df.columns)
    
    # Convert DataFrame to JSON
    json_result = custom_pred_df.to_json(orient='records')
    data = json.loads(json_result)  # This creates a list of dictionaries
    json_result = data[0]  # Get the first (and only) record as a dictionary
    
    st.write(json_result)
    
    if st.button("Predict"):
        session = requests.Session()
        csrf_url = 'http://127.0.0.1:5000/'  # Replace with the correct URL to get CSRF token
        predict_url = 'http://127.0.0.1:5000/regression/predict'  # Replace with your server URL

        try:
            # Step 1: Get CSRF token
            csrf_token = get_csrf_token(session, csrf_url)
            
            headers = {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrf_token  # Include the CSRF token in the headers
            }
            
            payload = {
                'Name': 'model_name',  # Replace with actual model name if required
                'Data': json_result  # Send as dictionary
            }
            
            st.write(f"Payload: {payload}")  # Debug print
            
            # Step 2: Make the prediction request with CSRF token
            response = session.post(predict_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)

            # Try to parse JSON response
            try:
                response_json = response.json()
                st.write(f'Response JSON: {response_json}')
            except json.JSONDecodeError:
                st.error('Error decoding JSON response from server.')
                st.write(f'Response text: {response.text}')
        
        except requests.RequestException as e:
            st.error(f'Error with request: {e}')
            st.write(f'Response status code: {e.response.status_code}')
            st.write(f'Response content: {e.response.content}')
    
# download predicted results as dataset csv
st.sidebar.title("4. Download")
if st.sidebar.button('Get the prediction results'):
    data_test = automl.data_test
    result = data_test.concat(result, axis=1)
    result = result.as_data_frame()
    result.to_csv('result.csv', index=None)
    with open('result.csv', 'rb') as f: 
        st.download_button('Download Data', f, file_name='result.csv')
    
# reset data
# st.sidebar.title("4. Reset")
# if st.sidebar.button('Delete data'):
#     if os.path.exists('./dataset.csv'): 
#         os.remove('./dataset.csv') 
        
st.sidebar.title("\n\n\n\n")