import streamlit as st
import pandas as pd
import numpy as np
import os 
from automl_regression import H2OModel
import requests


URL_get_csrf_token = 'http://127.0.0.1:5000/get_csrf_token'
URL_train = 'http://127.0.0.1:5000/regression/train'
URL_predict = 'http://127.0.0.1:5000/regression/predict'


# #get csrf token of API
# url = 'http://127.0.0.1:5000/get_csrf_token'
# response = requests.get(url)
# response = response.json()
# csrf_token = response['csrf_token']

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
file_to_request = None
with st.sidebar.expander("Dataset", expanded=True):
    file = st.file_uploader(
            "Upload a csv file", type="csv"
        )

    
    if file: 
        # df = pd.read_csv(file)
        # df.to_csv('dataset.csv', index=None)
        
        file_to_request = file
    
    y_target = st.selectbox('Choose the Target Column', df.columns)
    model_name = st.text_input('Model Name')
    st.warning("Please provide both the Target Column and Model Name.")
       
    
    automl = H2OModel(df, y_target)




st.sidebar.title("2. Processing")
if st.sidebar.checkbox("Call API", False):
    if file_to_request and y_target and model_name: 
        # Create a session to persist cookies
        session = requests.Session()
        
        # Get CSRF token from the API
        response = session.get(URL_get_csrf_token)
        if response.status_code == 200:
            csrf_token = response.json().get('csrf_token')
            
            if csrf_token:
                payload = {
                    'y_target': y_target,
                    'name': model_name,
                    'csrf_token': csrf_token  # Ensure this matches the server's expected field name
                }

                # Prepare files for multipart/form-data request
                files = {
                    'file': (file_to_request.name, file_to_request, 'text/csv')
                }

                # Send the POST request
                response = session.post(URL_train, data=payload, files=files)

                # Output the response
                st.write(f'Status Code: {response.status_code}')
                st.write(f'Response JSON: {response.json() if response.status_code == 200 else response.text}')
            else:
                st.write("Failed to retrieve CSRF token.")
        else:
            st.write(f"Failed to get CSRF token: Status Code {response.status_code}")
else:
    st.write("Please provide the necessary input parameters.")
    
    

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
if st.sidebar.checkbox("Predict new data", False):
    st.title('Prediction')
    value_to_predict = []
    for col in df.columns:
        selected_dtype = df[col].dtype
        if selected_dtype in ['int64', 'float64']:  # Check for int or float data types
            value = st.number_input(col)
        elif selected_dtype == 'object':  # If it's an object type, you might want to handle it differently
            try:
                pd.to_datetime(df[col])
                value = st.date_input(f'Choose the {col} value')
            except Exception as e:
                values = df[col].unique().tolist()
                value = st.selectbox(f'Choose the {col} value', values)

        value_to_predict.append(value)

    # set inputted values as dataframe
    df_columns = df.columns.tolist()  
    value_to_predict = np.array(value_to_predict).reshape(1, len(df_columns))
    custom_pred_df = pd.DataFrame(data=value_to_predict, columns=df_columns)
    st.dataframe(custom_pred_df)
    
    # encode(df, custom_pred_df)
    

    
    if st.button("Predict"):
        customprediction = automl.get_custompredict(value_to_predict)
        # Convert H2OFrame to pandas DataFrame
        customprediction_df = customprediction.as_data_frame()

        # Pick the last row
        last_row = customprediction_df.iloc[-1]

        
        st.subheader(last_row)

        # st.write(predictions[:, 'predict'][1, 0] )
    
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