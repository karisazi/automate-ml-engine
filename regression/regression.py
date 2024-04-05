import streamlit as st
import pandas as pd
import os 
from automl_regression import H2OModel

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
    st.title('Data')
    st.dataframe(df)


with st.sidebar: 
    st.image('https://i.ibb.co/FqD53pF/logo.png', width=200)
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
    
        y_target = st.selectbox('Choose the Target Column', df.columns)



st.sidebar.title("2. Processing")
if st.sidebar.checkbox(
            "Build model from data", False
        ):
    # run model automl
    automl = H2OModel(df, y_target)
    automl.run_modelling()
    
    st.title('Prediction Results')
    data_pred = automl.get_prediction_result()
    st.dataframe(data_pred.as_data_frame())
    
    # Display mae 
    mae = automl.get_mae()
    mae_formatted = "{:.2f}".format(mae)
    
    st.write(f'''<p style='color: blue;'>On average, our predictions may deviate from the actual sales numbers by up to {mae_formatted} units.
                Therefore, when using our regression for decision-making, we should be aware that the actual value
                could be below or higher than the predicted values by approximately {mae_formatted} units. Understanding this
                potential variability allows us to make more informed decisions and account for potential fluctuations in business.</p>''',
                unsafe_allow_html=True)

    # Show important variables   
    st.title('Important Factors')
    st.write(f'Below are some of the most important features that affect our {y_target} value')
    varimp = automl.get_important_features()
    
    for count, var in enumerate(varimp, start=1):
        st.write(f'{count}. {var}')

    # save predicted values as result
    result = automl.result

# download predicted results as dataset csv
st.sidebar.title("3. Download")
if st.sidebar.button('Get the prediction results'):
    data_test = automl.data_test
    result = data_test.concat(result, axis=1)
    result = result.as_data_frame()
    result.to_csv('result.csv', index=None)
    with open('result.csv', 'rb') as f: 
        st.download_button('Download Data', f, file_name='result.csv')
    
# reset data
st.sidebar.title("4. Reset")
if st.sidebar.button('Delete data'):
    if os.path.exists('./dataset.csv'): 
        os.remove('./dataset.csv') 
        
st.sidebar.title("\n\n\n\n")