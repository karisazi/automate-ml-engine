import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os 
from automl_forecasting import H2OModel
import plot
import warnings
import locale
locale.setlocale(locale.LC_ALL, 'en_GB.UTF-8')
# import locale
# locale.setlocale(locale.LC_ALL, "de_DE")
warnings.filterwarnings('ignore')

# info feature
with st.expander(
    "Streamlit app to build a forecasting model in a few clicks", expanded=False
):
    app_intro = """
                This app allows you to train, evaluate and optimize a Forecasting model in just a few clicks.
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
    st.image('https://i.ibb.co/WnLCgrs/9162969-removebg-preview.png', width=200)
    st.title("Forecasting ML Engine")
    
st.sidebar.title("1. Data")
# input data and target    
with st.sidebar.expander("Dataset", expanded=True):
    file = st.file_uploader(
            "Upload a csv file", type="csv"
        )
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        
# Column names
with st.sidebar.expander("Columns", expanded=True):
    date_col = st.selectbox("Date column",sorted(df.columns))
    target_col = st.selectbox("Target column",sorted(set(df.columns) - {date_col}))


st.sidebar.title("2. Processing")
if st.sidebar.checkbox(
            "Build model from data", False
        ):
    df = pd.read_csv('dataset.csv',index_col=df.columns.get_loc(date_col))
    st.write(df)
    # run model automl
    automl = H2OModel(df, target_col)
    automl.run_modelling()
    # st.dataframe(automl.leaderboard.as_data_frame())
    
    st.title('Prediction Results')
    df_results = automl.get_prediction_result()
    df_results['prediction'] = df_results['prediction'].apply(lambda x: format(x, '10.2f'))
    df_results['residual'] = df_results['residual'].apply(lambda x: format(x, '10.2f'))
    st.dataframe(df_results)
    
    # Display mae 
    mae = automl.get_mae()
    mae_formatted = format(mae, '10.2f')
    
    st.write(f'''On average, our predictions may deviate from the actual sales numbers by up to <span style='color: green;'>{mae_formatted} units </span>.
                Therefore, when using our regression for decision-making, we should be aware that the actual value
                could be below or higher than the predicted values by approximately <span style='color: blue;'>{mae_formatted} units </span>. Understanding this
                potential variability allows us to make more informed decisions and account for potential fluctuations in business.''',
                unsafe_allow_html=True)
    
    plot.plot_actual_vs_forecast(df_results, df.index, target_col)
    

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

st.title('Data')
