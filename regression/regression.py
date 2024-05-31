import streamlit as st
import pandas as pd
import numpy as np
import os 
from automl_regression import H2OModel
import h2o
from h2o.automl import H2OAutoML
import json
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Regression Engine",
    page_icon="logo.png"  # Replace with your logo file path or URL
)
# info feature
with st.expander(
    "Streamlit app to build a regression model in a few clicks", expanded=False
):
    app_intro = """
                This app allows you to train, evaluate and optimize a Regression model in just a few clicks.
                All you have to do is to upload a regression dataset, and follow the guidelines in the sidebar to:
                * __Prepare data__: Choose the dataset and the target column you want to predict.
                * __Processing__: Once your data is ready, you can start to build your model. It shows the prediction results and what parameter is impacting forecasts the most.
                * __Prediction__: Predict custom values \n


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

    cols = df.columns.tolist()
    choosen_cols = st.multiselect(
        "Select independent variables",
        cols,
        default=cols
    )
    y_target = st.selectbox('Select dependent variable', df.columns)
    
    df = df[choosen_cols + [y_target]]
    model_name = st.text_input('Model Name')
    # st.warning("Please provide both the Target Column and Model Name.")
    automl = H2OModel(df, y_target)




st.sidebar.title("2. Processing")

with st.sidebar.expander("Metrics", expanded=False):
    metrics = st.multiselect(
        "Select evaluation metrics",
        ["MAE", "MSE", "RMSE", "RMSLE"],
        default=["MAE"]
    )
        
modelling = st.sidebar.checkbox("Build model from data", value=True)


st.sidebar.title("3. Prediction")
predict = st.sidebar.checkbox("Predict new data", value=False)
if predict:
    st.sidebar.warning('Please do not let column left empty')

value_to_predict = []
cols_to_predict = [x for x in df.columns if x != y_target]
with st.sidebar.expander("Column to predict", expanded=False):
        for col in cols_to_predict:
            selected_dtype = df[col].dtype
            
            if selected_dtype in ['int64', 'float64']:  # Check for int or float data types
                value = st.number_input(col)
            elif selected_dtype == 'object':  # Check for object data types
                value = st.text_input(col)
            
            value_to_predict.append(value)

if st.checkbox('Launch regression'):
    if modelling:
        st.title('Data')
        st.dataframe(df)
        # run model automl
        automl.run_modelling()
        
        st.title('Processing Results')
        data_pred = automl.get_prediction_result()
        st.dataframe(data_pred.head())
        
        # Display mae 
        mae = automl.get_mae(metrics)
        
        mae_formatted = "{:.2f}".format(mae)
        
        st.subheader(f'''Error: {mae_formatted} ''')
        
        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        os.makedirs(model_directory, exist_ok=True)
        h2o.save_model(model=automl.model, path=model_directory, filename=f'regression', force=True)

        # Show important variables   
        st.title('Important Factors')
        st.write(f'Below are some of the most important features that affect our {y_target} value')
        
        varimp = automl.get_important_features()
        
        for count, var in enumerate(varimp, start=1):
            st.write(f'{count}. {var}')

        # plot varimp
        automl.model.varimp_plot()
        fig = plt.gcf()
        st.pyplot(fig)
        
        result = automl.result


    if predict:
        st.title('Prediction')
        
        # Create DataFrame from inputted values
        custom_pred_df = pd.DataFrame([value_to_predict], columns=cols_to_predict)
        
        # Convert DataFrame to H2OFrame
        custom_pred_hf = h2o.H2OFrame(custom_pred_df)
        
        # if st.button('Predict'):
        model_file = 'regression'
        model_path = os.path.join(model_directory, model_file)
        saved_model = h2o.load_model(model_path)
            
            # Make predictions
        predictions = saved_model.predict(custom_pred_hf)
        predictions_list = predictions.as_data_frame().values.flatten().tolist()
        prediction = format(predictions_list[0], '10.2f')
        st.subheader(f'{y_target} Prediction: {prediction}')
        
        
st.sidebar.title("\n\n\n\n")
