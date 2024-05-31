import streamlit as st
import pandas as pd
import os
from automl_forecasting import H2OModel
import plot
from datetime import timedelta
import h2o

st.set_page_config(
    page_title="Forecast Engine",
    page_icon="logo.png"
)

# info feature
with st.expander("Streamlit app to build a forecasting model in a few clicks", expanded=False):
    app_intro = """
                This app allows you to train, evaluate and optimize a Forecasting model in just a few clicks.
                All you have to do is to upload a forecasting dataset, and follow the guidelines in the sidebar to:
                * __Prepare data__: Choose the dataset and the target column you want to predict.
                * __Processing__: Once your data is ready, you can start to build your model. It shows the prediction results and what parameter is impacting forecasts the most.
                * __Forecast__: Forecast target values for future dates.

                """
    st.write(app_intro)
    st.write("")

# read saved data from local
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image('logo.png', width=200)
    st.title("Forecast ML Engine")

st.sidebar.title("1. Data")
# input data and target
with st.sidebar.expander("Dataset", expanded=True):
    file = st.file_uploader("Upload a csv file", type="csv")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)

# Column names
with st.sidebar.expander("Details", expanded=True):
    cols = df.columns.tolist()
    choosen_cols = st.multiselect(
        "Select column",
        cols,
        default=cols
    )
    date_col = st.selectbox("Date column", sorted(choosen_cols))
    target_col = st.selectbox("Target column", sorted(set(choosen_cols) - {date_col}))
    model_name = st.text_input('Model Name')

st.sidebar.title("2. Processing")
modelling = st.sidebar.checkbox("Build model from data", value=True)
with st.sidebar.expander("Metrics", expanded=False):
    metrics = st.multiselect(
        "Select evaluation metrics",
        ["MAE", "MSE", "RMSE", "RMSLE"],
        default=["MAE"]
    )

st.sidebar.title("3. Forecast")
predict = st.sidebar.checkbox("Predict future dates", value=False)
with st.sidebar.expander("Horizon", expanded=False):
    days = st.number_input('Forecast in days', step=1, min_value=1)
    

if st.checkbox('Launch forecast'):
    if modelling:
        df = pd.read_csv('dataset.csv')
        data = df[choosen_cols].head()
        st.title('Data')
        st.dataframe(data)
        # run model automl
        automl = H2OModel(df, target_col, date_col)
        automl.run_modelling()

        modified_df = automl.modified_data
        modified_df.to_csv('modified_df.csv')
        
        model = automl.model
        
        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_path = h2o.save_model(model=model, path=model_directory, filename='forecasting_model', force=True)
        
        st.header('Processing Results')
        df_results = automl.get_prediction_result()
        show_df = df_results.head()
        st.dataframe(show_df)
        
        # Display mae
        mae = automl.get_mae()
        mae_formatted = format(mae, '10.2f')
        
        st.subheader(f'''Error {mae_formatted} ''')
        
        plot.plot_actual_vs_forecast(df_results, df.index, target_col)
    
    if predict:
        modified_df = pd.read_csv('modified_df.csv')
        date_col = modified_df.columns[0]
        y_target = modified_df.columns[1]
        
        modified_df[date_col] = pd.to_datetime(modified_df[date_col])
        modified_df = modified_df.set_index(date_col)
        
        modified_df[y_target] = pd.to_numeric(modified_df[y_target], errors='coerce')
        modified_df = modified_df.fillna(method='ffill').dropna()

        # Load model using model_path
        model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'forecasting_model')
        saved_model = h2o.load_model(model_directory)

        # Make future predictions
        last_row = modified_df.iloc[-1].copy()
        last_date = last_row.name  # a datetime object

        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)

        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_predictions = []

        for date in future_dates:
            last_row[date_col] = date

            # Ensure lag columns are created
            for lag in [1, 2, 3]:
                if f'lag_{lag}' not in last_row:
                    last_row[f'lag_{lag}'] = last_row[y_target] if lag == 1 else modified_df[f'lag_{lag-1}'].iloc[-1]

            # Recalculate rolling features
            last_row['rolling_mean_7'] = modified_df[y_target].rolling(window=7).mean().iloc[-1]
            last_row['rolling_std_7'] = modified_df[y_target].rolling(window=7).std().iloc[-1]

            # Prepare data for prediction
            last_row_df = pd.DataFrame(last_row).T
            h2o_last_row = h2o.H2OFrame(last_row_df)
            prediction = saved_model.predict(h2o_last_row)
            predicted_value = prediction.as_data_frame().iloc[0, 0]

            future_predictions.append(predicted_value)
            last_row[y_target] = predicted_value
            
        future_df = pd.DataFrame({'date': future_dates, 'predicted_value': future_predictions})
        
        st.header("Forecast Results")
        st.dataframe(future_df)
 
    


# if st.sidebar.button('Get the prediction results'):
#     data_test = automl.data_test
#     result = data_test.concat(result, axis=1)
#     result = result.as_data_frame()
#     result.to_csv('result.csv', index=None)
#     with open('result.csv', 'rb') as f: 
#         st.download_button('Download Data', f, file_name='result.csv')
    
# # reset data
# st.sidebar.title("4. Download")
# st.sidebar.checkbox('Track experiments', False)
# if st.sidebar.button('Download forecasting results'):
#     pass
        
st.sidebar.title("\n\n\n\n")

