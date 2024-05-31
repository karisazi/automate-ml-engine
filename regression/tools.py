import streamlit as st
import pandas as pd

file = st.file_uploader(
            "Upload a csv file", type="csv"
        )
if file: 
    df = pd.read_csv(file, index_col=None)
    st.write(df)


# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# import streamlit as st
# import numpy as np

# def encode(from_df, to_encode_df):    
#     encoder = OneHotEncoder(sparse=False, drop='first')
#     categories_df = from_df.select_dtypes(include='object')
#     st.write(categories_df)
#     encoder.fit(categories_df)
    
#     cat_custom_pred_df = to_encode_df.select_dtypes(include='object')
#     st.write(cat_custom_pred_df.dtypes)
#     encoded_data = encoder.transform(cat_custom_pred_df)
    
#     encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categories_df.columns))

#     encoded_columns = encoder.get_feature_names_out(cat_custom_pred_df.columns)
#     remaining_columns = [col for col in to_encode_df.columns if col not in categories_df.columns]
#     df_encoded = pd.concat([to_encode_df[remaining_columns], encoded_df], axis=1)
#     # df_encoded = pd.concat([custom_pred_df.drop(columns=categories_df.columns), encoded_df], axis=1)
        
    
#     # encoded_custom_df = automl.encode_df(df, custom_pred_df)
#     st.dataframe(df_encoded)
    
    
# df = pd.read_csv('diamonds.csv')

# values = ["1",0.23,"Ideal","E","SI2",61.5,55,326,3.95,3.98,2.43]
# values = np.array(values).reshape(1, len(df.columns.tolist()))
# custom_df = pd.DataFrame(data=values, columns=df.columns)

# # st.write(df.head())
# # st.write(custom_df)

# encode(df, custom_df)


