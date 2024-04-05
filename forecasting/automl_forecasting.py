import h2o
from h2o.automl import H2OAutoML
import pandas as pd

class H2OModel:
    def __init__(self, df, y_target):
        self.df = df
        self.y_target = y_target
        self.data_train = None
        self.data_test = None
        self.x_features = None
        self.result = None
        self.model = None
        self.leaderboard = None

    def run_modelling(self):
        df_processed = process_data(self.df, self.y_target)
        # df_processed.reset_index(drop=True,inplace=True)

        h2o.init(nthreads=-1)
        self.data_train = h2o.H2OFrame(df_processed.loc[:int(df_processed.shape[0]*0.8),:])
        self.data_test = h2o.H2OFrame(df_processed.loc[int(df_processed.shape[0]*0.8):,:])
        
        self.x_features = self.df.columns.tolist()
        self.x_features = [x for x in self.x_features if x != self.y_target]

        aml = H2OAutoML(max_runtime_secs = 600, seed = 42)
        aml.train(x=self.x_features, y=self.y_target, training_frame=self.data_train, leaderboard_frame = self.data_test)
        self.model = aml.leader
        self.leaderboard = aml.leaderboard
        self.result = self.model.predict(self.data_test)
        
    def get_model(self):
        return self.model
    
    def get_leaderboard(self):
        return self.leaderboard

    def get_mae(self):
        return self.model.mae(valid=True)

    def get_shap(self):
        return self.model.shap_summary_plot(self.data_test)
    
    def get_prediction_result(self):
        df_results = pd.DataFrame()
        df_results['ground_truth'] = self.data_test[self.y_target].reset_index(drop=True)
        df_results['predictions'] = h2o.as_list(self.result,use_pandas=True)
        df_results['err'] = df_results[self.y_target] - df_results['predict']
        
        return df_results
    
    def get_important_features(self):
        varimp = self.model.varimp(use_pandas=True)['variable']
        
        return varimp[:5]


def process_data(df, target):
    numerical_df = df.select_dtypes(include=['number'])
    df2 = numerical_df.copy()
    num_lags = 3 # number of lags and window lenghts for mean aggregation
    delay = 1 # predict target one step ahead
    for column in df2:
        for lag in range(1,num_lags+1):
            df2[column + '_lag' + str(lag)] = df2[column].shift(lag*-1-(delay-1))
            if column != 'wnd_dir':
                df2[column + '_avg_window_length' + str(lag+1)] = df2[column].shift(-1-(delay-1)).rolling(window=lag+1,center=False).mean().shift(1-(lag+1))

    df2.dropna(inplace=True)

    mask = (df2.columns.str.contains(target) | df2.columns.str.contains('lag') | df2.columns.str.contains('window'))
    df_processed = df2[df2.columns[mask]]

    return df_processed