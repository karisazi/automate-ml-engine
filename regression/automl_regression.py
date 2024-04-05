import h2o
from h2o.automl import H2OAutoML

class H2OModel:
    def __init__(self, df, y_target):
        self.df = df
        self.y_target = y_target
        self.hf = None
        self.data_train = None
        self.data_test = None
        self.data_valid = None
        self.x_features = None
        self.result = None
        self.aml = None
        self.model = None
        self.mae = None
        self.shap = None

    def run_modelling(self):
        h2o.init()
        self.hf = h2o.H2OFrame(self.df)
        self.data_train, self.data_test, self.data_valid = self.hf.split_frame(ratios=[.8, .1])
        self.x_features = self.df.columns.tolist()
        self.x_features = [x for x in self.x_features if x != self.y_target]

        self.aml = H2OAutoML(max_models=10, seed=10, verbosity="info", nfolds=0)
        self.aml.train(x=self.x_features, y=self.y_target, training_frame=self.data_train, validation_frame=self.data_valid)

        self.model = self.aml.leader
        self.result = self.model.predict(self.data_test)

    def get_model(self):
        return self.model

    def get_mae(self):
        return self.model.mae(valid=True)

    def get_shap(self):
        return self.model.shap_summary_plot(self.data_test)
    
    def get_prediction_result(self):
        data_pred_hf = h2o.H2OFrame(self.result)
        data_pred = self.data_test[self.y_target].concat(data_pred_hf, axis=1)
        data_pred['Difference'] = self.data_pred[self.y_target] - self.data_pred['predict']
        return data_pred
    
    def get_important_features(self):
        varimp = self.model.varimp(use_pandas=True)['variable']
        
        return varimp[:5]



