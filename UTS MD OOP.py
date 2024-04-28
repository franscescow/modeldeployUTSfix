#Franscesco William Gazali
#2602138536

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import statistics as sts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle as pkl

class ChurnPrediction:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.input_df = self.df.drop(["Unnamed: 0", "id", "CustomerId", "Surname", "churn"], axis=1)
        self.output_df = self.df['churn']
    
    def fill_na(self):
        numerical_cols = ['Tenure', 'CreditScore', 'Balance', 'EstimatedSalary']
        for col in numerical_cols:
            self.input_df[col].fillna(self.input_df[col].mean(), inplace=True)
        
        categorical_cols = ['Gender', 'Geography']
        for col in categorical_cols:
            mode_val = sts.mode(self.input_df[col])
            self.input_df[col].fillna(mode_val, inplace=True)
    
    def feat_encode(self):
        gender_encode = {"Gender": {"Male": 1, "Female": 0}}
        geo_encode = {"Geography": {"France": 0, "Spain": 1, "Germany": 2, "Other": 3}}
        self.input_df.replace(gender_encode, inplace=True)
        self.input_df.replace(geo_encode, inplace=True)
    
    def resample_data(self):
        os = SMOTE(random_state=0)
        self.input_df, self.output_df = os.fit_resample(self.input_df, self.output_df)
    
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_df, self.output_df, test_size=0.2, random_state=0)

    def preprocess_data(self):
        self.fill_na()
        self.feat_encode()
        self.resample_data()
        self.split_data()
    
    def train_xgboost_model(self):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.fit(self.x_train, self.y_train)
    
    def evaluate_model(self):
        xgb_y_pred = self.xgb_model.predict(self.x_test)
        print('\nXGBoost Report\n')
        print(classification_report(self.y_test, xgb_y_pred, target_names=['1', '0']))
    
    def save_model(self, filename):
        pkl.dump(self.xgb_model, open(filename, 'wb'))

churn_predictor = ChurnPrediction("data_C.csv")
churn_predictor.preprocess_data()
churn_predictor.train_xgboost_model()
churn_predictor.evaluate_model()
churn_predictor.save_model('XGboost_churn.pkl')