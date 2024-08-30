import pandas as pd
import numpy as np

import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # create a directory to save the preprocessed data created with pipeline
    preprocess_obj_file_path: str=os.path.join('artifacts','preprocess_obj.pkl')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Initiating data transformation.")
            num_cols = ['reading score', 'writing score']
            cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            # build the preprocessing pipeline
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            logging.info('Categorical columns: {}'.format(cat_cols))
            logging.info('Numerical columns: {}'.format(num_cols))
            
            preprocessor = ColumnTransformer(
                [('num', num_transformer, num_cols),
                 ('cat', cat_transformer, cat_cols)]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read data from the dataset.")
            
            logging.info("Obatining the preprocessed object.")
            preprocess_obj = self.get_data_transformation_object()
            
            target_col = 'math score'   
            
            feature_train = train_df.drop(target_col, axis=1)
            target_train = train_df[target_col]
            
            feature_test = test_df.drop(target_col, axis=1)
            target_test = test_df[target_col]
            
            logging.info("Transforming the train data.")
            feature_train = preprocess_obj.fit_transform(feature_train)
            feature_test = preprocess_obj.transform(feature_test)
            
            # combine the features and target
            train_data_arr = np.column_stack((feature_train, target_train))
            test_data_arr = np.column_stack((feature_test, target_test))
            
            logging.info("Saving the preprocessed object.")
            
            save_object(file_path=self.transformation_config.preprocess_obj_file_path, obj=preprocess_obj)
            
            return train_data_arr, test_data_arr, self.transformation_config.preprocess_obj_file_path
        except Exception as e:
            raise CustomException(e, sys.exc_info())
            
            
    
    