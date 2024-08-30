import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging 

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_fiel_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Initiating Model Trainer")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            models = {
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'SVR': SVR(),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'CatBoost': CatBoostRegressor(),
                'XGBoost': XGBRegressor()
            }
            
            model_params = {
                'KNeighborsRegressor': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                'DecisionTreeRegressor': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 3]},
                'RandomForestRegressor': {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]},
                'AdaBoostRegressor': {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0], 'loss': ['linear', 'square', 'exponential']},
                'SVR': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'epsilon': [0.1, 0.01, 0.001]},
                'LinearRegression': {},
                'Ridge': {'alpha': [0.1, 1, 10], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], 'tol': [0.001, 0.0001, 0.00001]},
                'Lasso': {'alpha': [0.1, 1, 10], 'selection': ['cyclic', 'random'], 'max_iter': [1000, 2000, 3000]},
                'CatBoost': {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1.0], 'depth': [4, 6, 8]},
                'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}
            }
            
            logging.info("Evaluating Model")
            model_res, stacking_model = evaluate_model(X_train, y_train, X_test, y_test, models, model_params)
            
            logging.info("Saving Model")
            save_object(self.model_trainer_config.trained_model_fiel_path, obj=stacking_model)
            r2_scores = model_res['final_r2'].tolist()

            return r2_scores
            
            
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())