import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import sys
import os
@dataclass
class modelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting independent and dependent variables from train and test data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'linearRegression' :LinearRegression(),
                'Ridge': Ridge(),
                'Lasso' : Lasso(),
                'ElasticNet' : ElasticNet(),
                'Decision Tree': DecisionTreeRegressor()
            }
            model_report: dict = evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n=========================================================')
            logging.info(f'model report : {model_report}')
            #To get the best model score from dictonary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f"Best model found, model name : {best_model_name}, R2-Score : {best_model_score}")
            print("\n======================================================")
            logging.info(f"Best model found, model name : {best_model_name}, R2-Score : {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("exception occured at model training")
            raise CustomException(e,sys)