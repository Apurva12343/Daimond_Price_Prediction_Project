from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import sys,os
from dataclasses import dataclass
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
## Data Transformation Config
@dataclass
class dataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
## Data Ingestionconfig Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationconfig()
    def get_data_transformation_obj(self):
        try:
            logging.info("Data transformation has started")
            #define which column should be ordinal encoded and which should be scaled
            categorical_col = ['cut', 'color','clarity']
            numerical_col = ['carat', 'depth','table', 'x', 'y', 'z']
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info("Pipeline has initiated")
            #Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_col),
            ('cat_pipeline',cat_pipeline,categorical_col)
            ])
            return preprocessor
            logging.info("Pipeline Completed")
        except Exception as e:
            logging.info("Error occured in data transformation")
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Training and test data")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_obj()
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            #Dividing features into depndent and independent
            input_feature_train_df = train_df.drop(columns = drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns = drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            #Apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing obj on training and testing dataset")
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            save_object(
                file_path=dataTransformationconfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessor pickle is created")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error occured in initiate data transformation")
            raise CustomException(e,sys)