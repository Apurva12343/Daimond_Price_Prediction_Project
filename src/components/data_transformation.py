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
## Data Transformation Config
@dataclass
class dataIngestionconfig:
    
## Data Ingestionconfig Class