import yaml
from src.DataPreprocessing import DataPreprocessing


params_preprocess=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))['preprocess']

DataIngestion=DataPreprocessing(params_preprocess['input'],params_preprocess['output'],params_preprocess['X_sm'],params_preprocess['y_sm'])
DataIngestion.DataPreprocess()