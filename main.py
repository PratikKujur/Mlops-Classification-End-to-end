import yaml
from src.DataPreprocessing import DataPreprocessing
from src.ModelTrainer import ModelTrainer


params_preprocess=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))['preprocess']
params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))['train']
params_eval=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))['evaluation']
params_model=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))


#Step-1
DataIngestion=DataPreprocessing(params_preprocess['input'],params_preprocess['output'],params_preprocess['X_sm'],params_preprocess['y_sm'])
DataIngestion.DataPreprocess()


#Step-2
ModelTraining=ModelTrainer(params_preprocess['X_sm'],params_preprocess['y_sm'],
                           params_trainer['X_train'],params_trainer['y_train'],
                           params_eval['X_test'],params_eval['y_test'],
                           params_model['model'])
ModelTraining.StartTraining()

#Step-3

#Step-4