import yaml
from src.DataPreprocessing import DataPreprocessing
from src.ModelTrainer import ModelTrainer
from src.ModelPredictor import ModelPrediction
from src.ModelEvaluation import ModelEvaluation


params_preprocess=yaml.safe_load(open("params.yaml"))['preprocess']
params_trainer=yaml.safe_load(open("params.yaml"))['train']
params_eval=yaml.safe_load(open("params.yaml"))['evaluation']
params_model=yaml.safe_load(open("params.yaml"))
params_mlflow=yaml.safe_load(open("params.yaml"))['mlflow']


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
model_predictor=ModelPrediction(params_eval['X_test'],params_model['model'])
y_pred=model_predictor.ModelPredict()
print(y_pred)


#Step-4
model_eval=ModelEvaluation(params_eval['y_test'],y_pred,params_mlflow['uri'],params_model['model'],params_trainer['X_train'])
report=model_eval.ModelEval()
print(report)