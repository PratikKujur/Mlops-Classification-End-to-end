import mlflow
from mlflow.models import infer_signature
import pickle
import pandas as pd
from sklearn.metrics import classification_report
import logging



class ModelEvaluation:
    def __init__ (self,y_test,y_pred,mlflow_id,model_path,X_train):
        self.y_test_path=y_test
        self.y_pred=y_pred
        self.mlflow_id=mlflow_id
        self.model_path=model_path
        self.X_train=X_train
       
    
    def model_tracking(self,mlflow_id,model,report,X_train):

        mlflow.set_tracking_uri(mlflow_id)
        mlflow.set_experiment(experiment_name="Breast_cancer_metrics")
 
        model=pickle.load(open(model,'rb'))
        X_train=pd.read_csv(X_train)
        
        with mlflow.start_run():

            mlflow.log_params(model.get_params()) #logging best parameters

            for class_or_avg, metrics_dict in report.items():# Logging dictionary(classification report)
                if isinstance(metrics_dict, dict):  
                    for metric, value in metrics_dict.items():
                        mlflow.log_metric(f"{class_or_avg}_{metric}", value)
                else: 
                    mlflow.log_metric(class_or_avg, metrics_dict)

            mlflow.log_param("classification_report", report)

            signature=infer_signature(X_train,model.predict(X_train))

            model_info=mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="Breast_cancer",
                signature=signature,
                registered_model_name="tracking-quickstart",
            )


    def ModelEval(self):
        try:
            model_clf=pickle.load(open(self.model_path,'rb'))
            y_test=pd.read_csv(self.y_test_path)
            report=classification_report(y_pred=self.y_pred,y_true=y_test,output_dict=True,target_names=['M','B'])
            logging.info("classification report recorded successfully.")
            self.model_tracking(self.mlflow_id,self.model_path,report,self.X_train)
            return report
            
        except Exception as e:
            logging.exception("Exception occurred: %s", str(e))
            raise e