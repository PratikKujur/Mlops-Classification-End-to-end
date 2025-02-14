
import yaml
import pandas as pd
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))['train']
params_eval=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))['evaluation']
params_model=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-Classification-End-to-end/params.yaml"))


class ModelTrainer:
    def __init__ (self,X,Y,X_train,y_train,X_test,y_test,model_path):
        self.X=X
        self.y=Y
        self.X_train_path=X_train
        self.y_train_path=y_train
        self.y_test_path=y_test
        self.X_test_path=X_test
        self.ModelPath=model_path
        

    def savefiles(self,file,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        file.to_csv(path,index=False,header=True)

    def model_trainer(self,X,y,ModelPath):
        clf=RandomForestClassifier(criterion='entropy',max_features='sqrt')
        clf.fit(X,y)
        os.makedirs(os.path.dirname(ModelPath),exist_ok=True)
        pickle.dump(clf,open(ModelPath,'wb'))


    def StartTraining(self):
        try:
            X=pd.read_csv(self.X)
            y=pd.read_csv(self.y)

            X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42,test_size=0.2)

            saving_files=[
                           (X_train,self.X_train_path),
                           (X_test,self.y_train_path), 
                           (y_train,self.y_test_path), 
                           (y_test,self.X_test_path)
                        ]
            for file,path in (saving_files):
                self.savefiles(file,path)
            logging.info("Data Splited and Saved successfully.")

            self.model_trainer(X_train,y_train,self.ModelPath)
            logging.info("Model Trained and Saved successfully.")




            
        except Exception as e:
            logging.exception("Exception occurred: %s", str(e))
            raise e
        

