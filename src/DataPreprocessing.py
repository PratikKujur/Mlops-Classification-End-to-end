import yaml
import pandas as pd
import os
import logging
from imblearn.over_sampling import SMOTE


class DataPreprocessing:
    def __init__ (self,InputPath,OutputPath,XsmPath,ysmPath):
        self.input_path=InputPath
        self.output_path=OutputPath
        self.Xsm_path=XsmPath
        self.ysm_path=ysmPath

    def savefiles(self,file,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        file.to_csv(path,index=False,header=True)

    def DataPreprocess(self):
        try:
            df=pd.read_csv(self.input_path)
            df.drop(columns=['id','Unnamed: 32'],inplace=True,axis="columns")
            df['diagnosis']=df['diagnosis'].map({'M':0,'B':1})
            self.savefiles(df,self.output_path)
            
            #feature and target split
            y=df.iloc[:,0:1]
            X=df.iloc[:,1:]

            #handling data imbalance
            class_balancer=SMOTE()
            X_sm,y_sm=class_balancer.fit_resample(X,y)
            self.savefiles(X_sm,self.Xsm_path)
            self.savefiles(y_sm,self.ysm_path)


            logging.info("Data preprocessed successfully.")

            
        except Exception as e:
            logging.exception("Exception occurred: %s", str(e))
            raise e