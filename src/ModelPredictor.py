import pandas as pd
import logging
import pickle



class ModelPrediction:
    def __init__ (self,X_test,model_path):
        self.X_test_path=X_test
        self.ModelPath=model_path
    


    def ModelPredict(self):
        try:
            model_clf=pickle.load(open(self.ModelPath,'rb'))
            X_test=pd.read_csv(self.X_test_path)
            y_pred=model_clf.predict(X_test)
            logging.info("Prediction completed successfully.")
            return y_pred
            
        except Exception as e:
            logging.exception("Exception occurred: %s", str(e))
            raise e