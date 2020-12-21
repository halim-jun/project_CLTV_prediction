import pandas as pd 
import retrieve_data
import os
import keras
import preprocess
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime
def predict_save_result():
    best_weights=os.getenv('HOME')+'/aiffel/kaggle/Lpoint/weights/2020_12_17__15_17_12_weights'
    saved_model=keras.models.load_model(best_weights)
    data=preprocess.common_preprocess()
    data_X=data.drop(['clnt_id', 'sep_sum_price'],axis=1)
    scaler = MinMaxScaler()
    data_X=pd.DataFrame(scaler.fit_transform(data_X))
    y_pred=saved_model.predict(data_X)
    def to_origin(data):
        return np.exp(data)-1
    y_pred_origin=to_origin(y_pred)
    result=pd.DataFrame({'clnt_id':data['clnt_id'],'prediction': y_pred_origin.flatten()})
    result.loc[result['prediction']<100000, 'predicted_level']=0
    result.loc[result['prediction']>=100000, 'predicted_level']=1
    result.loc[result['prediction']>=250000, 'predicted_level']=2
    result.loc[result['prediction']>=500000, 'predicted_level']=3
    result.loc[result['prediction']>=800000, 'predicted_level']=4
    result.loc[result['prediction']>=1500000, 'predicted_level']=5
    path=os.getenv('HOME')+'/aiffel/kaggle/Lpoint/result/'
    now=datetime.datetime.now()
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    result.to_csv(path+date_time+'_restult.csv')
    return result

def get_predicted_result(query_clnt_id):
    path=os.getenv('HOME')+'/aiffel/kaggle/Lpoint/result/'
    result=pd.read_csv(path+'2020_12_17__17_59_05_restult.csv')
    predicted_purcahse=result.loc[result['clnt_id']==query_clnt_id, 'prediction'][0]
    return predicted_purcahse

