import pandas as pd 
import os

import boto3
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

def all_data():
    path=os.getenv('HOME')+'/aiffel/kaggle/Lpoint/6th_Lpoint_competition'
    trans_original=pd.read_csv(path+'/Transaction.csv', engine='python')
    return trans_original



def download_data(path=''):
    now=datetime.datetime.now()
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    file_name=date_time+'source_file.csv'
    s3 = boto3.client('s3', aws_access_key_id='***********', 
    aws_secret_access_key='**********')
    s3.download_file('projectltv', 'Transaction.csv', file_name)
    df=pd.read_csv(path+file_name)
    return df




#model 은 Json 으로 저장하면 쉽게 올릴 수 있다.
# uploadDirectory(os.getenv('HOME')+'/aiffel/kaggle/Lpoint/weights/2020_12_17__15_17_12_weights', 'projectltv')

def upload_model_weight(weight_path,bucketname):
    s3 = boto3.client('s3', aws_access_key_id='********',
     aws_secret_access_key='***************')
    saved_model=load_model(weight_path, compile=False)
    model_to_json=saved_model.to_json()
    with open('model_to_json.json', "w") as json_file:
        json_file.write(model_to_json)
    now=datetime.datetime.now()
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    as_file_name='weight_'+date_time
    s3.upload_file('model_to_json.json',bucketname, as_file_name)
    return as_file_name

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

def download_model_weight(weight_to_download,as_file_name):
    s3 = boto3.client('s3', aws_access_key_id='*************', aws_secret_access_key='*************')
    s3.download_file('projectltv', weight_to_download, as_file_name)
    # load json and create model
    json_file = open(as_file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    return loaded_model
