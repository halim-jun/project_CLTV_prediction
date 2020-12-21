import preprocess
import retrieve_data
import tensorflow
import sklearn
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

partial_x_train,partial_y_train,x_val,y_val,X_test,Y_test = preprocess.preprocess_for_model()

def model_compile(model_name, loss='mae', epoch=200, batch_size=512):
    model_name.compile(loss=loss,  metrics='mae', optimizer='adam')
    # fit the keras model on the dataset
    # Mae 로 LOSS 를 설정하지 않으면 모델이 제대로 업데이트 되지 않음

    history=model_name.fit(partial_x_train, partial_y_train, epochs=epoch, validation_data=(x_val, y_val),batch_size=batch_size)
    history_dict=history.history
    history_dict.keys()

    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']
    epochs=range(1, len(history_dict['val_loss'])+1)
    fig=plt.figure()
    plt.plot(epochs, history_dict['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')

    results = model_name.evaluate(X_test, Y_test)
    return results, fig

def model_update():
    
    model = keras.Sequential(
    [
        layers.Dense(4, activation="relu", input_dim=np.shape(partial_x_train)[1]),
        layers.Dense(8,activation="relu"),
        layers.Dense(8,activation="tanh"),
        layers.Dense(8,activation="tanh"),
        layers.Dense(1,activation="linear"),
    ]
)

    results, fig= model_compile(model, epoch=800)
    now=datetime.datetime.now()
    date_time = now.strftime("%Y_%m_%d__%H_%M_%S")
    path_to_save=os.getenv('HOME')+'/aiffel/kaggle/Lpoint/weights/'
    
    model.save(path_to_save+date_time+'_weights')
    model_path=path_to_save+date_time+'_weights'

    #preprocess 된 모델을 받아서 machine learning 을 돌리고 -> 해당 결과를 저장합니다. 
    return results, fig, model_path



