print("hello,world!")
import tensorflow as tf
import keras
from tensorflow import keras

import numpy as np
(x_train,y_train),(x_test,y_test) = keras.datasets.boston_housing.load_data()
order = np.argsort(np.random.random(x_train.shape))
x_train = x_train[order]
y_train = y_train[order]
from DataProcessing import DataStandardization
x_train = DataStandardization.StandardScaler(x_train)[0]
x_test = DataStandardization.StandardScaler(x_test)[0]

def buildmodel():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64,activation =tf.nn.relu,input_shape=(x_train.shape[1],)))
    model.add(keras.layers.Dense(64,activation =tf.nn.relu))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.RMSprop()
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
    return model
model = buildmodel()
model.summary()




class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch%100 ==0:
            print('')
        print('.',end='')
EPOCHS = 500
history = model.fit(x_train,y_train,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[PrintDot()])
print(history.history.keys())
import matplotlib.pyplot as plt
def plot_history(history):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('Mame[1000$]')
    plt.plot(history.epoch,np.array(history.history['mae']),label='TrainLoss')
    plt.plot(history.epoch, np.array(history.history['val_mae']), label='ValLoss')
    plt.legend()
    plt.ylim([0,5])
    plt.show()
plot_history(history)
