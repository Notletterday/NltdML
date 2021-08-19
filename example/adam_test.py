from tensorflow.keras.datasets import mnist
from tensorflow import keras

batch_size = 128
num_classes = 10
epochs = 2
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
'''
对y进行onehot编码
输入层：输入维度（*，784），输出维度（*，512）。
隐藏层：输入维度（*，512），输出维度（*，512）。
输出层：输入维度（*，512），输出维度（*，10）。
'''
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from DataProcessing import Optimizer

model, history = Optimizer.AdaptiveMomentEstimation(x_train, y_train,
                                                    batch_size=batch_size,
                                                    num_classes=num_classes,
                                                    epochs=epochs)
print(history.history.keys())
import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy for Adam')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for Adam')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
