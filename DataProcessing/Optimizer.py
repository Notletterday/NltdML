import keras

__all__ = ['StochasticGradientDescent', 'AdaptiveMomentEstimation']

from tensorflow import keras
from keras.layers import Dense, Dropout


def StochasticGradientDescent(x, y, batch_size, num_classes, epochs):
    """

    :param x:
    :param y:
    :param batch_size:
    :param num_classes:
    :param epochs:
    :return:
    """
    model = keras.models.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x, y))
    return model, history


def AdaptiveMomentEstimation(x, y, batch_size, num_classes, epochs):
    """

    :param x:
    :param y:
    :param batch_size:
    :param num_classes:
    :param epochs:
    :return:
    """
    model = keras.models.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x, y))
    return model, history
def AdaptiveLearningRateAdjustment(x, y, batch_size, num_classes, epochs):
    model = keras.models.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    adadelta = keras.optimizers.Adadelta(lr = 1.0,rho = 0.95,epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x, y))
    return model, history
def RootMeanSquareProp(x, y, batch_size, num_classes, epochs):
    model = keras.models.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x, y))
    return model, history

