# imports
import numpy as np
import createData
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json


# global variables
imgx = 36
imgy = 36
model_file_name = "model.json"
weights_file_name = "weights.h5"
label_consonents = ['ka', 'kha', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', 'yna', 'ta', 'tha', 'da'
                    , 'dha', 'ana', 'taa', 'thaa', 'daa', 'dhaa', 'na', 'pa', 'pha', 'ba', 'dha', 'ma'
                    , 'ya', 'ra', 'la', 'va', 'motosaw', 'petchiryosaw', 'patalosaw', 'ha', 'ksha', 'tra', 'gya']
label_vowels = ['a', 'aa', 'i', 'ee', 'u', 'oo', 'ae', 'ai', 'o', 'au', 'an', 'ah']
label_numerals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# creating the convolution network model
def train():
    X_train, y_train, X_test, y_test = createData.main()
    X_train, y_train, input_shape = preprocess(X_train, X_test, y_train, y_test)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(58, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)
    save_model(model)

# preprocessing the training data
def preprocess(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(X_train.shape[0], imgx, imgy, 1)
    X_test = X_test.reshape(X_test.shape[0], imgx, imgy, 1)
    input_shape = (imgx, imgy, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return np.append(X_train, X_test, axis=0), np.append(y_train, y_test, axis=0), input_shape

# saving cnn model and weights
def save_model(model):
	model_json = model.to_json()
	with open(model_file_name, "w+") as json_file:
	    json_file.write(model_json)
	model.save_weights(weights_file_name)

# load cnn model and weights
def load_model():
	json_file = open(model_file_name, 'r+')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights_file_name)
	return loaded_model

# predicting output label
def predict(model, X_test):
    y_test = int(model.predict_classes(X_test, steps=15)[0])
    labels = label_consonents + label_vowels + label_numerals
    output = ""
    if y_test < 36:
        output = "Consonant: "
    elif y_test < 48:
        output = "Vowel: "
    else:
        output = "Numeral: "
    return output + labels[y_test]

# ensuring call from same module
if __name__ == '__main__':
    train()
