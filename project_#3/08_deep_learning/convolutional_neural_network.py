from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling3D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the Convolutional neural network (CNN)
classifier = Sequential()

# Convolution layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))