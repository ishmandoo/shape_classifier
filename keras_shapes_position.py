import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, polygon
import random
from subprocess import call
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from scipy.misc import imsave
from keras import backend as K

def drawCircle(x, y, r):
    frame = np.zeros((100, 100, 3) , dtype=np.uint8)
    rr, cc = circle(x, y, r)
    frame[rr, cc] = 255
    return frame

def drawSquare(x, y, w):
    frame = np.zeros((100, 100, 3) , dtype=np.uint8)
    h = w
    r = np.array([x, x+w, x+w, x, x])
    c = np.array([y, y, y+h, y+h, y])
    rr, cc = polygon(r, c)
    frame[rr, cc] = 255
    return frame

def drawTriangle(x, y, b):
    frame = np.zeros((100, 100, 3) , dtype=np.uint8)
    r = np.array([x, x+b, x, x])
    c = np.array([y, y, y+b, y])
    rr, cc = polygon(r, c)
    frame[rr, cc] = 255
    return frame

def generateBatch(n):
    batch = []
    answers = []
    for i in range(n):
        rand = random.randint(0,2)
        x = random.randint(20,80)
        y = random.randint(20,80)
        size = random.randint(5,10)
        if rand == 0:
            batch.append(drawCircle(x ,y ,size/2))
            answers.append([x/100., y/100.])
        if rand == 1:
            batch.append(drawSquare(x, y, size))
            answers.append([x/100., y/100.])
        if rand == 2:
            batch.append(drawTriangle(x, y, size))
            answers.append([x/100., y/100.])
    return np.array(batch), np.array(answers)




model = Sequential()

img_width = 100
img_height = 100

model.add(Convolution2D(8, 3, 3, border_mode='valid', input_shape=(100, 100,3), dim_ordering="tf"))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, dim_ordering="tf"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(2))
#model.add(Activation('linear'))

sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
model.compile(loss='mean_absolute_error', optimizer=sgd)

for i in range(10):
    X_train, Y_train = generateBatch(10000)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

X_eval, Y_eval = generateBatch(10)
print model.predict(X_eval)
print Y_eval

print model.layers[-1].get_weights()
