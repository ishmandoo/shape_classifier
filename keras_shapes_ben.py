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
    answers_onehot = []
    for i in range(n):
        rand = random.randint(0,2)
        x = random.randint(30,70)
        y = random.randint(30,70)
        size = random.randint(5,10)
        if rand == 0:
            batch.append(drawCircle(x ,y ,size))
            answers_onehot.append([1,0,0])
            answers.append(0)
        if rand == 1:
            batch.append(drawSquare(x, y, size))
            answers_onehot.append([0,1,0])
            answers.append(1)
        if rand == 2:
            batch.append(drawTriangle(x, y, size))
            answers_onehot.append([0,0,1])
            answers.append(2)
    return np.array(batch), np.array(answers_onehot), np.array(answers)




model = Sequential()

img_width = 100
img_height = 100
model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3), dim_ordering="tf"))
first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.input

model.add(Convolution2D(8, 3, 3, border_mode='valid',       input_shape=(100, 100,3), dim_ordering="tf"))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, dim_ordering="tf"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train, Y_train, answers = generateBatch(10000)
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

X_eval, Y_eval, answers = generateBatch(10)
print model.predict_classes(X_eval)
print answers
