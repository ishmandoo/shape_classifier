import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, polygon
import random
from subprocess import call
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def drawCircle(frame, x, y, r):
    rr, cc = circle(x, y, r)

    frame[rr, cc] = 255
    return frame


def drawSquare(frame, x, y, w):
    h = w
    r = np.array([x, x+w, x+w, x, x])
    c = np.array([y, y, y+h, y+h, y])
    rr, cc = polygon(r, c)
    frame[rr, cc] = 255
    return frame

def drawTriangle(frame, x, y, b):
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
        frame = np.zeros((100, 100, 3) , dtype=np.uint8)
        if rand == 0:
            batch.append(drawCircle(frame, random.randint(30,70),random.randint(30,70),10))
            answers.append([1,0,0])
        if rand == 1:
            batch.append(drawSquare(frame, random.randint(30,70),random.randint(30,70),10))
            answers.append([0,1,0])
        if rand == 2:
            batch.append(drawTriangle(frame, random.randint(30,70),random.randint(30,70),10))
            answers.append([0,0,1])
    return np.array(batch), np.array(answers)




model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(100, 100,3), dim_ordering="tf"))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, dim_ordering="tf"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train, Y_train = generateBatch(10000)
print np.shape(X_train)
print np.shape(Y_train)
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

X_eval, Y_eval = generateBatch(10)
print model.predict_classes(X_eval)
print Y_eval
