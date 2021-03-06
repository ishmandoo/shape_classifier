import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, polygon, line_aa
import random
from subprocess import call
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from scipy.misc import imsave
from keras import backend as K
import os

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

def drawAALine(frame, x, y, l, theta):
    x1 = int(x + l * np.cos(theta))
    y1 = int(x + l * np.sin(theta))
    rr, cc, val = line_aa(x, y, x1 , y1)
    frame[rr, cc,0] = val*255
    return frame

def generateBatch(n):
    batch = []
    answers = []
    for i in range(n):
        angle = random.random() * np.pi
        frame = np.zeros((100, 100, 3) , dtype=np.uint8)

        batch.append(drawAALine(frame, random.randint(30,70),random.randint(30,70),10, angle))
        answers.append([np.cos(angle),np.sin(angle)])
    return np.array(batch), np.array(answers)


loadFileFlag = True
saveFileFlag = True

weightFile = 'weights_ben_rotation.h5'
if loadFileFlag and os.path.isfile(weightFile):
    print("loaded model")
    model = load_model(weightFile)
else:
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

    sgd = SGD(lr=0.05, decay=0, momentum=0, nesterov=False)
    model.compile(loss='mean_absolute_error', optimizer=sgd)

for i in range(50):
    X_train, Y_train = generateBatch(10000)
    model.fit(X_train, Y_train, batch_size=1000, nb_epoch=1)
    if saveFileFlag:
        model.save(weightFile)

X_eval, Y_eval = generateBatch(10)
predictions = model.predict(X_eval)

print Y_eval
print predictions
print Y_eval-predictions
