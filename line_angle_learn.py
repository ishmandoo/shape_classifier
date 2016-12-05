import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, polygon, line_aa, line
import random
from subprocess import call
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from scipy.misc import imsave
from keras import backend as K
import tensorflow as tf
#K._LEARNING_PHASE = tf.constant(0) # test mode
from keras.models import load_model




def drawCircle(frame, x, y, r):
    rr, cc = circle(x, y, r)

    frame[rr, cc] = 255
    return frame

def drawLine(frame, x0, y0, x1, y1):
    rr, cc = circle(x, y, r)

    frame[rr, cc] = 255
    return frame

def drawAALine(frame, x, y, l, theta):
    x1 = int(x + l * np.cos(theta))
    y1 = int(x + l * np.sin(theta))
    rr, cc, val = line_aa(x, y, x1 , y1)
    frame[rr, cc,0] = val*255
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
        angle = random.random() * np.pi
        frame = np.zeros((100, 100, 3) , dtype=np.uint8)

        batch.append(drawAALine(frame, random.randint(30,70),random.randint(30,70),10, angle))
        answers.append([np.cos(angle),np.sin(angle)])
    return np.array(batch), np.array(answers)



loadfileflag = True
import os

weightfile = 'weights_line.h5'
if loadfileflag and os.path.isfile(weightfile):
    print("loaded model")
    model = load_model(weightfile)

else:

    model = Sequential()



    model.add(Convolution2D(8, 5, 5, border_mode='valid',  input_shape=(100, 100,3), dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3, dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3, dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    #model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)




X_train, Y_train = generateBatch(10000)
#print(Y_train)
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
model.save(weightfile)


X_eval, Y_eval = generateBatch(10)
#print(model.predict_classes(X_eval))
print(model.predict_proba(X_eval))
print(model.predict_proba(X_eval) - Y_eval)


print(Y_eval)


