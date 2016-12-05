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
        answers.append([angle])
    return batch, answers


import matplotlib.pyplot as plt
frame, angle = generateBatch(1)
imgplot = plt.imshow(frame[0])
plt.show()



