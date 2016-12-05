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
import tensorflow as tf
#K._LEARNING_PHASE = tf.constant(0) # test mode
from keras.models import load_model
K.set_learning_phase(0)


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0)) #transposition for Theano
    x = np.clip(x, 0, 255).astype('uint8')
    return x

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
            batch.append(drawCircle(frame, random.randint(30,70),random.randint(30,70),random.randint(5,10)))
            answers.append([1,0,0])
        if rand == 1:
            batch.append(drawSquare(frame, random.randint(30,70),random.randint(30,70),random.randint(5,10)))
            answers.append([0,1,0])
        if rand == 2:
            batch.append(drawTriangle(frame, random.randint(30,70),random.randint(30,70),random.randint(5,10)))
            answers.append([0,0,1])
    return np.array(batch), np.array(answers)


import os.path
img_width = 100
img_height = 100

weightfile = 'weights.h5'
if os.path.isfile(weightfile):
    print("loaded model")
    model = load_model(weightfile)
    #print(model.layers)
    first_layer = model.layers[0]
    # this is a placeholder tensor that will contain our generated images
    #print(first_layer)
    input_img = first_layer.input
    #print(input_img)
else:

    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3), dim_ordering="tf")) #? Is this really necessary?
    first_layer = model.layers[-1] #WHHHHHHYYYYY???????
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    model.add(Convolution2D(8, 3, 3, border_mode='valid',       input_shape=(100, 100,3), dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3, dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    '''
    model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    '''
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)




X_train, Y_train = generateBatch(1000)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

model.save(weightfile)


X_eval, Y_eval = generateBatch(10)
print(model.predict_classes(X_eval))


print(np.argmax(Y_eval, axis=1))

layer_dict = dict([(layer.name, layer) for layer in model.layers])

print(layer_dict)





#layer_name = 'convolution2d_1'
layer_name = 'dense_2'
print("yo")
#K._LEARNING_PHASE = tf.constant(0) # test mode
for filter_index in range(3):  # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    print(layer_output)

    #loss = K.mean(layer_output[:, :, :, filter_index]) #for convolutional layer
    loss = K.mean(layer_output[:, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    print(grads)

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128.
    # run gradient ascent for 20 steps
    step = 1.
    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_image(img)
    imsave('filter_visual/%s_filter_%d.png' % (layer_name, filter_index), img)




