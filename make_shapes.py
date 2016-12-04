import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, polygon
from PIL import Image
import random
from subprocess import call

n = 1000

myfile = open("label.txt", "w")



frame = np.zeros((100, 100, 3) , dtype=np.uint8)

def drawCircle(frame, x, y, r):
    rr, cc = circle(x, y, r)

    frame[rr, cc] = 255


def drawSquare(frame, x, y, w):
    h = w
    r = np.array([x, x+w, x+w, x, x])
    c = np.array([y, y, y+h, y+h, y])
    rr, cc = polygon(r, c)
    frame[rr, cc] = 255

def drawTriangle(frame, x, y, b):
    r = np.array([x, x+b, x, x])
    c = np.array([y, y, y+b, y])
    rr, cc = polygon(r, c)
    frame[rr, cc] = 255

for i in range(n):
    drawCircle(frame, random.randint(30,70),random.randint(30,70),10)

    im = Image.fromarray(frame)
    im.save("images/"+str(i)+".png")
    myfile.write(str(i) + ".png " + str(0) + "\n")
    frame[:] = 0

for i in range(n,2*n):
    drawSquare(frame, random.randint(30,70),random.randint(30,70),10)

    im = Image.fromarray(frame)
    im.save("images/"+str(i)+".png")
    myfile.write(str(i) + ".png " + str(1) + "\n")
    frame[:] = 0

for i in range(2*n,3*n):
    drawTriangle(frame, random.randint(30,70),random.randint(30,70),10)

    im = Image.fromarray(frame)
    im.save("images/"+str(i)+".png")
    myfile.write(str(i) + ".png " + str(2) + "\n")
    frame[:] = 0

myfile.close()
call(["../caffe/build/tools/convert_imageset", "images/", "label.txt", "data"])
