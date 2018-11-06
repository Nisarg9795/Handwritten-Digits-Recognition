import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
import scipy.sparse

def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl) , 784)

    return (lbl, img)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


(label,data)=read()
(test_label,test_data)=read(dataset="testing")

def loss(w,x,y):
    m = x.shape[0]
    m1 = y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(m1), (y, np.array(range(m1)))))
    OHX = np.array(OHX.todense()).T
    scores = np.dot(x,w)
    prob = smax(scores)
    grad = (-1 / m) * np.dot(x.T,(OHX - prob))
    return grad

def smax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm


def getP(someX):
    probs = smax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return preds

w = np.zeros([data.shape[1],len(np.unique(label))])

loops = 10
rate = 1e-4
for i in range(0,loops):
    gradient = loss(w,data,label)
    w = w - (rate * gradient)


def fx_arc(tX,tY):
    prede = getP(tX)
    arc = sum(prede == tY)/(float(len(tY)))
    return arc

print ('Function accuracy obtained: ', fx_arc(test_data,test_label))