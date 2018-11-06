import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from scipy.stats import mode
from scipy.spatial import distance

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
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), 784)

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


k=[1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
a=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


distances=(distance.cdist(test_data[:10000],data))

s=np.argsort(distances)
sd=s[:,:10000]
for i in range(0,10000):
    vector=s[i,:]
    l=0
    for j in k:
        neigh = label[vector[:j]]
        if test_label[i]==mode(neigh)[0][0]:
            a[l]=a[l]+1
        l=l+1

divide = 100
percentage = [o / divide for o in a]
print(percentage)
