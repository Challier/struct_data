from sklearn.svm import SVR
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse

from kernel import Kernel_B as Ker

SZ = 20
bin_n = 16  # Number of bins


# Extract training data : hog = histogram of gradients
# Pull all images from training folder

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

mypath_images='data/cows-images/'
onlyfiles = [f for f in listdir(mypath_images) if isfile(join(mypath_images,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
train_hog = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath_images,onlyfiles[n]) )
  train_hog[n] = hog(images[n])

# Extract training labels

mypath_labels='data/cows-labels/'
onlyfiles = [f for f in listdir(mypath_labels) if isfile(join(mypath_labels,f)) ]
train_labels = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    input_file =  open(join(mypath_labels,onlyfiles[n]))
    for line in input_file:
        if 'Bounding' in line:
            item = line[70:-2]
            values = []
            for elem in item.split('-'):
                elem = elem.strip()
                elem = elem.strip(')')
                elem = elem.strip('(')
                values.extend(elem.split(','))
    train_labels[n] = [int(x) for x in values]

# Building classifier

clf = SVR(C=1.0, epsilon=0.2)
clf.fit(np.array(train_hog, dtype=np.float64), np.array(train_labels, dtype=np.float64))


