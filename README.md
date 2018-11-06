# Handwritten-Digits-Recognition
Implemented logistic regression model and k-Nearest Neighbor (kNN) algorithm for MNIST handwritten digits recognition. The MNIST dataset contains 70,000 images of handwritten digits, out of which 60,000 images are training set and 10,000 images are test set. Each image has 28 × 28 pixels which are considered as the features of the image. The code uses the training set and then uses the results to predict the digits in the test set.
- For Logistic regression, a multi-class model was implemented and a gradient ascent algorithm was was used to train the classifier. No regularization term was taken into consideration and tolerance was set to e^-4.
- For kNN, Euclidean distance was used to measure the distance between the data points. The number of neighbors to be taken into account were set variable and later the accuracies were plotted to see th trend.


PyCharm 2018.2.4 was used as the IDE.
The libraries used were numpy, scipy and matplotlib
