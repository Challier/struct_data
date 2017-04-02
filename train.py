import numpy as np
import h5py
from sklearn.svm import SVR
import sklearn
import matplotlib.pyplot as plt


def load_data():
    h5f = h5py.File("dataset_4.h5", 'r')

    data = np.array(h5f['data'])
    labels = np.array(h5f['labels'])

    h5f.close()

    print(np.shape(data))
    print(np.shape(labels))

    average = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    data = np.subtract(data, average)
    data = np.divide(data, std)

    x_train = data[:200]
    x_test = data[200:]
    y_train = labels[:200]
    y_test = labels[200:]

    clf = SVR(kernel='linear', C=1.0, epsilon=0.05, max_iter=5000)
    clf.fit(x_train, y_train)

    res = clf.predict(x_test)
    print(res)
    print(y_test)
    print(res - y_test)

    plt.hist(res - y_test, 50)
    plt.show()
    # print(len(clf.coef_[0]))

    h5f_c = h5py.File("coef.h5", 'w')
    h5f_c.create_dataset("coef", data=clf.coef_[0])
    h5f_c.create_dataset("avg", data=average)
    h5f_c.create_dataset("std", data=std)
    h5f_c.close()
    # print(np.dot(x_test[0], clf.coef_[0]))
    # print((np.multiply(x_test[0], clf.coef_[0])))
    # print(x_test[10:8])
    # print(np.sum((np.abs(np.multiply(x_test[0], clf.coef_[0])) + (np.multiply(x_test[0], clf.coef_[0])))/2))


if __name__ == "__main__":
    load_data()
