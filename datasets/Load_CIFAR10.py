import os
import pickle as pickle
import numpy as np

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        print(f)
        datadict = pickle.load(f, encoding="latin1")
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    #print(Xtr.shape, Xte.shape)
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=45000, num_val=5000, num_test=10000, show_sample=True):
    """
    Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
    """

    cifar10_dir = './datasets/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # print(X_train.shape, X_test.shape)  

    # subsample the data for validation set
    X_val = X_train[num_training:num_training+num_val,:,:,:]
    y_val = y_train[num_training:num_training+num_val]
    X_train = X_train[:num_training,:,:,:]
    y_train = y_train[:num_training]
    X_test = X_test[:num_test,:,:,:]
    y_test = y_test[:num_test]

    # print(X_train.shape, X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_classes_data(classes, X_train_raw, y_train_raw, X_val_raw,y_val_raw, X_test_raw, y_test_raw):
    # Subset 'plane' and 'car' classes to perform logistic regression
    idxs = np.logical_or(y_train_raw == 0, y_train_raw == 1)
    X_train = X_train_raw[idxs, :]
    y_train = y_train_raw[idxs]
    # validation set
    idxs = np.logical_or(y_val_raw == 0, y_val_raw == 1)
    X_val = X_val_raw[idxs, :]
    y_val = y_val_raw[idxs]
    # test set
    idxs = np.logical_or(y_test_raw == 0, y_test_raw == 1)
    X_test = X_test_raw[idxs, :]
    y_test = y_test_raw[idxs]
    return X_train, y_train, X_val, y_val, X_test, y_test
    

def preprocessing_CIFAR10_data(X_train, y_train, X_val, y_val, X_test, y_test):

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train/255, (X_train.shape[0], -1)) # [49000, 3072]
    X_val = np.reshape(X_val/255, (X_val.shape[0], -1)) # [1000, 3072]
    X_test = np.reshape(X_test/255, (X_test.shape[0], -1)) # [10000, 3072]
    #print(np.max(X_train), np.min(X_train))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

    return X_train, y_train, X_val, y_val, X_test, y_test
