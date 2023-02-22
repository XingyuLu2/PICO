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
    
    
import requests
from os.path import exists
import os
import gzip
import numpy as np
from random import choices

#####################################
# dataset generation and processing #
#####################################

# to download the corresponding datasets from URL
def Data_Download():
    '''
    download the dataset for training and testing 
    from the corresponding URL
    '''
    # # training set images (9912422 bytes)
    train_data_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    # training set labels (28881 bytes)
    train_label_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    # test set images (1648877 bytes)
    test_data_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    # test set labels (4542 bytes)
    test_label_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    # save all the URL in one list for iterative downloading
    url_list = [train_data_url, train_label_url, test_data_url, test_label_url]
    # save the filenames of the datasets in a list
    filenames = ['training_images.gz', 'training_labels.gz', 'test_images.gz', 'test_labels.gz']
    
    print("### Start downloading the dataset ! ###")
    
    for i in range(len(url_list)):
        if not exists('datasets'):
            os.mkdir('datasets')
        # check whether the datasets exist already
        file_url = 'datasets/' + filenames[i]
        files_exist = exists(file_url)
        if files_exist == True:
            print("# the ", filenames[i], " is already existed ! ")
            continue
        # download the dataset if not existed
        print("# dowloading the ", filenames[i], " !")
        r = requests.get(url_list[i], allow_redirects = True)
        open('datasets/' + filenames[i], 'wb').write(r.content)


# add the bias term in an input vector
def AddBiasTerm(input_vector):
    '''
    add a bias term on the input vector as x <- [x, 1]
    '''
    return np.concatenate((input_vector, np.array([1])), axis=None)


# normalization the 1-D input image vector
def Z_Score_Normalize(input_vector):
    '''
    normalize the input vector using Z-score
    '''
    v_mean = np.mean(input_vector)
    v_std = np.std(input_vector)
    return (input_vector - v_mean) / v_std


#  load the training dataset and format them into the target type
def Train_Data_Format():
    '''
    load the downloaded training images & labels, and format them as :
    (1). 2D images(28 * 28) ---> one-dim vector(784 + 1 = 785) : 1-D image + bias term
    (2). labels from "0"~"9" digits to binary classification: {0, 1}
    '''
    print("### Start loading the training images and labels ! ###")
    # load the training images & labels in gzip file
    training_images = gzip.open('datasets/training_images.gz')
    training_images = training_images.read()
    training_labels = gzip.open('datasets/training_labels.gz')
    training_labels = training_labels.read()

    # to save the 1-d image vectors
    vector_list = []
    # to save the true labels
    label_list = []

    print("# Formating the images into 1-d vector ... ...")
    # Images : start from the 16-th data point, to ignore the headers 
    # : offset magic(4), num_data(4), num_row(4) and num_col(4)
    offset = 16
    num_pixels = len(training_images) - offset
    for i in range(num_pixels):
        if i % (28 * 28) == 0:
            if i != 0:
                curr_vector = np.array(curr_vector)
                # normalize the image vector and add the bias term
                curr_vector = Z_Score_Normalize(curr_vector)
                curr_vector = AddBiasTerm(curr_vector)
                # save the current formatted image vetor into a list
                vector_list.append(curr_vector)
            curr_vector = []
        # add each pixel of current image into one vector
        pixel = training_images[i + offset]
        curr_vector.append(pixel)  

    curr_vector = np.array(curr_vector)
    # normalize the image vector and add the bias term
    curr_vector = Z_Score_Normalize(curr_vector)
    curr_vector = AddBiasTerm(curr_vector)
    # save the last formatted image vetor into a list
    vector_list.append(curr_vector)

    # Labels : start from the 8-th data point, to ignore the headers 
    # : offset magic(4), num_items(4)
    offset = 8
    num_labels = len(training_labels) - offset
    for i in range(num_labels):
        true_label = training_labels[i + offset]
        label_list.append(true_label)
    
    return vector_list, label_list


# load the dataset and format them into the target type
def Test_Data_Format():
    '''
    load the downloaded test images & labels, and format them as :
    (1). 2D images(28 * 28) ---> one-dim vector(784 + 1 = 785) : 1-D image + bias term
    (2). labels from "0"~"9" digits to binary classification: {0, 1}
    '''
    print("### Start loading the Test images and labels ! ###")
    # load the test images & labels in gzip file
    test_images = gzip.open('datasets/test_images.gz')
    test_images = test_images.read()
    test_labels = gzip.open('datasets/test_labels.gz')
    test_labels = test_labels.read()

    # to save the 1-d image vectors
    vector_list = []
    # to save the true labels
    label_list = []

    print("# Formating the images into 1-d vector ... ...")
    # Images : start from the 16-th data point, to ignore the headers 
    # : offset magic(4), num_data(4), num_row(4) and num_col(4)
    offset = 16
    num_pixels = len(test_images) - offset
    for i in range(num_pixels):
        if i % (28 * 28) == 0:
            if i != 0:
                curr_vector = np.array(curr_vector)
                # normalize the image vector and add the bias term
                curr_vector = Z_Score_Normalize(curr_vector)
                curr_vector = AddBiasTerm(curr_vector)
                # save the current formatted image vetor into a list
                vector_list.append(curr_vector)
            curr_vector = []
        # add each pixel of current image into one vector
        pixel = test_images[i + offset]
        curr_vector.append(pixel)  

    curr_vector = np.array(curr_vector)
    # normalize the image vector and add the bias term
    curr_vector = Z_Score_Normalize(curr_vector)
    curr_vector = AddBiasTerm(curr_vector)
    # save the last formatted image vetor into a list
    vector_list.append(curr_vector)

    # Labels : start from the 8-th data point, to ignore the headers 
    # : offset magic(4), num_items(4)
    offset = 8
    num_labels = len(test_labels) - offset
    for i in range(num_labels):
        true_label = test_labels[i + offset]
        label_list.append(true_label)
    
    return vector_list, label_list
