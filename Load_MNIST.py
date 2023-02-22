
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
