import gzip

import cPickle as pickle
import numpy as np
import skimage
import random

def load_mnist(dataset="/home/wogong/Datasets/mnist/mnist.pkl.gz"):
    """
    Load MNIST handwritten digit images in 32x32.
    
    Return:
	(train_input, train_output), (validation_input, validation_output), (test_input, test_output)
	
	in [n, d1, d2, c] format
    """

    # Load images, in [n, c, d1, d2]
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
  
    train_set_y = train_set_y.astype('uint8')
    test_set_y = test_set_y.astype('uint8')
    valid_set_y = valid_set_y.astype('uint8')
    
    # Reshape to [n, d1, d2, c]
    c = 1
    d1 = 28
    d2 = 28
    ntrain = train_set_x.shape[0]
    nvalid = valid_set_x.shape[0]
    ntest = test_set_x.shape[0]

    train_set_x = np.reshape(train_set_x, (ntrain, d1, d2, c)).astype('float32')
    valid_set_x = np.reshape(valid_set_x, (nvalid, d1, d2, c)).astype('float32')
    test_set_x = np.reshape(test_set_x, (ntest, d1, d2, c)).astype('float32')

    new_shape = (32, 32, 1)
    train_set_x_new = np.empty(shape=(ntrain,) + new_shape)
    valid_set_x_new = np.empty(shape=(nvalid,) + new_shape)
    test_set_x_new = np.empty(shape=(ntest,) + new_shape)

    for idx in range(ntrain):
        train_set_x_new[idx] = skimage.transform.resize(train_set_x[idx], new_shape)
    for idx in range(nvalid):
        valid_set_x_new[idx] = skimage.transform.resize(valid_set_x[idx], new_shape)
    for idx in range(ntest):
        test_set_x_new[idx] = skimage.transform.resize(test_set_x[idx], new_shape)

    return (train_set_x_new, train_set_y), (valid_set_x_new, valid_set_y), (test_set_x_new, test_set_y)

def load_svhn(dataset="/home/wogong/Datasets/svhn/svhn.pkl.gz"):
    """
    Load grayscaled SVHN digit images 

    Return:
	(train_input, train_output), (test_input, test_output)
    	
	in [n, d1, d2, c] format
    """
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f)
    f.close()
   
    X_train = data['train_dataset'].astype('float32')
    y_train = data['train_labels'].astype('uint8') 
    X_test = data['test_dataset'].astype('float32')
    y_test = data['test_labels'].astype('uint8') 

    idx10 = np.where(y_train == 10)
    y_train[idx10] = 0

    idx10 = np.where(y_test == 10)
    y_test[idx10] = 0

    return (X_train, y_train), (X_test, y_test)
