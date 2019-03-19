import scipy.io as sio
import numpy
import cPickle as pickle
import gzip
import os

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

train_path = os.path.join("/home/wogong/Datasets/svhn", "train_32x32.mat")
test_path = os.path.join("/home/wogong/Datasets/svhn", "test_32x32.mat")
train = sio.loadmat(train_path)
test = sio.loadmat(test_path)

for i in range(0, 10):
    print (test['y'][i])

train_size = len(train['y'])
test_size = len(test['y'])

train_set = (numpy.zeros((train_size, 32 * 32), dtype=numpy.float32), numpy.zeros((train_size), dtype=numpy.int64))
test_set = (numpy.zeros((test_size, 32 * 32), dtype=numpy.float32), numpy.zeros((test_size), dtype=numpy.int64))

print ("begin...")

X_train_rgb = numpy.rollaxis(train['X'],3,0)
X_test_rgb = numpy.rollaxis(test['X'],3,0)

X_train = numpy.expand_dims(rgb2gray(X_train_rgb), axis=3)
X_test = numpy.expand_dims(rgb2gray(X_test_rgb), axis=3)

save = {
    'train_dataset': X_train,
    'train_labels': train['y'],
    'test_dataset': X_test,
    'test_labels': test['y']
}

f = open('svhn.pkl', 'wb')
pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

f_in = open('svhn_gray.pkl', 'rb')
f_out = gzip.open('svhn_gray.pkl.gz', 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()
