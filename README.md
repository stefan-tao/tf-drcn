# Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)

This code is an implementation of the DRCN algorithm presented in [1].

[1] M. Ghifary, W. B. Kleijn, M. Zhang, D. Balduzzi, and W. Li. ["Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)"](https://arxiv.org/abs/1607.03516), European Conference on Computer Vision (ECCV), 2016

Contact:
```
Muhammad Ghifary (mghifary@gmail.com)
```

## Requirements

* Python 2.7
* Tensorflow-1.0.1
* Keras-2.0.0
* numpy
* h5py

## Datasets

Original datasets are not provided, this repo uses datasets as follows:

- for MNIST, we use `mnist.pkl.gz` from <https://www.kaggle.com/adrienchevrier/mnist.pkl.gz> and an altered version of `load_mnist`
- for SVHN, we use `prep_data.py` to create the grayscaled SVHN dataset, `svhn_gray.pkl.gz`. You can also download the processed dataset [here](https://www.dropbox.com/s/py2zeat0tmtz6o6/svhn_gray.pkl.gz?dl=0), including only train and test set.

## Usage

To run the experiment with the (grayscaled) SVHN dataset as the source domain and the MNIST dataset as the target domain
```
python main_sm.py
```

The core algorithm is implemented in __drcn.py__.
Data augmentation and denoising strategies are included as well.

## Results

The source to target reconstruction below (SVHN as the source) indicates the successful training of DRCN.

```
python reconstruct_images.py
```

![alt text](https://github.com/ghif/drcn/blob/master/rec_src.png "Source to Target Reconstruction")

The classification accuracies of one DRCN run are plotted as follows -- the results may vary due to the randomness:

```
python plot_results.py
```

![alt text](https://github.com/ghif/drcn/blob/master/svhn-mnist_plot.png "Accuracy Plot")
