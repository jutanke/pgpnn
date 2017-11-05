# [Predictive Gating Pyramid](https://papers.nips.cc/paper/5549-modeling-deep-temporal-dependencies-with-recurrent-grammar-cells)
Michalski et. al. [2] show that bi-linear models such as the factored gated autoencoder can be modelled as recurrent networks. The goal is to treat transformations as 'first-class objects' and be able to pass them around to higher layers in the network, e.g. the first layer learns 'velocity' of objects while the second layer learns 'acceleration'.

## Structure

![pgp_diagram](https://user-images.githubusercontent.com/831215/32362587-df2d50ca-c06a-11e7-9150-b5ac18d855f6.png)

## Predicting a new frame

```python
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from pgpnn import PredictiveGatingPyramid

file_name = 'mnist_test_seq.npy'
url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
if not os.path.isfile(file_name):
    print("could not find dataset: download it..")
    urllib.request.urlretrieve(url, file_name)
    print("download complete")
    
# Moving Mnist: 10.000 sequences of length 20 showing 2 digits moving in 64x64
moving_mnist = np.load(file_name) # shape: 20,10000,64,64
moving_mnist = np.rollaxis(moving_mnist, 1) # --> 10000,20,64,64

model = PredictiveGatingPyramid(depth=2, modelname='test_pgp_norm2')
model.train(
    moving_mnist,
    epochs=500,
    learningRate=0.0001,
    save_results=True,
    load_stages=True)

a = moving_mnist[0,0]
b = moving_mnist[0,1]
c = moving_mnist[0,2]
im = model.predict(a,b,c)

plt.imshow(im)
```

Using a pyramid of depth 2 we can predict the next frame given the dynamics of the scene:

![prediction_mmnist_2](https://user-images.githubusercontent.com/831215/30984602-0ce91894-a48e-11e7-9c4a-1f4fcd1e0518.png)

## Reference

[1] Memisevic, Roland. "Gradient-based learning of higher-order image features." Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011.

[2] Michalski, Vincent, Roland Memisevic, and Kishore Konda. "Modeling deep temporal dependencies with recurrent grammar cells""." Advances in neural information processing systems. 2014.
