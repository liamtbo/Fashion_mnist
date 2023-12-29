# what if I organized fashion into 3 categories instead of 10
# for example : shoes, bottoms, & tops

import random
import numpy as np
from utils import mnist_reader
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')



# class Network():

#     def __init__(self, net_size: list[int])