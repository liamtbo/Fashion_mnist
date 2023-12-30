import numpy as np

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    images = images / 255
    return [images, labels]


# first we need to copy the format mnist_loader.load_data gives - done
# second we copy most of mnist_loader.load_data_wrapper
# this will give us the desired format
def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` will be after this function
    a list containing 50,000 2-tuples ``(x, y)``.  ``x`` is a 
    784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    # tr_d, va_d, te_d = load_data()
    tr_d = load_mnist('data/fashion', kind='train')
    te_d = load_mnist('data/fashion', kind='t10k')
    # tr_d[0] must contain every image, this turns each image into a 781x1 matrix
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # tr_d[1] must contains an array of keys for the images, this 
    # code make 10x1 vectors with 1 being the correct digit and 
    # 0 being not
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # connects the corresponding image with it's correct number
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


