import numpy as np

data_path = "/zhome/3e/7/43276/speciale/data/"

import config
K = config.K
C = config.C
IMG_DIM = config.img_dim
IMG_WIDTH_AND_HEIGHT = IMG_DIM[:-1]


class Dataset():
    def __init__(self, data, C=20):
        assert len(data.shape) == 5, "wrong shape"
        self.data = data
        self.C = C
        self.batch_size = 1
        self.data_batched = None
        
    def batch_data(self, batch_size):
        """ Store a batched dataset in Dataset.data_batched, from Dataset.data.
            Drops remaining data points, if batch_size doesn't evenly divide 
            the total nr of data points.
        """
        assert len(self.data.shape) == 5, "wrong shape"
        self.batch_size = batch_size
        # drop remaining data points if necessary
        quotient = self.data.shape[0] // batch_size
        self.data_batched = self.data[: quotient * batch_size].reshape( (-1, batch_size, C) + IMG_DIM )
        
    def shuffle(self):
        """ Shuffle Dataset.data in place,
            both between tasks/classes and within.
        """
        # between tasks
        np.random.shuffle(self.data)
        # within tasks
        for i in range( len(self.data) ):
            np.random.shuffle(self.data[i])
        
    def get_flatten_dataset(self, K):
        """ Break up each task, making a flat dataset of each image - 
        Choose K examples of each character.
        Size of flatten dataset:
            N * K
            N: number of tasks
            K: number of exmaples kept from each task - maximum C
            C: number of examples in each task
            
        """
        # keep K different, randomly chosen examples from each task
        rand_idx = np.random.choice( range(C), size=K )
        dataset_reduced = self.data[: , rand_idx, :, :, :]
        dataset_reduced = dataset_reduced.reshape( (-1,) + IMG_DIM )
        return dataset_reduced

    @staticmethod
    def choose_N_images(images, N, replace=False):
        """ Choose K different random images """
        rand_idx = np.random.choice(range(len(images)), size=N, replace=replace)
        return images[rand_idx]






def load_dataset(dataset_name):
    return np.load(data_path + dataset_name)

def dataset_info(dataset):
    print(dataset.shape)
    print(dataset.dtype)
    print( np.min(dataset), np.max(dataset) )
    print( np.histogram(dataset[0,0]) )

def swap_black_and_white(img):
    return 1.0 - img

def normalize_dataset(dataset):
    ds = dataset.astype(np.float32)
    return ds / np.max(ds)

def binarize_dataset(dataset, threshold=0.5):
    return (dataset > threshold).astype(np.float32)

def digitize_dataset(dataset, nr_bins=11):
    bins = np.linspace(0, 1, nr_bins)
    digitized = np.digitize(dataset, bins)
    return np.asarray([bins[x-1] for x in digitized]).astype(np.float32)





