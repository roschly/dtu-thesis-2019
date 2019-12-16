import numpy as np

import config
from dataset.dataset_utils import load_dataset, normalize_dataset, binarize_dataset
from dataset.dataset_utils import Dataset

K = config.K
seen_ds = config.seen_datasets
unseen_ds = config.unseen_datasets
policy = config.dataset_policy
test_size_percent = config.test_size_percent


def transform_to_fake_tasked_datasets(ds):
    ds_tasked = ds.reshape( (ds.shape[0], -1, 20, ds.shape[-2], ds.shape[-1] ) )
    tasked_datasets = []
    for image_class in ds_tasked:
        tasked_datasets.append(image_class)
    return tasked_datasets

def get_dataset(dataset_name, policy):
    if policy == "unbalanced":
        if dataset_name == "mnist":
            return [mnist]
        if dataset_name == "fashion":
            return [fashion]
    elif policy == "fake balanced":
        if dataset_name == "mnist":
            return transform_to_fake_tasked_datasets(mnist)
        if dataset_name == "fashion":
            return transform_to_fake_tasked_datasets(fashion)
    else:
        raise ValueError("Choose a policy!")


def get_seen_and_unseen_datasets(seen_ds, unseen_ds, policy):
    """Return actual datasets from names in seen and unseen ds, depending on policy"""

    all_datasets = {
        "omniglot": [omni],
        "quickdraw": [quickdraw],
        "mnist": get_dataset("mnist", policy),
        "fashion": get_dataset("fashion", policy)
    }

    seen_datasets = []
    for ds_name in seen_ds:
        for ds in all_datasets[ds_name]:
            seen_datasets.append( ds )
    unseen_datasets = []
    for ds_name in unseen_ds:
        for ds in all_datasets[ds_name]:
            unseen_datasets.append( ds )
    return seen_datasets, unseen_datasets


def get_subset_from_datasets(datasets, mode):
    """Choose a random subset from each class/tasks in entire dataset"""
    assert mode == "train" or mode == "test", "Mode has to be either 'train' or 'test'"
    if mode == "train":
        K_ = K*2 # Note: K*2 exampls (train + val)
    elif mode == "test":
        K_ = K # Note: only K examples
        
    # choose K_ subset from all classes
    subset_datasets = []
    for ds in datasets:
        rand_images_idx = np.random.choice( range(ds.shape[1]), size=K_, replace=False )
        subset = ds[:,rand_images_idx]
        subset_datasets.append( subset )
    return subset_datasets

def get_test_set_splits(datasets):
    """Split datasets into train and test sets"""
    # choose test set from each subset
    train_sets = []
    test_sets = []
    for ds in datasets:
        test_size = int(ds.shape[0] * test_size_percent)
        train_sets.append(ds[test_size : ])
        test_sets.append( ds[ : test_size])
    return train_sets, test_sets



# LOAD DATASETS

quickdraw = load_dataset("quick_draw_1000.npy")
quickdraw = normalize_dataset(quickdraw)
quickdraw = binarize_dataset(quickdraw)
# quickdraw = digitize_dataset(quickdraw)
# dataset_info(quickdraw)

omni = load_dataset("omniglot.npy")
omni = normalize_dataset(omni)
omni = binarize_dataset(omni)
# omni = digitize_dataset(omni)
# dataset_info(omni)

fashion = load_dataset("fashion_mnist.npy")
fashion = normalize_dataset(fashion)
fashion = binarize_dataset(fashion, threshold=0.1)
# fashion = digitize_dataset(fashion)
# dataset_info(fashion)

mnist = load_dataset("mnist.npy")
mnist = normalize_dataset(mnist)
mnist = binarize_dataset(mnist)
# mnist = digitize_dataset(mnist)
# dataset_info(mnist)



# ===============================

seen_datasets, unseen_datasets = get_seen_and_unseen_datasets(seen_ds, unseen_ds, policy="fake balanced")

# unbalanced (just add all)
# split seen datasets into train and test

assert K*2 <= 20, "K cannot be more than half of 20 (max nr of images in some dataset classes)"

seen_subset_datasets = get_subset_from_datasets(seen_datasets, mode="train")
unseen_subset_datasets = get_subset_from_datasets(unseen_datasets, mode="test")

seen_train_sets, seen_test_sets = get_test_set_splits(seen_subset_datasets)
_, unseen_test_sets = get_test_set_splits(unseen_subset_datasets)

# concatenate subsets
combined_train_dataset = np.concatenate( seen_train_sets )
combined_seen_test_dataset = np.concatenate( seen_test_sets )
combined_unseen_test_dataset = np.concatenate( unseen_test_sets )


print("Seen datasets:", seen_ds)
print("Unseen datasets:", unseen_ds)

print("Full train set:", combined_train_dataset.shape)
print("Full seen test set:", combined_seen_test_dataset.shape)
print("Nr seen test sets:", len(seen_test_sets))
print("Seen test set shapes:")
for t in seen_test_sets:
    print(t.shape)

print("Nr unseen test sets:", len(unseen_test_sets))
print("Unseen test set shapes:")
for t in unseen_test_sets:
    print(t.shape)

# Exported to other scripts

# train_ds = combined_train_dataset
# test_ds = np.asarray(seen_test_sets)
# seen_test_datasets = seen_test_sets
# unseen_test_datasets = unseen_test_sets


train_ds = Dataset( np.expand_dims(combined_train_dataset,axis=-1) )
test_ds  = Dataset( np.expand_dims(combined_seen_test_dataset,axis=-1) )
unseen_test_ds  = Dataset( np.expand_dims(combined_unseen_test_dataset,axis=-1) )