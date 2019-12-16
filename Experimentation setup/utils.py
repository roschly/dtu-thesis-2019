
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import numpy as np

import config

def negloglik(x, rv_x): 
    """Negative log likelihood"""
    return -rv_x.log_prob(x)


def save_fig(fig, name, ext=".png"):
    fig.savefig(config.experiment_folder_path + "images/" + name + ext)
    plt.close()


def save_model_plots(models_and_names, ext=".png"):
    for model, name in models_and_names:
        plot_model(model, to_file=config.experiment_folder_path + "models/" + name + ext, show_shapes=True)

def sample_images(images_dist):
    return images_dist.sample(), images_dist.mean(), images_dist.mode()

def save_model_weights(model, filename, ext=".h5"):
    filepath = config.experiment_folder_path + "models/" + filename + ext
    model.save_weights(filepath)

def save_losses(losses, filename):
    filepath = config.experiment_folder_path + "losses/" + filename + ".npy"
    np.save(filepath, losses)



def plot_val_loss(losses):
    fig = plt.figure(figsize=(10,5))
    plt.plot(losses, label="val losses")
    plt.legend(loc='best')
    plt.title('validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    # plt.savefig(config.experiment_folder_path + "val_loss_vs_adapt_gain.png")
    return fig

def plot_test_adapt_losses(test_adapt_losses, num_adaptations):
    fig = plt.figure(figsize=(10,5))
    for i in range(num_adaptations):
        plt.plot(test_adapt_losses[:,i], label=f"{i}. adaptation losses")

    plt.title(f"Test performance - {num_adaptations} adaptation steps")
    plt.legend(loc="best")
    plt.ylabel("Loss value")
    plt.xlabel("Test tasks")
    # plt.show()
    # plt.savefig(config.experiment_folder_path + "test_adapt_losses.png")
    return fig

def plot_losses_vs_adapt_gain(train_losses, val_losses, batch_adapt_losses, dataset):
    BATCHES_IN_EPOCH = config.batches_in_epoch
    EPOCHS = config.epochs

    batch_adapt_gains = batch_adapt_losses[:,0] - batch_adapt_losses[:,-1]
    # -1 means take the maximum number of batches in an epoch
    batches_in_epoch = BATCHES_IN_EPOCH if BATCHES_IN_EPOCH > 0 else dataset.data_batched.shape[0]
    epoch_converted_to_batch_nr = range(0, (EPOCHS+1)*batches_in_epoch, batches_in_epoch)
    fig = plt.figure(figsize=(10,5))

    plt.plot(batch_adapt_gains, label="Max adaptation gain")
    plt.plot(epoch_converted_to_batch_nr[1:], train_losses, '--', label="Training loss")
    plt.plot(epoch_converted_to_batch_nr[1:], val_losses, '--', label="Validation loss")

    plt.legend(loc='best')
    plt.xticks(ticks=epoch_converted_to_batch_nr, labels=range(EPOCHS+1))
    plt.title('Max. adaptation gain per batch during training \n vs. train/validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    # plt.savefig(config.experiment_folder_path + "val_loss_vs_adapt_gain.png")
    return fig


from functools import reduce
def chain_layer_functions(input_layer, functions):
    """ Util function for tf.keras layers - chain layer functions together.
        Layer function: layer -> layer
        reduce: input_layer, [layer_f1, layer_f2]
        --> layer_f2( layer_f1(input_layer) )
    """
    return reduce(lambda layer, func: func(layer), functions, input_layer)
