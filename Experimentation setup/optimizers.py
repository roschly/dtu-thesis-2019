from tensorflow.keras import optimizers as op
import config


def create_sgd_optimizer(**kwargs):
    # remove kwargs with None value - in order to keep default values of optimizer
    cleaned_kwargs = {k:v for k,v in kwargs.items() if v }
    optim = op.SGD(**cleaned_kwargs) # pass options (dict) as key-value params 
    config.meta_optimizer_options = optim.get_config()
    config.meta_optimizer = optim
    config.meta_optimizer_name = "SGD"


def create_rmsprop_optimizer(**kwargs):
    # remove kwargs with None value - in order to keep default values of optimizer
    cleaned_kwargs = {k:v for k,v in kwargs.items() if v }
    optim = op.RMSprop(**cleaned_kwargs) # pass options (dict) as key-value params 
    config.meta_optimizer_options = optim.get_config()
    config.meta_optimizer = optim
    config.meta_optimizer_name = "RMSprop"

def create_adam_optimizer(**kwargs):
    # remove kwargs with None value - in order to keep default values of optimizer
    cleaned_kwargs = {k:v for k,v in kwargs.items() if v }
    optim = op.Adam(**cleaned_kwargs) # pass options (dict) as key-value params 
    config.meta_optimizer_options = optim.get_config()
    config.meta_optimizer = optim
    config.meta_optimizer_name = "Adam"






