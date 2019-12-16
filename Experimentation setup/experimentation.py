
import config

# load dataset

all_datasets = ["omniglot", "mnist", "quickdraw", "fashion"]
config.seen_datasets = ["omniglot", "quickdraw"]
config.unseen_datasets = ["mnist", "fashion"]


from optimizers import create_adam_optimizer
create_adam_optimizer(
    # lr=None, # 0.001
    # lr=0.0001, # 0.001
    lr=0.001,
    # clipnorm=config.clipnorm_outer,
    # clipvalue=1.0,
    decay=None, # 0.0
    amsgrad=None # False
)

from dataset.datasets import train_ds, test_ds
from ModelTester import ModelTester

# load trained model

# META MODEL - SOME LAYERS
from models.meta_models.MetaModelSomeLayers import MetaModelSomeLayers
from trainers.meta_trainers.LeoMetaTrainer import LeoMetaTrainer
meta_model = MetaModelSomeLayers()



"""
load weights
"""

weights_path = "/zhome/3e/7/43276/speciale/chosen experiments/Some layers/4640461 - 44 Some mnist No/models/" + "learner_model_weights.h5"



# meta_trainer = LeoMetaTrainer(model=meta_model, dataset=train_ds)
# meta_trainer.train()
# meta_tester = ModelTester(meta_trainer)
# meta_tester.visualize_adaptations()




# get output of encoder

"""
get an image from dataset
access meta_model
"""



