
############################
##### IMPORTS
############################

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import time
import itertools
import math

speciale_path = "/zhome/3e/7/43276/speciale/"






















############################
##### ARGUMENTS
############################

# from arg_parser import arg_parser
# jobid = arg_parser()

import argparse
ap = argparse.ArgumentParser()
# ap.add_argument("-j", "--jobid", required=True)
ap.add_argument("-j", "--jobid", required=False)
ap.add_argument("--epochs", type=int, required=False)
ap.add_argument("--beta", type=float, required=False)
ap.add_argument("--model", type=str, required=True)
ap.add_argument("--seen-datasets", nargs='+', required=False)
args = vars(ap.parse_args())

print(args)

if args["jobid"]:
    jobid = args["jobid"]
else:
    from datetime import datetime
    now = datetime.now()
    jobid = now.strftime("%Y-%m-%d_%H:%M:%S.%f")



experiment_folder_path = speciale_path + "experiments/" + jobid + "/"

try:
    os.makedirs(experiment_folder_path)
except:
    OSError("Failed to create experiment folder")

try:
    os.makedirs(experiment_folder_path + "models/")
    os.makedirs(experiment_folder_path + "losses/")
    os.makedirs(experiment_folder_path + "images/")
except:
    OSError("Failed to create folders in experiment folder")

























############################
##### CONFIG
############################


import config

config.experiment_folder_path = experiment_folder_path



config.epochs = 75 #75    
config.batches_in_epoch = 15 #20
config.batch_size = 10 #12
config.num_adaptations = 3

config.C = 20
config.K = 10
config.jobid = jobid
config.img_dim = (28,28,1)

config.alpha = 0.01 # 0.001 - 0.0001 with no clip norm/value
config.gamma = 0.1

config.dropout_rate = 0.1

config.meta_prior_beta = 1.0 # set ABOVE 1.0 for beta-vae

if args["epochs"]:
    config.epochs = args["epochs"]
if args["beta"]:
    config.meta_prior_beta = args["beta"]

# config.meta_learner_hidden_activation = tf.nn.leaky_relu
# config.meta_learner_hidden_activation = tf.nn.relu
config.meta_learner_hidden_activation = tf.nn.tanh

config.meta_learner_output_activation = tf.nn.tanh


config.learner_hidden_activation = tf.nn.leaky_relu
# config.learner_hidden_activation = tf.nn.tanh

config.clipnorm_outer = 1.0
config.clipnorm_inner = 10.0


from utils import negloglik
# config.loss_metric = "bce_loss"
# config.loss_metric = "negloglik"
config.loss_fn = negloglik


config.learner_base_depth = 16
config.learner_encoded_size = 16
config.meta_encoded_size = 32 #8 # 32

config.meta_kernel_reg_l2 = 0.01


config.dataset_policy = "fake balanced"

# SEEN
# config.seen_datasets = ["omniglot", "quickdraw"]
# config.seen_datasets = ["fashion", "quickdraw"]
# config.seen_datasets = ["quickdraw"]
# config.seen_datasets = ["omniglot", "mnist"]
# config.seen_datasets = ["omniglot"]
# config.seen_datasets = ["mnist"]

# UNSEEN
# config.unseen_datasets = ["mnist", "fashion"]
# config.unseen_datasets = ["quickdraw"]
# config.unseen_datasets = ["quickdraw", "omniglot"]
# config.unseen_datasets = ["quickdraw", "mnist"]
# config.unseen_datasets = ["omniglot"]

all_datasets = ["omniglot", "mnist", "quickdraw", "fashion"]

# baseline combinations
# config.seen_datasets = ["fashion"]
# config.seen_datasets = ["mnist"]
# config.seen_datasets = ["omniglot"]
# config.seen_datasets = ["quickdraw"]
# config.seen_datasets = ["mnist", "fashion"]
# config.seen_datasets = ["mnist", "omniglot"]
# config.seen_datasets = ["mnist", "quickdraw"]
# config.seen_datasets = ["fashion", "omniglot"]
# config.seen_datasets = ["fashion", "quickdraw"]
# config.seen_datasets = ["omniglot", "quickdraw"]

if args["seen_datasets"]:
    config.seen_datasets = args["seen_datasets"]
    print(args["seen_datasets"])

assert config.seen_datasets != [], "No seen datasets!"

config.unseen_datasets = [ d for d in all_datasets if d not in config.seen_datasets ]



from optimizers import create_sgd_optimizer, create_rmsprop_optimizer, create_adam_optimizer
# None values are ignored when creating optimizer

create_adam_optimizer(
    # lr=None, # 0.001
    # lr=0.0001, # 0.001
    lr=0.001,
    clipnorm=config.clipnorm_outer,
    # clipvalue=1.0,
    decay=None, # 0.0
    amsgrad=None # False
)


















############################
##### SETUP, TRAINING, EVAL
############################

assert args["model"] in ["some", "all"], "must specify valid model"
config.model_type = args["model"]

from config_writer import ConfigWriter
ConfigWriter.write_to_file()

from ModelTester import ModelTester
from dataset.datasets import train_ds, test_ds



# check GPU availability
# gpu_checks = [tf.test.is_built_with_cuda(), tf.test.is_built_with_cuda(), tf.test.is_built_with_gpu_support()]
# print(gpu_checks)


# from models.learner_models.learner_model_vae_dist import LearnerModelVaeDist
# from trainers.learner_trainers.no_task_learner_trainer import NoTaskLearnerTrainer
# from dataset.datasets import mnist

# # flatten mnist from: (10,6000,28,28) -> (60000,28,28,1)
# mnist_flatten = np.expand_dims(mnist, axis=-1)
# mnist_flatten = np.reshape(mnist_flatten, (-1,28,28,1) )
# print(mnist_flatten.shape)

# pretrain_model = LearnerModelVaeDist()
# pretrain_trainer = NoTaskLearnerTrainer(model=pretrain_model, dataset=mnist_flatten)
# pretrain_trainer.train()
# # baseline_tester = ModelTester(baseline_trainer)
# # baseline_tester.test()
# # baseline_tester.visualize_adaptations()


# NO TASK TRAINER
# from models.learner_models.LearnerModelVaeDist import LearnerModelVaeDist
# from trainers.learner_trainers.NoTaskLearnerTrainer import NoTaskLearnerTrainer
# baseline_model = LearnerModelVaeDist()
# baseline_trainer = NoTaskLearnerTrainer(model=baseline_model, dataset=train_ds)
# baseline_trainer.train()
# baseline_tester = ModelTester(baseline_trainer)
# # # baseline_tester.test()
# baseline_tester.visualize_adaptations()



if args["model"] == "some":
    # META MODEL - SOME LAYERS
    from models.meta_models.MetaModelSomeLayers import MetaModelSomeLayers
    from trainers.meta_trainers.LeoMetaTrainer import LeoMetaTrainer
    meta_model = MetaModelSomeLayers()
    meta_trainer = LeoMetaTrainer(model=meta_model, dataset=train_ds)
    # meta_trainer.model.apply_pretrained_weights(baseline_model)
    meta_trainer.train()
    meta_tester = ModelTester(meta_trainer)
    # meta_tester.test()
    meta_tester.visualize_adaptations()

if args["model"] == "all":
    # META MODEL - ALL LAYERS
    from models.meta_models.MetaModelAllLayers import MetaModelAllLayers
    from trainers.meta_trainers.LeoMetaTrainer import LeoMetaTrainer
    meta_model = MetaModelAllLayers()
    meta_trainer = LeoMetaTrainer(model=meta_model, dataset=train_ds)
    meta_trainer.train()
    meta_tester = ModelTester(meta_trainer)
    # meta_tester.test()
    meta_tester.visualize_adaptations()














