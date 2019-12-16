
# This module/file is used as a singleton
# These globals are imported in other modules
# Any modification to these variables from other modules, 
# are reflected in all other modules

jobid = None # batch job id
experiment_folder_path = ""

# Keep track of models plotted, so the learner isn't plotted several times
learner_models_plotted = False
meta_models_plotted = False
model_type = None

K = 10 # K examples from each class
C = None # total num of characters in each class
img_dim = (28,28,1) # default to MNIST dimensions



epochs = 25
batches_in_epoch = 100
batch_size = 10
num_adaptations = 5

alpha = 0.001
gamma = 0.01 # z regularizer
beta = None

dropout_rate = 0.2
learner_base_depth = None  # 16
learner_encoded_size = None # 16
meta_encoded_size = None # 64
meta_kernel_reg_l2 = None # 0.01

meta_prior_beta = 1.0

meta_optimizer = None
meta_optimizer_options = {}


loss_metric = None
loss_fn = None

learner_hidden_activation = None
meta_learner_hidden_activation = None
meta_learner_output_activation = None

clipnorm_outer = None
clipnorm_inner = None


seen_datasets = []
unseen_datasets = []
dataset_policy = None
test_size_percent = 0.1