
import config


class BaseModel:
    def __init__(self):
        self.class_name = self.__class__.__name__
        self.optimizer = config.meta_optimizer
        self.GAMMA = config.gamma
        self.loss_fn = config.loss_fn

    

    