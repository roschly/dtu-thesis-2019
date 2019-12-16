
import tensorflow as tf

from trainers.BaseTrainer import BaseTrainer


import utils
import config



class NoTaskLearnerTrainer(BaseTrainer):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.train_dataset = dataset
        self.Optimizer = tf.keras.optimizers.Adam


    # For a VAE
    def record_adaptations(self, images):
        learner_prior_copy, _, learner_decoder_copy, learner_model_copy = self.model.copy_learner()
        learner_model_copy.compile(
            optimizer=self.Optimizer(),
            loss=self.loss_fn
        )
        reconstructions = []
        # before any adaptation
        images_dist = learner_model_copy( images )
        _, images_mean, _ = utils.sample_images(images_dist)
        reconstructions.append(images_mean)
        # begin adaptations
        for _ in range(self.NUM_ADAPTATIONS):
            learner_model_copy.fit(x=images, y=images, batch_size=len(images), epochs=1, verbose=0)
            images_dist = learner_model_copy(images)
            _, images_mean, _ = utils.sample_images(images_dist)
            reconstructions.append(images_mean)
        
        # generate new images from adapted model
        gen_sample, gen_mean, gen_mode = self.generate_new_samples(prior=learner_prior_copy, decoder=learner_decoder_copy)
        new_images = [gen_sample, gen_mean, gen_mode]        
        return reconstructions, new_images


    def train(self):
        epoch_val_losses, _ = self.training()
        figure = utils.plot_val_loss(epoch_val_losses)
        utils.save_fig(fig=figure, name=f"{self.model.class_name}-val_loss")


    def training(self):
        self.train_dataset.shuffle()

        # flatten data
        train_flatten = self.train_dataset.data[:,:self.K].reshape( (-1,) + self.IMG_DIM )
        print("train_flatten shape", train_flatten.shape)
        val_flatten = self.train_dataset.data[:,self.K:self.K*2].reshape( (-1,) + self.IMG_DIM )

        self.model.learner_model.compile(
            optimizer=self.Optimizer(),
            loss=self.loss_fn
            )
        # self.BATCH_SIZE is for batches of tasks, each containing K images
        # since there are no tasks here, multiply by K
        no_tasks_batch_size = self.BATCH_SIZE * self.K
        hist = self.model.learner_model.fit(
            x=train_flatten,
            y=train_flatten,
            validation_data=(val_flatten, val_flatten),
            shuffle=True,
            batch_size=no_tasks_batch_size,
            epochs=self.EPOCHS
        )
        epoch_val_losses = hist.history["val_loss"]
        # no adaptation losses
        return epoch_val_losses, None


    

