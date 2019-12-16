import time

import numpy as np
import tensorflow as tf

import config
import utils
# from dataset.datasets import train_ds
from models.meta_models.MetaModelAllLayers import MetaModelAllLayers
from models.meta_models.MetaModelBase import MetaModelBase
from trainers.BaseTrainer import BaseTrainer

class LeoMetaTrainer(BaseTrainer):
    def __init__(self, model:MetaModelBase, dataset):
        super().__init__()
        self.model = model
        self.train_dataset = dataset


    @staticmethod
    def stack_images(images):
        # massage images to fit into latent model
        stacked_images = np.squeeze(images, axis=-1)
        stacked_images = np.stack(stacked_images, axis=-1)
        stacked_images = np.expand_dims(stacked_images, axis=0)
        return stacked_images
        # return np.expand_dims(images, axis=0)


    def record_adaptations(self, images):
        reconstructions = []

        # perform adaptations and recover params weights
        stacked_images = self.stack_images(images)
        _, ws, _ = self.inner_loop(stacked_images=stacked_images, images=images)
        # apply weights to learner model (includes pre-adaptation weights)
        for w in ws:
            self.model.apply_weights_from_mapping(w)
            images_dist = self.model.learner_model(images)
            _, images_mean, _ = utils.sample_images(images_dist)
            reconstructions.append(images_mean)

        # generate new images from adapted model
        # TODO: is this model adapted at all??
        # it should be, since the loop above applies the last weights
        self.model.apply_weights_from_mapping(ws[-1]) # redundant?
        gen_sample, gen_mean, gen_mode = self.generate_new_samples(prior=self.model.learner_prior, decoder=self.model.learner_decoder)
        new_images = [gen_sample, gen_mean, gen_mode]        
        
        # how likely are the means from each dist
        # gen_in_rec = images_dist.log_prob(gen_mean)
        # print("Gen mean likelihood in images_dist", gen_in_rec)

        return reconstructions, new_images

    def adapt_to_task(self, images):
        stacked_images = self.stack_images(images)
        _, _, adapt_losses = self.inner_loop(stacked_images=stacked_images, images=images)
        np_adapt_losses = np.asarray(adapt_losses)
        return np.mean(np_adapt_losses, axis=1)


    def save_weights_and_losses(self, train_losses, val_losses, batch_adapt_losses):
        # weights
        utils.save_model_weights(self.model.meta_model, "meta_model_weights")
        utils.save_model_weights(self.model.learner_model, "learner_model_weights")
        # losses
        utils.save_losses(train_losses, "train_losses")
        utils.save_losses(val_losses, "val_losses")
        utils.save_losses(batch_adapt_losses, "batch_adapt_losses")


    def train(self):
        train_losses, val_losses, batch_adapt_losses = self.training()
        # save
        figure = utils.plot_losses_vs_adapt_gain(train_losses, val_losses, batch_adapt_losses, self.train_dataset)
        utils.save_fig(fig=figure, name="losses_vs_adapt_gain")
        self.save_weights_and_losses(train_losses, val_losses, batch_adapt_losses)
        
        


    def training(self):
        epoch_train_losses = []
        epoch_val_losses = []

        batch_adapt_losses = []
        start_time = time.time()
        print("TRAINING")
        print("epochs -- avg epoch loss -- epoch execution loss")
        for e in range(self.EPOCHS):
            # reset losses in new epoch
            batch_losses = []
            self.train_dataset.shuffle()
            self.train_dataset.batch_data(self.BATCH_SIZE)
            
            for batch in self.train_dataset.data_batched[ : self.BATCHES_IN_EPOCH]:
                batch_loss, batch_adapt_loss = self.batch_training_step(batch=batch)
                batch_losses.append(batch_loss)
                batch_adapt_losses.append( batch_adapt_loss )

            
            avg_train_batch_loss = np.mean( np.asarray(batch_adapt_losses)[:,-1] ) # all values of the final adaptation step
            avg_val_batch_loss = np.mean(batch_losses)
            epoch_train_losses.append(avg_train_batch_loss)
            epoch_val_losses.append(avg_val_batch_loss)

            time_spent = time.time() - start_time
            # print and write to file, the training progress
            progress = f"{e+1}/{self.EPOCHS} - {avg_train_batch_loss:.2f} - {avg_val_batch_loss:.2f} - {time_spent:.2f}s"
            print(progress)
            # write training history to file continuously (accept overhead of freq. open/close)
            with open(config.experiment_folder_path + "train_history.txt", "a") as f:
                f.write(progress + "\n")
            start_time = time.time()
        
        # TODO: implement a way to only save the best weights
        # full_model.save_weights(config.experiment_folder_path + "last_weights.h5")

        return np.asarray(epoch_train_losses), np.asarray(epoch_val_losses), np.asarray(batch_adapt_losses)


    def batch_training_step(self, batch):
        K = self.K
        task_adapt_losses = []
        with tf.GradientTape(persistent=True) as meta_tape:
            # meta_tape.watch(self.ALPHA)
            meta_losses = []
            for task in batch:
                images_train = task[:K] # first K images
                stacked_images = self.stack_images(images_train) # input for latent encoder
                zs, ws, adapt_losses = self.inner_loop(stacked_images=stacked_images, images=images_train)
                task_adapt_losses.append(adapt_losses)

                images_val = task[K:K*2] # next K images
                meta_loss = self.model.compute_meta_loss(images_val, ws[-1], zs[0], zs[-1])
                meta_losses.append(meta_loss)

            batch_loss = tf.reduce_mean(meta_losses)
        # update meta learner
        self.model.meta_learning_step(meta_tape, batch_loss)
        # self.ALPHA.assign_sub( meta_tape.gradient(batch_loss, self.ALPHA) ) 
        # print(self.ALPHA)
        
        # TODO: problem - the reported train loss is the AVERAGE of all adaptation steps, but it should be the last adaptation loss
        batch_adapt_loss = np.asarray(task_adapt_losses)
        batch_adapt_loss = np.mean( batch_adapt_loss, axis=-1 ) # mean over images
        batch_adapt_loss = np.mean( batch_adapt_loss, axis=0 ) # mean over tasks --> avg loss for each adaptation step

        return batch_loss.numpy(), batch_adapt_loss


    def inner_loop(self, stacked_images, images):
        adapt_losses = []
        zs = []
        ws = []
        with tf.GradientTape(persistent=True) as tape:
            z = self.model.meta_encoder(stacked_images)
            # z_dist = self.model.meta_encoder(stacked_images)
            # z = z_dist.sample()
            w = self.model.meta_decoder(z)
            # before adaptation weights and encoding
            zs.append(z)
            ws.append(w)
            # adaptations
            for _ in range(self.NUM_ADAPTATIONS):
                loss = self.model.compute_loss(images, w)
                grads = tape.gradient(loss, z)
                # TODO: try value clipping instead of norm
                grads_clipped, global_norm = tf.clip_by_global_norm([grads], clip_norm=config.clipnorm_inner)
                # grads_clipped = tf.clip_by_value(grads[0], -10.0, 10.0)
                # print("inner clipped", np.max(grads_clipped))

                # update z and w
                z -= self.ALPHA * grads_clipped[0]
                # z -= self.ALPHA * grads_clipped
                # z -= self.ALPHA * grads
                w = self.model.meta_decoder(z)
                # save info
                zs.append(z)
                ws.append(w)
                adapt_losses.append(loss.numpy())
            # TODO: compute loss with the latest weights? Now, there are 1 more w than loss
        return zs, ws, adapt_losses



    # def inner_loop(self, stacked_images, images):
    #     adapt_losses = []
    #     zs = []
    #     ws = []

    #     with tf.GradientTape(persistent=False) as tape:
    #         z = self.model.meta_encoder(stacked_images)
    #         w = self.model.meta_decoder(z)
    #         loss = self.model.compute_loss(images, w)
    #     grads = tape.gradient(loss, z)
    #     print(grads)
    #     # grads_clipped, global_norm = tf.clip_by_global_norm([grads], clip_norm=config.clipnorm_inner)

    #     zs.append(z)
    #     ws.append(w)

    #     with tf.GradientTape(persistent=False) as tape:
    #         z -= self.ALPHA * grads
    #         w = self.model.meta_decoder(z)
    #         loss = self.model.compute_loss(images, w)
    #     z = tf.stop_gradient(z)
    #     grads = tape.gradient(loss, z)
    #     print(grads)
    #     # grads_clipped, global_norm = tf.clip_by_global_norm([grads], clip_norm=config.clipnorm_inner)

    #     zs.append(z)
    #     ws.append(w)

    #     adapt_losses.append(loss.numpy())
    #     return zs, ws, adapt_losses
