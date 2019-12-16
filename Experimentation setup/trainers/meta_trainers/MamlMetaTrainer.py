import time

import numpy as np
import tensorflow as tf

import config
import utils

from trainers.BaseTrainer import BaseTrainer

class MamlMetaTrainer(BaseTrainer):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model

        assert len(self.model.learner_model.trainable_variables) != 0, "in init"

        self.train_dataset = dataset


    def record_adaptations(self, images):
        reconstructions = []

        # perform adaptations and recover params weights
        ws, _ = self.inner_loop(images=images)
        # apply weights to learner model (includes pre-adaptation weights)
        for w in ws:
            self.model.set_weights(w)
            images_dist = self.model.learner_model(images)
            _, images_mean, _ = utils.sample_images(images_dist)
            reconstructions.append(images_mean)

        # generate new images from adapted model
        gen_sample, gen_mean, gen_mode = self.generate_new_samples(prior=self.model.learner_prior, decoder=self.model.learner_decoder)
        new_images = [gen_sample, gen_mean, gen_mode]        
        
        return reconstructions, new_images

    def adapt_to_task(self, images):
        _, adapt_losses = self.inner_loop(images=images)
        np_adapt_losses = np.asarray(adapt_losses)
        return np.mean(np_adapt_losses, axis=1)


    def train(self):
        train_losses, val_losses, batch_adapt_losses = self.training()
        figure = utils.plot_losses_vs_adapt_gain(train_losses, val_losses, batch_adapt_losses, self.train_dataset)
        utils.save_fig(fig=figure, name="losses_vs_adapt_gain")


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
            meta_losses = []
            for task in batch:
                images_train = task[:K] # first K images
                ws, adapt_losses = self.inner_loop(images=images_train)
                task_adapt_losses.append(adapt_losses)

                images_val = task[K:K*2] # next K images
                meta_loss = self.model.compute_loss(images_val, ws[-1])
                meta_losses.append(meta_loss)

            batch_loss = tf.reduce_mean(meta_losses)
        self.model.meta_learning_step(meta_tape, batch_loss)
        
        batch_adapt_loss = np.asarray(task_adapt_losses)
        batch_adapt_loss = np.mean( batch_adapt_loss, axis=-1 ) # mean over images
        batch_adapt_loss = np.mean( batch_adapt_loss, axis=0 ) # mean over tasks --> avg loss for each adaptation step

        return batch_loss.numpy(), batch_adapt_loss



    def inner_loop(self, images):
        adapt_losses = []
        ws = []

        assert len(self.model.learner_model.trainable_variables) != 0, "in inner loop"

        w_init = self.model.get_weights()
        assert len(self.model.learner_model.trainable_variables) != 0, "in inner loop"
        # ws.append(w) # initial weights, no adaptation
        with tf.GradientTape(persistent=True) as tape:
            for _ in range(self.NUM_ADAPTATIONS):
                w = self.model.get_weights()
                print("WEIGHTS LEN", len(w))
                assert len(self.model.learner_model.trainable_variables) != 0, "in inner loop"
                loss = self.model.compute_loss(images, w)
                assert len(self.model.learner_model.trainable_variables) != 0, "in inner loop"
                grads = tape.gradient(loss, self.model.learner_model.trainable_variables)

                raise ValueError("!")
                self.model.apply_gradients(grads)
                # w = self.model.get_weights()
                ws.append(w)
                adapt_losses.append(loss.numpy())
        # TODO: add an extra loss, so it matches len(ws)?

        # IMPORTANT: reset model to initial weights
        self.model.set_weights(w_init)
        return ws, adapt_losses
