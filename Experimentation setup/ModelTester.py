
import numpy as np
import matplotlib.pyplot as plt


from dataset.dataset_utils import Dataset
from dataset.datasets import test_ds, unseen_test_ds
import utils
import config

NUM_ADAPTATIONS = config.num_adaptations
K = config.K

class ModelTester:
    def __init__(self, trainer):
        self.trainer = trainer


    def test(self):
        # TESTING
        # seen test
        test_adapt_losses = self.test_task_adaptation(dataset=test_ds)
        figure = utils.plot_test_adapt_losses(test_adapt_losses, NUM_ADAPTATIONS)
        utils.save_fig(fig=figure, name=f"{self.trainer.model.class_name}-test_adapt_losses")

        # unseen test
        unseen_test_adapt_losses = self.test_task_adaptation(dataset=unseen_test_ds)
        figure = utils.plot_test_adapt_losses(unseen_test_adapt_losses, NUM_ADAPTATIONS)
        utils.save_fig(fig=figure, name=f"{self.trainer.model.class_name}-unseen_test_adapt_losses")

        
    def test_task_adaptation(self, dataset):
        test_adapt_losses = []
        print("TESTING")
        print("Task nr -- No adapt loss -- Max adapt loss:")
        ntasks = len(dataset.data)
        for i,task in enumerate(dataset.data):
            images = Dataset.choose_N_images(task, K)
            adapt_losses = self.trainer.adapt_to_task(images)
            test_adapt_losses.append(adapt_losses)
            
            print(f"{i+1}/{ntasks} -- {adapt_losses[0]} -- {adapt_losses[-1]}")
        return np.asarray(test_adapt_losses)


    # def visualize_task_adaptation_vae(self, dataset):
    #     # choose random task
    #     task = dataset.data[ np.random.randint( len(dataset.data) ) ]
    #     offset = 1 # row for task images
    #     nr_rows = offset + NUM_ADAPTATIONS
    #     nr_cols = K
    #     fig, axes = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(15, 5))
    #     fig.subplots_adjust(wspace=0, hspace=0.1)
        
    #     images = task[:K]
    #     # choose a random image as seed for generating new images
    #     # rand_idx = np.random.randint(K)
    #     # seed_image = images[rand_idx]
    #     for i,img in enumerate(images):
    #         axes[0,i].imshow(img[:,:,0], cmap="gray")
    #         axes[0,i].axis('off')

    #     generated_images = self.trainer.record_adaptations(images)
    #     for n in range(offset, NUM_ADAPTATIONS+offset):
    #         adapt_images = generated_images[n]
    #         for i,img in enumerate(adapt_images):
    #             axes[n,i].imshow(img[:,:,0], cmap="gray")
    #             axes[n,i].axis('off')
    #     return fig


    def visualize_task_adaptation_vae(self, dataset):
        # choose random task
        task = dataset.data[ np.random.randint( len(dataset.data) ) ]
       
        
        images = task[:K] # List[Images]
        adapt_reconstructions, generated_images = self.trainer.record_adaptations(images) # List[List[Images]]

        # Reconstructions
        all_images = [] # List[List[Images]]
        all_images.append(images)
        all_images.extend(adapt_reconstructions)

        fig_recon, axes = plt.subplots(nrows=len(all_images), ncols=K, figsize=(15, 5))
        fig_recon.subplots_adjust(wspace=0, hspace=0.1)

        for i,images_row in enumerate(all_images):
            for j,img in enumerate(images_row):
                axes[i,j].imshow(img[:,:,0], cmap="gray")
                axes[i,j].axis('off')

        # Generated
        all_images = [] # List[List[Images]]
        all_images.append(images)
        all_images.extend(generated_images)

        fig_gen, axes = plt.subplots(nrows=len(all_images), ncols=K, figsize=(15, 5))
        fig_gen.subplots_adjust(wspace=0, hspace=0.1)

        for i,images_row in enumerate(all_images):
            for j,img in enumerate(images_row):
                axes[i,j].imshow(img[:,:,0], cmap="gray")
                axes[i,j].axis('off')

        return fig_recon, fig_gen

    

    def visualize_adaptations(self):
        # VISUALIZE TASK ADAPTATION
        # seen test
        # TODO: do this multiple times
        for i in range(3):
            fig_recon, fig_gen = self.visualize_task_adaptation_vae(dataset=test_ds)
            utils.save_fig(fig=fig_recon, name=f"{i}_{self.trainer.model.class_name}-adapt_reconstructions_seen_test")
            utils.save_fig(fig=fig_gen, name=f"{i}_{self.trainer.model.class_name}-adapt_generated_seen_test")

            # unseen test
            fig_recon, fig_gen = self.visualize_task_adaptation_vae(dataset=unseen_test_ds)
            utils.save_fig(fig=fig_recon, name=f"{i}_{self.trainer.model.class_name}-adapt_reconstructions_unseen_test")
            utils.save_fig(fig=fig_gen, name=f"{i}_{self.trainer.model.class_name}-adapt_generated_unseen_test")


