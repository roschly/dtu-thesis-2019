# ===== SETUP =====
learner_model # full model
learner_encoder, learner_decoder, 
meta_model # full model
meta_encoder, meta_decoder
prior # MultivariateNormal distribution used for KL divergence loss
ALPHA # inner loop learning rate
AdamOptim(BETA) # outer loop optimizer with BETA learning rate
GAMMA # meta loss parameter - constrains meta encoding
NUM_ADAPTATIONS # nr of gradient steps in latent space


# ===== TRAINING =====
for batch in dataset:
    meta_losses = []
    for task in batch:
        images_train, images_val = task.sample()
        meta_encoder_input = stack_images(images_train) # encoder takes K images as input
        z_first, z_last = inner_loop(meta_encoder_input, images_train)
        
        # meta/validation loss
        w_last = meta_decoder(z_last) # final weights
        apply_weights_to_learner(w_last, learner_model) # apply final weights from adaptations
        meta_loss = compute_meta_loss(images_val, learner_model, z_first, z_last)
        meta_losses.append(meta_loss)
    batch_loss = mean(meta_losses)
    # meta update step
    ModelAllLayers.meta_learning_step(batch_loss)
    # ModelSomeLayers.meta_learning_step(batch_loss)


def inner_loop(meta_encoder_input, images):
    z = meta_encoder(meta_encoder_input)
    z_first = z # store for meta_loss
    w = meta_decoder(z) # generate weights for learner model
    for n in range(NUM_ADAPTATIONS):
        apply_weights_to_learner(w, learner_model)
        loss = compute_loss(images, learner_model)
        grads = gradients(loss, z)
        # update z and generate new weights
        z -= ALPHA * grads
        w = meta_decoder(z)
    return z_first, z

def compute_loss(images, learner_model):
    image_dist = learner_model(images) # learner_model returns a Bernoulli distribution
    recon_loss = -image_dist.log_prob(images) # negative log likelihood
    KL_loss = prior.kl_divergence(image_dist)
    return recon_loss + KL_loss

def compute_meta_loss(images, learner_model, z_first, z_last):
    loss = compute_loss(images, learner_model)
    z_loss = GAMMA * MSE(z_first, stopgrad(z_last)) # constrain z values closer together
    return loss + z_loss

def ModelAllLayers.meta_learning_step(batch_loss):
    # All layers in learner model are updated by the meta model
    grads = gradients(batch_loss, meta_model)
    AdamOptim.apply_gradients(grads, meta_model)

def ModelSomeLayers.meta_learning_step(batch_loss):
    # Only some layers in learner are updated by meta model.
    # The rest are trained in the normal fasion.
    for model in [meta_model, learner_model]:
        grads = gradients(batch_loss, model.trainable_vars)
        AdamOptim.apply_gradients(grads, model)


# ===== TESTING =====
# For each task, the model is allowed to adapt to the task 
# for NUM_ADAPTATIONS in the inner loop.
test_losses = []
for task in test_dataset:
    images, _ = task.sample() # no need for validation images
    meta_encoder_input = stack_images(images)
    # for ModelSomeLayers, perform adaptation on a learner_model copy
    # learner_model = copy(learner_model)
    _, z_last = inner_loop(meta_encoder_input, images)
    w = meta_decoder(z_last)
    apply_weights_to_learner(w, learner_model)
    loss = compute_loss(images, learner_model)
    test_losses.append(loss)





