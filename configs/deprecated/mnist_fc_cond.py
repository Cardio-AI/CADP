#########
# Data: #
#########

exp_name = "mnist_fc_cond"
dataset = "mnist"

add_image_noise = 0.15

##############
# Training:  #
##############

model = "fc"
cond_size = 10
conditional = True

lr = 5e-4
batch_size = 512
decay_by = 0.005
weight_decay = 1e-8
betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 120 * 12
n_its_per_epoch = 2**16

init_scale = 0.03
pre_low_lr = 1

latent_noise = 0.05

#################
# Architecture: #
#################

# fc small
n_blocks = 24
internal_width = 512
dropout = 0.3
clamp = 1.5

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
# preview_upscale = 3                         # Scale up the images for preview
sampling_temperature = 0.8                  # Sample at a reduced temperature for the preview
live_visualization = True                   # Show samples and loss curves during training, using visdom
progress_bar = True

###################
# Loading/saving: #
###################

from_checkpoint = True
checkpoint_save_interval = 2
# checkpoint_save_overwrite = True            # Overwrite each checkpoint with the next one
# checkpoint_on_error = True                  # Write out a checkpoint if the training crashes
