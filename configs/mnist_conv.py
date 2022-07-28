#########
# Data: #
#########

exp_name = "mnist_conv_cond"
dataset = "mnist"
model = "conv"
# cond_size = 6

add_image_noise = 0.15

##############
# Training:  #
##############

# conditional = True
# make_class_cond = False
# cond_feature_channels = 16
# fc_cond_length = 64
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
pre_low_lr = 0#1

latent_noise = 0.05

#################
# Architecture: #
#################

coupling_block = "gin"
internal_width = 392
dropout = 0.3
clamp = 1.5

conditional = True
ica = False
empirical_var = True

# conv
n_blocks_fc = 2
depths = [4, 4]#, 6]
channels = [16, 32]
# splits = [False, False]#0.5, 0.5, 0.25]
# reshapes = ["reshape", "reshape"] # "haar"]
# kernel_size = 1

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

from_checkpoint = False
checkpoint_save_interval = 10
# checkpoint_save_overwrite = True            # Overwrite each checkpoint with the next one
# checkpoint_on_error = True                  # Write out a checkpoint if the training crashes
