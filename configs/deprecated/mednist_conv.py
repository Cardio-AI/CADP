#########
# Data: #
#########

exp_name = "mednist_conv"
dataset = "mednist"

add_image_noise = 0.15

##############
# Training:  #
##############

model = "conv"
# cond_size = 6

conditional = False
# cond_feature_channels = 16
# fc_cond_length = 64
log10_lr = -4.0                     # Log learning rate
lr = 10**log10_lr
batch_size = 64
decay_by = 0.005
weight_decay = 1e-8
betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 120 * 4
n_its_per_epoch = 2**16

init_scale = 0.03
pre_low_lr = 0

latent_noise = 0.05

#################
# Architecture: #
#################

# fc small
internal_width = 1024
dropout = 0.
clamp = 1.5

# conv
n_blocks_fc = 8
depths = [4, 6, 6, 6]
channels = [32, 64, 128, 256]
splits = [False, False, 0.5, 0.5]
reshapes = ["reshape", "reshape", "reshape", "haar"]
kernel_size = 1
scale_l_fwd = True

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
# preview_upscale = 3                         # Scale up the images for preview
sampling_temperature = 1.0                  # Sample at a reduced temperature for the preview
live_visualization = True                   # Show samples and loss curves during training, using visdom
progress_bar = True

###################
# Loading/saving: #
###################

from_checkpoint = True
checkpoint_save_interval = 5
# checkpoint_save_overwrite = True            # Overwrite each checkpoint with the next one
# checkpoint_on_error = True                  # Write out a checkpoint if the training crashes
