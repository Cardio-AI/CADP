#########
# Data: #
#########

exp_name = "celeba_conv_cond"
dataset = "celeba"
model = "conv"
coupling_block = "glow"

add_image_noise = 0.15

##############
# Training:  #
##############

conditional = True
# cond_size = 40
# cond_feature_channels = 16
# fc_cond_length = 64
# log10_lr = -4.0                     # Log learning rate
lr = 3e-4 # 10**log10_lr
batch_size = 16
decay_by = 0.0005
weight_decay = 1e-8
betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 120 * 4
n_its_per_epoch = 2**16

init_scale = 0.03
pre_low_lr = 0

latent_noise = 0.05

ica = False
empirical_var = True

#################
# Architecture: #
#################

# fc small
internal_width = 1024
dropout = 0.
clamp = 1.2

# conv
n_blocks_fc = 12
depths = [6, 6, 6, 6]
channels = [32, 64, 64, 128]#, 256, 256]
splits = [False, 0.5, 0.5, 0.25]
#reshapes = ["reshape", "reshape", "reshape", "reshape", "reshape"]# "haar", "haar"]
# kernel_size = 3

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
# preview_upscale = 3                         # Scale up the images for preview
sampling_temperature = 0.7                  # Sample at a reduced temperature for the preview
live_visualization = True # False                   # Show samples and loss curves during training, using visdom
progress_bar = True

###################
# Loading/saving: #
###################

from_checkpoint = True
checkpoint_timestamp = "1645359795.879994"
checkpoint_save_interval = 1
# checkpoint_save_overwrite = True            # Overwrite each checkpoint with the next one
# checkpoint_on_error = True                  # Write out a checkpoint if the training crashes
