#########
# Data: #
#########

exp_name = "chest_xray_conv_cond"
dataset = "chest_xray"

# input_noise_sigma = 0.15

##############
# Training:  #
##############

coupling_block = "glow"
model = "conv"
# cond_size = 6

conditional = True
# cond_feature_channels = 16
# fc_cond_length = 64
log10_lr = -4.0                     # Log learning rate
lr = 3e-4 # 10**log10_lr
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

ica = False
empirical_var = True

# fc small
internal_width = 1024
dropout = 0.
clamp = 1.5

# conv
n_blocks_fc = 4
depths = [4, 4, 4, 4]#6, 6, 6]
channels = [32, 64, 64, 64]# 128, 256]
# splits = [False, False, False]#, False]
# reshapes = ["reshape", "reshape", "reshape"]#, "haar"]
# kernel_size = 1
# scale_l_fwd = True

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
# preview_upscale = 3                         # Scale up the images for preview
sampling_temperature = 0.8                  # Sample at a reduced temperature for the preview
live_visualization = False                  # Show samples and loss curves during training, using visdom
progress_bar = True

###################
# Loading/saving: #
###################

from_checkpoint = False
checkpoint_timestamp = "1645514087.2942946"
checkpoint_save_interval = 5
# checkpoint_save_overwrite = True            # Overwrite each checkpoint with the next one
# checkpoint_on_error = True                  # Write out a checkpoint if the training crashes
