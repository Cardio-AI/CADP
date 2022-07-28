import numpy as np
np.seterr(all='raise')
import torch
import torchvision
import viz
import data
import latent_dp
DATA_DIR = "/mnt/ssd//data"

epsilons = [3.0, 6.0] # [.2, .5, .8, 1.0, 2.0, 3.0, 6.0]
sensitivity = 1.

chest_xray = data.chest_xray_train_data(f"{DATA_DIR}/chest_xray")

max_s = 4.0
for epsilon in epsilons:
    save_path = f"{DATA_DIR}/priv_chest_xray/{epsilon}"
    timestamp = "1645635622.8975923"
    exp_name = f"chest_xray_conv_cond/{timestamp}"
    priv_chest_xray = latent_dp.privatize_pipeline(exp_name=exp_name,
                                                   save_path=save_path,
                                                   data_path=DATA_DIR,
                                                   epsilon=epsilon,
                                                   sensitivity=min(epsilon/2, max_s),
                                                   mechanism="laplace")

# max_s = 4.0
# for epsilon in epsilons:
#     save_path = f"{DATA_DIR}/priv_chest_xray_test/{epsilon}"
#     timestamp = "1645635622.8975923"
#     exp_name = f"chest_xray_conv_cond/{timestamp}"
#     priv_chest_xray = latent_dp.privatize_pipeline(exp_name=exp_name,
#                                                    save_path=save_path,
#                                                    data_path=DATA_DIR,
#                                                    epsilon=epsilon,
#                                                    sensitivity=min(epsilon/2, max_s),
#                                                    dataset="test",
#                                                    mechanism="laplace")