import numpy as np
import matplotlib.pyplot as plt
import torch
import os

load_dir = '/media/hdd2/research/privacy_pytorch/direct_inference_lowrank/fixed_matrix_transpose_face_identification/proj_noise0_meas0p25_lr1e-5_decreasefreq_200'
checkpoint = torch.load(os.path.join(load_dir, 'latest_network.tar'))
accs = checkpoint['accs']
x = range(len(accs['tr']))

plt.plot(x, [z/100 for z in accs['tr']], x, [z/100 for z in accs['val']])
plt.legend(('tr', 'val'))
plt.ylim(0, 1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')