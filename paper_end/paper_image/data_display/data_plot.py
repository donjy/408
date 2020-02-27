# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral

# loading the data....

# indian_pines data

# input_image = loadmat('/root/donjy/datasets/matlabel/Indian_pines_corrected.mat')['indian_pines_corrected']
# output_image = loadmat('/root/donjy/datasets/matlabel/Indian_pines_gt.mat')['indian_pines_gt']

# input_image = loadmat('/root/donjy/datasets/matlabel/EMP_Indina_profiles325.mat')['EMP_KPCA_Indian']
# output_image = loadmat('/root/donjy/datasets/matlabel/Indian_pines_gt.mat')['indian_pines_gt']

# PaviaU data
input_image = loadmat('/root/donjy/datasets/matlabel/PaviaU.mat')['paviaU']
output_image = loadmat('/root/donjy/datasets/matlabel/PaviaU_gt.mat')['paviaU_gt']

#


ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9, 9))
# plt.show(ground_truth)

# the one of the data image...

view = spectral.imshow(input_image, (30, 19, 9), classes=output_image)

# 标签覆盖到label上。
# view.set_display_mode('overlay')
# view.class_alpha = 0.5


plt.axis('off')  # 去掉坐标
plt.savefig("/root/donjy/imgs/{}.png".format('aaa'))
plt.show()



# ground_truth.set_display_mode('overlay')
# ground_truth.class_alpha = 0.5
