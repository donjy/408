# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
output_image = loadmat('/root/donjy/datasets/matlabel/Indian_pines_corrected.mat')['indian_pines_corrected']
print(output_image)

