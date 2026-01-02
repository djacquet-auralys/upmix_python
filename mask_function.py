# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:01:22 2025

@author: Damien
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1.0, 1.0, 0.001)
PAN = -0.24
W = 0.12
SLOPE = 80
FLOOR = -30
y = 10 ** (np.maximum(np.minimum(SLOPE * (W / 2 - np.abs(x - PAN)), 0), FLOOR) / 20.0)

plt.figure()
plt.plot(x, y)
plt.grid(True)  # 'both' est équivalent à True
plt.title(
    "loi de gain d"
    "extraction pour pan = "
    + str(PAN)
    + ", w = "
    + str(W)
    + ", slope = "
    + str(SLOPE)
    + "dB, floor = "
    + str(FLOOR)
    + "dB"
)
