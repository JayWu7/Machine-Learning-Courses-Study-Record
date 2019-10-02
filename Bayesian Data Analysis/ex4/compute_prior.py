'''
From the two independent Normal distribution to a joint prior distribution
A = N(0, 2^2), B= N(10, 10^2)
'''

import numpy as np
# from scipy.special import expit
import matplotlib.pyplot as plt
from . import plot_tools

# edit the default plot settings
plt.rc('font', size=12)

# apply custom background plotting style
plt.style.use()