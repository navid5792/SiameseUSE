#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 00:22:59 2019

@author: bob
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from matplotlib import pyplot, transforms
import numpy as np
rcParams['figure.figsize'] = 1, 6
a = [0.0211, 0.0123, 0.0653, 0.1619, 0.1476, 0.1203, 0.1052, 0.0512, 0.0294,
         0.0475, 0.0258, 0.1519, 0.0460, 0.0146]
#a = np.array(a)
#a /= max(a)
#a = list(a)

df = pd.DataFrame({"": a},
                  index=['<s>', 'The', 'young', 'boys', 'are', 'playing', 'outdoors', 'and', 'the', 'man', 'is', 'smiling', 'nearby', '</s>'])

sns.heatmap(df,   cmap="YlGnBu", cbar=False)

base = pyplot.gca().transData
rot = transforms.Affine2D().rotate_deg(90)
plt.show()

