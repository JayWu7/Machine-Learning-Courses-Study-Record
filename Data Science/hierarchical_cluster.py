from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np

X = np.array([[0, 5, 4, 9, 8],
              [5, 0, 5, 10, 7],
              [4, 5, 0, 14, 3],
              [9, 10, 14, 0, 2],
              [8, 7, 3, 2, 0]])

linked = linkage(X, 'single', optimal_ordering=True)

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top')
plt.show()
