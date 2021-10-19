import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing

lblt = preprocessing.LabelEncoder()

outlook = [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2]
temperature = [0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1]
himidity = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
windy = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
play = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

transform = lblt.fit_transform(play)
feature = list(zip(outlook, temperature, himidity, windy))

clf = DecisionTreeClassifier().fit(feature, transform)
plot_tree(clf, filled = True)
plt.show()