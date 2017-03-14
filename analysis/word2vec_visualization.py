#coding=utf-8

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib  
font = {'size'   : 12}

matplotlib.rc('font', **font)
myfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc")
matplotlib.rcParams['axes.unicode_minus'] = False

word_embedding = np.load('./word_embedding.pkl.npy')
vocabulary = np.load('./vocab.pkl')

pca = PCA(n_components=2)
data = pca.fit(word_embedding).transform(word_embedding)
for i in range(100):
    x = data[i, 0]
    y = data[i, 1]
    plt.scatter(x, y)
    plt.annotate(vocabulary[i], (x, y), fontproperties=myfont)
plt.show()



