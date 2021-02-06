# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:30:29 2021

@author: lucho
"""

import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.svm import SVC


iris = datasets.load_iris()

print(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']
#########################################################################
new_target =  np.where(iris.target<1, 0, 1)
colormap =np.array(['BLUE','GREEN','CYAN'])
plt.scatter(x.Sepal_Length, x.Sepal_width,c=colormap[new_target],s=40)

new_data = x[['Sepal_Length','Sepal_width']]
plt.scatter(new_data.Sepal_Length, new_data.Sepal_width,
            c=colormap[new_target],s=40)


svm = SVC(C=1)
svm.fit(new_data, new_target)
svm.support_vectors_

# 1. Tracer l'hyperplan de marge maximale séparateur


svm = SVC(kernel='linear', C=1000)
svm.fit(new_data, new_target)

plt.scatter(new_data.Sepal_Length, new_data.Sepal_width, c=colormap[new_target],s=40)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()



# 2. Evaluer l'algorithme de classification

false_positive_rate, true_positive_rate, thresholds = sm.roc_curve(new_target, svm.predict(new_data))
roc_auc = sm.auc(false_positive_rate, true_positive_rate)
print(roc_auc)


confusion_mat = sm.confusion_matrix(new_target, svm.predict(new_data))
print(confusion_mat)


# 3. Choisir le C optimal en utilisant la Validation Croisée 
from sklearn.model_selection import GridSearchCV

model = SVC()
params = {'C': np.linspace(1e-3, 1e3, 100)}

model_grid = GridSearchCV(model, params, cv=10)
model_grid.fit(new_data, new_target)

print(model_grid.best_params_)
