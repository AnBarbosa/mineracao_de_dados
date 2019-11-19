
# coding: utf-8

# # Exercício
# 
# Considere a base de dados [Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase). Identifique, usando validação cruzada de 5 pastas, qual classificador apresenta melhores resultados:
# 
#     Árvore de Decisão com max_depth= None ou 2
#     KNN com número de vizinhos 3 ou 5 e ponderação uniforme ou pela distância
# 
# 

# In[41]:


import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[21]:


dados = pd.read_csv("spambase.data", header=None)


# In[22]:


X, y = dados.drop(57, axis=1), dados[57]


# In[23]:


# Algoritmos
knn = KNeighborsClassifier()
tree = DecisionTreeClassifier()


# In[28]:


param_grid_tree = [{'max_depth': [None, 2]}]
param_grid_knn = [{'n_neighbors': [3, 5], 'weights': ['uniform','distance']}]

grid_search_knn = GridSearchCV(knn, param_grid_knn, cv = 5)
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv = 5)


# In[29]:


grid_search_knn.fit(X,y)
grid_search_tree.fit(X,y)


# In[30]:


means_knn = grid_search_knn.cv_results_['mean_test_score']
stds_knn = grid_search_knn.cv_results_['std_test_score']

means_tree = grid_search_tree.cv_results_['mean_test_score']
stds_tree = grid_search_tree.cv_results_['std_test_score']


# In[38]:


print(f"KNN:  {np.mean(means_knn):.3f} +/- {np.mean(stds_knn):.3f}\n"+
      f"TREE: {np.mean(means_tree):.3f} +/- {np.mean(stds_tree):.3f}")


# ## OUTRA SOLUCAO

# In[45]:



param_grid= [{'max_depth': [None, 2]} ,  # Tree
             {'n_neighbors': [3, 5], 'weights': ['uniform','distance']}] # KNN

# Deve ser criado fora do loop, para sempre comparar a mesma coisa
divisor = StratifiedKFold(n_splits = 5, random_state = 1) 
for clf, parameters in zip([tree, knn], param_grid):
    grid_search = GridSearchCV(clf, parameters, cv = divisor)
    grid_search.fit(X, y)
    
    
    print( grid_search.best_estimator_)
    print( np.mean(grid_search.cv_results_['mean_test_score']))
    print( np.mean(grid_search.cv_results_['std_test_score']))
    print("-"*20)
    

?pd.read_csv
