#!/usr/bin/env python
# coding: utf-8

# In[64]:

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import warnings

warnings.filterwarnings('ignore')
p_penalty = sys.argv[1]
p_Tol = float(sys.argv[2])
p_C = float(sys.argv[3])
p_fit_intercept = True if sys.argv[4] == "True" else False
p_random_state = None if sys.argv[5] =="None" else int(sys.argv[5])
p_solver = sys.argv[6]
p_max_iter = int(sys.argv[7])
p_n_jobs = None if sys.argv[8]=="None" else int(sys.argv[8])
p_dual = True if sys.argv[9]=="True" else False
p_l1_ratio = None if sys.argv[10]=="None" else float(sys.argv[10])
#


df_reglog = pd.read_csv('./out.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[119]:

#df_reglog.to_csv('out.csv')

X_train, X_test, y_train, y_test = train_test_split(df_reglog.drop('classe', axis=1), df_reglog['classe'])

# In[120]:


LogReg = LogisticRegression(penalty = p_penalty, solver=p_solver, tol=p_Tol, max_iter=p_max_iter,
                                C=p_C, fit_intercept=p_fit_intercept, random_state=p_random_state,
                            n_jobs=p_n_jobs, dual=p_dual, l1_ratio=p_l1_ratio)

LogReg.fit(X_train, y_train)



y_pred = LogReg.predict(X_test)

# In[122]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

# In[123]:


from sklearn.metrics import f1_score

f1_score(y_test, y_pred)


mon_dictionnaire = {"score": f1_score(y_test, y_pred), "TN": confusion_matrix(y_test, y_pred)[0][0],
                    "FP": confusion_matrix(y_test, y_pred)[0][1], "FN": confusion_matrix(y_test, y_pred)[1][0],
                    "TP": confusion_matrix(y_test, y_pred)[1][1]}

# In[127]:


print(mon_dictionnaire)

# In[ ]:




