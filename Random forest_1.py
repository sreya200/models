#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/sreya/Downloads/C6_Bmi.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


Gender={"Gender":{"male":1,"female":2}}


# In[8]:


df=df.replace(Gender)
df


# In[9]:


df["Gender"].value_counts()


# In[10]:


x=df.drop("Gender",axis=1)
y=df["Gender"]


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[13]:


rf=RandomForestClassifier()
params={'max_depth':[1,2,3,4,5],
       'min_samples_leaf':[2,4,6,8,10],
       'n_estimators':[1,3,4,5,7]}


# In[14]:


from sklearn.model_selection import GridSearchCV


# In[15]:


grid_search=GridSearchCV(estimator=rf,param_grid=params,cv=2,scoring='accuracy')


# In[16]:


grid_search.fit(x_train,y_train)


# In[17]:


rf_best=grid_search.best_estimator_
rf_best


# In[18]:


from sklearn.tree import plot_tree


# In[19]:


plt.figure(figsize=(40,40))


# In[20]:


plot_tree(rf_best.estimators_[4],feature_names=None,class_names=['Yes','No'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




