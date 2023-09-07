#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("C:/Users/sreya/Downloads/C10_Loan1.csv")
df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


MaritalStatus={"Marital Status":{"Married":1,"Divorced":2,"Single":3}}


# In[10]:


df=df.replace(MaritalStatus)
df


# In[11]:


df["Marital Status"].value_counts()


# In[12]:


df = df.copy() 


# In[35]:


x=df[['Home Owner','Annual Income','Defaulted Borrower']]
y=df["Marital Status"]


# In[39]:


HomeOwner={'Home Owner':{'Yes':1,'No':0}}
x=x.replace(HomeOwner)


# In[43]:


DefaultedBorrower={'Defaulted Borrower':{'Yes':1,'No':0}}
x=x.replace(DefaultedBorrower)


# In[44]:


x


# In[45]:


y


# In[46]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40)


# In[47]:


x_train


# In[48]:


y_train


# In[49]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[50]:


rfc.fit(x_train,y_train)


# In[51]:


rf=RandomForestClassifier()
params={'max_depth':[1,2,3,4,5],
       'min_samples_leaf':[2,4,6,8,10],
       'n_estimators':[1,3,4,5,7]}


# In[52]:


from sklearn.model_selection import GridSearchCV


# In[53]:


grid_search=GridSearchCV(estimator=rf,param_grid=params,cv=2,scoring='accuracy')


# In[54]:


grid_search.fit(x_train,y_train)


# In[55]:


rf_best=grid_search.best_estimator_
rf_best


# In[56]:


from sklearn.tree import plot_tree


# In[57]:


plt.figure(figsize=(40,40))


# In[59]:


plot_tree(rf_best.estimators_[2],feature_names=None,class_names=['Yes','No'])


# In[ ]:




