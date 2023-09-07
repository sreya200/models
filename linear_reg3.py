#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[62]:


df=pd.read_csv(r"C:\Users\sreya\Downloads\3_Fitness.csv")


# In[63]:


df


# In[15]:


df.tail()


# In[16]:


df.describe()


# In[8]:


df.info()


# In[17]:


df.count()


# In[18]:


df.columns


# In[19]:


sns.pairplot(df)


# In[20]:


sns.displot(df['TOTAL SALES'])


# In[21]:


df1=df[['SALESMAN', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'TOTAL SALES']]


# In[22]:


df1


# In[38]:


df.fillna(0, inplace = True)


# In[39]:


df


# In[40]:


sns.heatmap(df1.corr())


# In[50]:


x=df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']]
y=df['TOTAL SALES']


# In[51]:


pip install sklearn


# In[52]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[54]:


x_train


# In[55]:


pip install scikit-learn


# In[56]:


from sklearn import linear_model


# In[57]:


from sklearn.linear_model import LinearRegression  
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[58]:


predx=lr.predict(x_test)
print(predx)


# In[59]:


print(lr.score(x_test,y_test))


# In[60]:


plt.scatter(y_test,predx)


# In[ ]:





# In[ ]:




