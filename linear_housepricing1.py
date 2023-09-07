#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv(r'C:\Users\sreya\Downloads\10_USA_Housing (1).csv')


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.count()


# In[10]:


df.columns


# In[11]:


sns.pairplot(df)


# In[12]:


sns.displot(df['Price'])


# In[13]:


sns.displot(df['Avg. Area Income'])


# In[14]:


df1=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']]


# In[15]:


df1


# In[16]:


sns.heatmap(df1.corr())


# In[23]:


x=df1[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']


# In[24]:


pip install sklearn


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[26]:


pip install scikit-learn


# In[27]:


from sklearn import linear_model


# In[28]:


from sklearn.linear_model import LinearRegression  
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[29]:


predx=lr.predict(x_test)
print(predx)


# In[30]:


print(lr.score(x_test,y_test))


# In[31]:


plt.scatter(y_test,predx)


# In[ ]:




