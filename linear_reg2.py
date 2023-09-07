#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r"C:\Users\sreya\Downloads\4_Drug200.csv")


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.count()


# In[8]:


df.columns


# In[9]:


sns.pairplot(df)


# In[11]:


sns.displot(df['Drug'])


# In[14]:


df1=df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K','Drug']]


# In[15]:


df1


# In[70]:


Drug={"Drug":{"drugY":0,"drugX":1,"drugB":2,"drugA":3,"drugC":4}}
df1=df1.replace(Drug)
df1


# In[71]:


sns.heatmap(df1.corr())


# In[77]:


x=df1[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y=df1['Drug']


# In[78]:


Sex={"Sex":{"F":1,"M":2}}
x=x.replace(Sex)


# In[79]:


BP={"BP":{"LOW":0,"NORMAL":1,'HIGH':2}}
x=x.replace(BP)


# In[80]:


Cholesterol={"Cholesterol":{"HIGH":0,"NORMAL":1}}
x=x.replace(Cholesterol)
x


# In[81]:


y


# In[32]:


pip install sklearn


# In[82]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[83]:


y


# In[38]:


pip install scikit-learn


# In[57]:


from sklearn import linear_model


# In[84]:


from sklearn.linear_model import LinearRegression  
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[85]:


predx=lr.predict(x_test)
print(predx)


# In[86]:


print(lr.score(x_test,y_test))


# In[31]:


plt.scatter(y_test,predx)


# In[ ]:




