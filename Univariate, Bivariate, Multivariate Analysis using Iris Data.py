#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")


# In[5]:


df.head()


# In[7]:


df.shape


# #### Univariate Analysis

# In[10]:


df_setosa=df.loc[df['species']=='setosa']


# In[11]:


df_setosa


# In[12]:


df_virginica=df.loc[df['species']=='virginica']


# In[13]:


df_virginica


# In[14]:


df_versicolor=df.loc[df['species']=='versicolor']


# In[15]:


df_versicolor


# In[19]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']))
plt.show()


# In[25]:


plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'+')
plt.show()


# In[27]:


plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'+')
plt.show()


# In[30]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'+')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'+')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'+')
plt.xlabel('Sepal Length')
plt.show()


# ###### Bivariate Analysis

# In[37]:


sns.FacetGrid(df,hue="species",size=10).map(plt.scatter,"petal_length","petal_width").add_legend();
plt.show()


# #### Multivariate Analysis

# In[45]:


sns.pairplot(df,hue="species",size=2)


# In[ ]:




