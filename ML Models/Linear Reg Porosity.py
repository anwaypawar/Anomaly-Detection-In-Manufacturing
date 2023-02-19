#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import pandas as pd
# Load the Ground Truth Excel File, While loading the data directory should be changed/upadted as file is upload as the data file during project submission
df = pd.read_excel('../Ground Truth - Voxel Tensor Index_ Sample1.xlsx')
df.columns


# In[3]:


# Normalize the column values. (Ex: If Column values are 1,5,3,2. Output -> 0.2, 1, 0.6, 0.4)
df['Total Pore Area (Pixel2)'] = df['Total Pore Area (Pixel2)'] /df['Total Pore Area (Pixel2)'].abs().max()
df['Average Pore Size (Pixel2)']=df['Average Pore Size (Pixel2)']/df['Average Pore Size (Pixel2)'].abs().max()
df['%Area Porosity']=df['%Area Porosity']/df['%Area Porosity'].abs().max()
df['Pore Count']=df['Pore Count']/df['Pore Count'].abs().max()


# In[4]:


# Plot Pore Count column
df['Pore Count'].plot()


# In[5]:


# Plot Area Porosity column
df['%Area Porosity'].plot()


# In[6]:


# Sort Values by % area porosity to gain insights of the linear relationships
df = df.sort_values(by='%Area Porosity')


# In[7]:


# since, we have 90 observations in the excel, we define a list of 90 elements
x=list(range(0,90))

# Define figure size ratio
plt.figure(figsize=(18, 5))

# Assign values to plot
plt.plot(x,df['Total Pore Area (Pixel2)'], label='Total Pore Area (Pixel2)')  # etc.
plt.plot(x,df['Average Pore Size (Pixel2)'], label='Average Pore Size (Pixel2)')
plt.plot(x,df['%Area Porosity'], label='%Area Porosity')
plt.plot(x,df['Pore Count'], label='Pore Count')

# Define Plot Labels
plt.xlabel('Ground Truth Sample Index')
plt.ylabel('Respective Normalized value')
plt.title("PoreCount vs Porosity Study")

# Save the figure as png for future reference
plt.savefig('porecount_area_poresize.png', dpi=900)

# Plot in notebook output
plt.legend();


# In[23]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics

# defining feature matrix(X) and response vector(y)
X=df['Pore Count'] .copy()
X=[[x] for x  in X]
y=df['%Area Porosity'].copy()

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics

# defining feature matrix(X) and response vector(y)
X=df['Pore Count'] .copy()
X=[[x] for x  in X]
y=df['%Area Porosity'].copy()

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()


# In[29]:


from sklearn.metrics import r2_score
y_pred = reg.predict(X_test)
y_true = np.array(y_test)
r2_score(y_true, y_pred)


# In[33]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_true, y_pred)


# In[24]:





# In[ ]:




