#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# # Reading the data

# In[106]:


d=pd.read_csv('Wine.csv',sep=',',encoding='latin')


# In[107]:


d.head()


# In[108]:


d.isnull().sum()


# In[109]:


t={1:0,2:1,3:2}


# In[110]:


d['Customer_Segment']=d['Customer_Segment'].map(t)


# In[205]:


x=d.drop(columns=['Customer_Segment'])
y=d['Customer_Segment']


# # Spliting the data

# In[206]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[207]:


from sklearn.preprocessing import StandardScaler
s=StandardScaler()


# In[208]:


x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)


# In[209]:


from keras.models import Sequential
from keras.layers import Dense


# In[210]:


x_train.shape


# In[211]:


from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
ct=OneHotEncoder()


# In[212]:


y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)


# In[213]:


y_train


# # Building the Model

# In[214]:


clf=Sequential()
clf.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=13))
clf.add(Dense(output_dim=6,init='uniform',activation='relu'))
clf.add(Dense(output_dim=3,init='uniform',activation='sigmoid'))


# In[215]:


clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
clf.fit(x_train,y_train,batch_size=10,nb_epoch=40)


# In[216]:


clf.summary()


# In[223]:


y_pred=clf.predict(x_test)


# In[227]:


l=[]
for i in range(0,len(y_test)):
    k=np.argmax(y_pred[i])
    l.append(k)
              


# In[229]:


y_pred=np.array(l)


# # Using pytorch

# In[132]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader


# In[161]:


x=d.drop(columns=['Customer_Segment']).values
y=d['Customer_Segment'].values


# In[162]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[163]:


x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)


# In[164]:


x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)


# In[165]:


y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


# In[166]:


trainloader=DataLoader(x_train,batch_size=60,shuffle=True)
testloader=DataLoader(x_test,batch_size=60,shuffle=False)


# In[195]:


class Model(nn.Module):
    def __init__(self,in_features=13,h1=10,h2=10,out_features=3):
        super().__init__()
        self.fc1=nn.Linear(in_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,out_features)
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        
        return x


# In[196]:


model=Model()


# In[197]:


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


# In[198]:


epochs=100
losses=[]


for i in range(epochs):
    i=i+1
    y_pred=model.forward(x_train)
    loss=criterion(y_pred,y_train)
    losses.append(loss)
    
    
    if i%10==1:
        print(f'epoch:{i:2}  loss:{loss.item():10.8f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    


# In[199]:


plt.plot(range(epochs),losses)
plt.ylabel('Loss')
plt.xlabel('epoch')


# In[200]:


with torch.no_grad():
    y_val=model.forward(x_test)
    loss=criterion(y_val,y_test)
print(f'{loss:.8f}')


# In[201]:


correct=0
with torch.no_grad():
    for i ,data in enumerate(x_test):
        y_val=model.forward(x_test)
        if y_val[0].argmax().item()==y_test[i]:
            correct=correct+1
print(f'\n{correct} out of {len(y_test)} = {100*correct/len(y_test):.2f}% correct')


# In[ ]:




