#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np


# In[49]:


data = pd.read_csv("/Users/iremcinar/Desktop/iş zekası/archive__1_/column_2C_weka.csv")


# In[50]:


Labels = data.loc[:,'class']


# In[51]:


data.loc[:,'class'] = [1 if each == 'Normal' else 0 for each in data.loc[:,'class'] ]
Labels = data.loc[:,'class']


# In[52]:


Labels


# In[53]:


drop = data.drop(["class"],axis = 1)


# In[54]:


drop_norm = (drop - np.min(drop))/(np.max(drop) - np.min(drop))


# In[57]:


print("NORMALİZASYON İŞLEMİ ÖNCESİ:",
      "\nMin :")
print(np.min(drop))
print("\nMax :")
print(np.max(drop))


print("\n\nNORMALİZASYON İŞLEMİ SONRASI:",
      "\nMin :")
print(np.min(drop_norm))
print("\nMax :")
print(np.max(drop_norm))


# In[58]:


from sklearn.model_selection import train_test_split
drop_train, drop_test, y_train, y_test = train_test_split(drop_norm, Labels, test_size = 0.3, random_state = 1)


# In[59]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(drop_train,y_train)
prediction = knn.predict(drop_test)
print(" {} nn score: {} ".format(3,knn.score(drop_test,y_test)))


# In[60]:


score_list = []
for each in range(3,25):
 knn2 = KNeighborsClassifier(n_neighbors = each)
 knn2.fit(drop_train,y_train)
 score_list.append(knn2.score(drop_test,y_test))


# In[61]:


import matplotlib.pyplot as plt


# In[62]:


plt.plot(range(3,25),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In[63]:


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(drop_train,y_train)


# In[65]:


print("print accuracy of naive bayes algo: ",nb.score(drop_test,y_test))


# In[68]:


from sklearn.svm import SVC


# In[70]:


svm = SVC(random_state = 1)
svm.fit(drop_train,y_train)


# In[71]:


print("print accuracy of svm algo: ",svm.score(drop_test,y_test))


# In[83]:


from sklearn.tree import DecisionTreeClassifier


# In[86]:


dt = DecisionTreeClassifier()   # random sate = 0


# In[87]:


dt.fit(drop_train,y_train)


# In[88]:


print("score: ",dt.score(drop_test,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




