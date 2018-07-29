
# coding: utf-8

# In[1]:


import scipy.io as sio
import numpy as np
import cv2
import os


# In[2]:


data= sio.loadmat('nyu_training_data.mat')


# In[3]:


images = data['images']


# In[6]:


os.mkdir('Images')
for idx in range(images.shape[0]):
    img = images[idx, :, :, :]
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('Images/img_' + str(idx) + '.jpg', img)
    
    print(idx, 'of', images.shape[0])

