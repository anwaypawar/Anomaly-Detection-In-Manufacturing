#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#importing image fromt the dict
img = cv.imread("voxel 0 0.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 0.jpg image '), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[19]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("voxel 0 1.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 1.jpg Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[20]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("voxel 0 2.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 3.jpg Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[25]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("voxel 0 3.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 3.jpg Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[22]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("voxel 0 4.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 4.jpg Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[23]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("voxel 0 5.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 5.jpg Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[24]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("voxel 1 0.jpg",0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('voxel 0 6.jpg Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:





# In[ ]:




