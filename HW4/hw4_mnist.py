import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=np.expand_dims(X_train,axis=1)
X_test=np.expand_dims(X_test,axis=1)

# Check the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# select three random number images
num_plot=3
sample_index = np.random.randint(0,X_train.shape[0],(num_plot,))

# plot the selected images
for i in range(num_plot):
  img=np.squeeze(X_train[sample_index[i]])
  ax=plt.subplot(1,num_plot,i+1)
  #ax=plt.subplot('1'+str(num_plot)+str(i))
  plt.imshow(img,cmap=plt.get_cmap('gray'))
  ######
  ## Q5. Complete the below function ax.set_title
  #####
  ax.set_title(y_train[sample_index[i]])

plt.show()
