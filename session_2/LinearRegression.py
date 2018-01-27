
# coding: utf-8

# In[1]:


# Importing All dependencies to be Used

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
rng=np.random


# In[4]:


# Building the Graph!


# Declaring Tensorflow variables and placeholders for Linear regression Model
# The equation of a line is : y=wx+b
# We will declare two tensorflow variables (w and b) and two tensorflow placeholders (x and y)

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

print (W)


# In[5]:


# Defining the Equation of Line

pred = tf.add(tf.multiply(X, W), b)  # Equivalent to : pred_y=wx+b

# Defining the Loss ( Mean Squared Error )
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Defining the Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Initializing all variables
init=tf.global_variables_initializer()

# End Defining the Graph! 


# In[8]:


# Start training
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(1000):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % 50 == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                 "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

