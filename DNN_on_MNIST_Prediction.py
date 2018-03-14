from sklearn.datasets import fetch_mldata
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

mnist=fetch_mldata("MNIST Original")
X=mnist['data']
y=mnist['target'].astype('int64')
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
shuffle_index=np.random.permutation(60000)
X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]
X_data=X_test[:1]
y_data=y_test[:1]
print("y_data",y_data)


# Construction Phase
tf.reset_default_graph()

n_inputs=28*28
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int64,shape=(None),name="y")
n_neurons=100

# DNN with 5 hidden layers
with tf.name_scope('dnn'):
    he_init=tf.contrib.layers.variance_scaling_initializer()
    hidden1=fully_connected(X,n_neurons,weights_initializer=he_init,scope="hidden1",activation_fn=tf.nn.elu)
    hidden2=fully_connected(hidden1,n_neurons,scope="hidden2",activation_fn=tf.nn.elu)
    hidden3=fully_connected(hidden2,n_neurons,scope="hidden3",activation_fn=tf.nn.elu)
    hidden4=fully_connected(hidden3,n_neurons,scope="hidden4",activation_fn=tf.nn.elu)
    hidden5=fully_connected(hidden4,n_neurons,scope="hidden5",activation_fn=tf.nn.elu)
    logits=fully_connected(hidden5,10,scope="output")
    
with tf.name_scope('loss'):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")
    
learning_rate=0.001
with tf.name_scope("train"):
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
            
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    
init=tf.global_variables_initializer()
saver=tf.train.Saver()    


with tf.Session() as sess:
    saver.restore(sess,"./final.ckpt")
    z=logits.eval(feed_dict={X:X_data})
    y_pred=np.argmax(z,axis=1)
    
print("y_pred",y_pred)    
    
    