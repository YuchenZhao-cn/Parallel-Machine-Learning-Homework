import tensorflow.compat.v1 as tf
import time as t
import math
tf.disable_v2_behavior()

a = tf.random.uniform([10000, 10000], 1, 10)
b = tf.random.uniform([10000, 10000], 1, 10)
c = []

t1 = t.time()
with tf.device('/gpu:0'):
    a1 = pow(a, 10)
    b1 = pow(b, 10)
with tf.Session() as sess:
    c = sess.run(a1 + b1)

print("time (1 gpu): %f" %(t.time() - t1))

t1 = t.time()
with tf.device('/gpu:0'):
    a2 = pow(a, 10)
with tf.device('/gpu:1'):
    b2 = pow(b, 10)
with tf.Session() as sess:
    c = sess.run(a2 + b2)

print("time (2 gpu): %f" %(t.time() - t1))
