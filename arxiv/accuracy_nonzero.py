import numpy as np
import tensorflow as tf

BATCH = 10
#NPIX  = 512*512
NPIX  = 10

a = tf.placeholder(tf.float32,[BATCH,NPIX])

b = np.zeros([BATCH,NPIX],dtype=np.float32)
for i in xrange(BATCH): 
    if i%2 == 0:
        b[i,:] = 2.*i
    print b[i,:].sum() / NPIX

idx = tf.where(a>BATCH/2)
nonzero = tf.gather_nd(a,idx)
mean = tf.reduce_mean(nonzero)
sess=tf.InteractiveSession()
res = sess.run([idx,nonzero,mean],feed_dict={a:b})

print b
print
print res[0]
print
print res[1]
print
print res[2]
print
#print b.sum() / len(np.where(b>BATCH/2)) / NPIX
