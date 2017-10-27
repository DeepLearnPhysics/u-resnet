#IMPORT NECESSARY PACKAGES
import os,sys
#from toy_config import config
from uresnet_config import config

# Define constants
#
cfg = config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
  print 'Exiting...'
  sys.exit(1)

# Print configuration
print cfg

# ready to import heavy packages
import numpy as np
import tensorflow as tf
#from toydata import make_classification_images as make_images

#START ACTIVE SESSION                                                         
sess = tf.InteractiveSession()

#PLACEHOLDERS                                                                 
data_tensor    = tf.placeholder(tf.float32,  [None, 262144],name='x')
data_tensor_2d = tf.reshape(data_tensor,[-1,512,512,1])
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')
weight_tensor  = tf.placeholder(tf.float32, [None,512*512])
#RESHAPE IMAGE IF NEED BE                                                     
tf.summary.image('input',data_tensor_2d,10)

#BUILD NETWORK
net = uresnet
cmd = 'import uresnet; net=uresnet.build(data_tensor_2d, num_classes=cfg.NUM_CLASS)'
exec(cmd)

#Reshape
net = tf.reshape(net, [-1,cfg.NUM_CLASS])

#SOFTMAX
with tf.name_scope('softmax'):
  softmax = tf.nn.softmax(logits=net)

#CROSS-ENTROPY
with tf.name_scope('cross_entropy'):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=tf.argmax(net,1))
  weighted_loss = tf.multiply(loss,weight_tensor)
  cross_entropy = tf.reduce_mean(weighted_loss)
  tf.summary.scalar('cross_entropy',cross_entropy)

#TRAINING (RMS OR ADAM-OPTIMIZER OPTIONAL)                                    
with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.0003).minimize(cross_entropy)

#ACCURACY                                                                     
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), label_tensor,1)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

saver= tf.train.Saver()

sess.run(tf.global_variables_initializer())

#MERGE SUMMARIES FOR TENSORBOARD                                              
merged_summary=tf.summary.merge_all()

#WRITE SUMMARIES TO LOG DIRECTORY LOGS6                                       
writer=tf.summary.FileWriter(cfg.LOGDIR)
writer.add_graph(sess.graph)

if cfg.LOAD_FILE:
  vlist=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  reader=tf.train.Saver(var_list=vlist)
  reader.restore(sess,cfg.LOAD_FILE)

#TRAINING
save_iter = int(cfg.SAVE_ITERATION)
for i in range(cfg.ITERATIONS):

    batch = make_images(cfg.BATCH_SIZE,debug=cfg.DEBUG, multiplicities= False)

    if save_iter and (i+1)%save_iter == 0:
        
        s = sess.run(merged_summary, feed_dict={data_tensor:batch[0], label_tensor:batch[1]})
        writer.add_summary(s,i)
        
        train_accuracy = accuracy.eval(feed_dict={data_tensor:batch[0], label_tensor: batch[1]})
    
        print("step %d, training accuracy %g"%(i, train_accuracy))

        save_path = saver.save(sess,cfg.ARCHITECTURE,global_step=i)
        print 'saved @',save_path

    sess.run(train_step,feed_dict={data_tensor: batch[0], label_tensor: batch[1]})

    if i%1000 ==0:
        batchtest = make_images(cfg.BATCH_SIZE*10,debug=cfg.DEBUG,multiplicities=False)
        test_accuracy = sess.run(accuracy,feed_dict={data_tensor:batchtest[0], label_tensor:batchtest[1]})
        print("step %d, test accuracy %g"%(i, test_accuracy))

# post training test
#batch = None
print("Final test accuracy %g"%accuracy.eval(feed_dict={data_tensor: batch[0], label_tensor: batch[1]}))

# inform log directory
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % cfg.LOGDIR)
