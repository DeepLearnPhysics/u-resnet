import tensorflow as tf
import numpy as np

def build(name, data_dim, num_class, train=True, debug=False):

  data_size=1
  weight_size=1
  for v in data_dim: 
    if v>0: data_size *=v

  # Set input data and label for training
  data_tensor   = tf.placeholder(tf.float32, [None, data_size/data_dim[0] ], name='data')
  weight_tensor = tf.placeholder(tf.float32, [None, data_size/data_dim[0] ], name='weight')
  label_tensor  = tf.placeholder(tf.int64,   [None, data_size/data_dim[0] ], name='label')
  
  label_reshape  = tf.reshape(label_tensor,  [-1,data_dim[1],data_dim[2]])
  weight_reshape = tf.reshape(weight_tensor, [-1,data_dim[1],data_dim[2]])
  data_reshape   = tf.reshape(data_tensor,   data_dim)

  # record some input images in tensorboard if it's 2D data
  if len(data_dim) == 4:
    tf.summary.image('data_example',data_reshape,10)

  # Call network build function (then we add more train-specific layers)
  net = None
  cmd = 'import models; net = models.%s.build(input_tensor=data_reshape,num_class=%d,base_num_filters=16,debug=%d)' % (name,num_class,int(debug))
  exec(cmd)

  #net = tf.reshape(net,[-1,num_class])

  # Define accuracy
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(net,3), label_reshape, 'label_to_int')
    forward = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', forward)

  if not train:
    return (data_tensor,label_tensor,forward,None,None)

  # Define loss + backprop as training step
  with tf.name_scope('train'):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_reshape, logits=net)
    loss = tf.reduce_mean(tf.multiply(weight_reshape, loss))
    tf.summary.scalar('loss',loss)
    train = tf.train.RMSPropOptimizer(0.0003).minimize(loss)

  return (data_tensor,label_tensor,weight_tensor,forward,loss,train)

if __name__ == '__main__':
  data_tensor, label_tensor, weight_tensor, forward, loss, train = build('uresnet',[2,512,512,1],num_class=3,train=True,debug=True)

  batch_data = np.zeros([2,512*512*1],dtype=np.float32)
  batch_label = np.zeros([2,512*512*1],dtype=np.int64)
  batch_weight = np.zeros([2,512*512*1],dtype=np.float32)
  
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  ops = [loss,forward,train]
  #ops = [forward]
  feed_dict = {data_tensor: batch_data,
               label_tensor: batch_label,
               weight_tensor: batch_weight}

  sess.run(ops,feed_dict=feed_dict)
