# Basic imports
import os,sys,time
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
# Import more libraries (after configuration is validated)
import tensorflow as tf
from libs.uresnet import uresnet
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

NUM_CLASS         = 3
BASE_NUM_FILTERS  = 16
FILLER_CONFIG     = 'config/train_io.cfg'
LOGDIR            = 'ssnet_train_log'
SAVE_FILE         = 'ssnet_checkpoint/uresnet'
LOAD_FILE         = ''
AVOID_LOAD_PARAMS = []
BATCH_SIZE        = 10
ITERATIONS        = 100000

TRAIN             = True
NORMALIZE_WEIGHTS = False
CHECKPOINT_STEPS  = 200
SUMMARY_STEPS     = 20
KEYWORD_DATA      = 'data'
KEYWORD_LABEL     = 'label'
KEYWORD_WEIGHT    = 'weight'
#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure
if not FILLER_CONFIG:
  'Must provide larcv data filler configuration file!'
  sys.exit(1)
proc = larcv_threadio()
filler_cfg = {'filler_name' : 'TrainIO',
              'verbosity'   : 0, 
              'filler_cfg'  : FILLER_CONFIG}
proc.configure(filler_cfg)
# Start IO thread
proc.start_manager(BATCH_SIZE)
# Storage ID
storage_id=0
# Retrieve image/label dimensions
proc.next()
dim_data = proc.fetch_data(KEYWORD_DATA).dim()

net = uresnet(rows=dim_data[1], 
              cols=dim_data[2], 
              num_class=3, 
              base_num_outputs=BASE_NUM_FILTERS, 
              debug=False)

if TRAIN:
  net.construct(trainable=TRAIN,use_weight=True)
else:
  net.construct(trainable=TRAIN,use_weight=False)

#
# 2) Configure global process (session, summary, etc.)
#
# Create a bandle of summary
merged_summary=tf.summary.merge_all()
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Create a summary writer handle
writer=tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
# Create weights saver
saver = tf.train.Saver()

# Override variables if wished
if LOAD_FILE:
  vlist=[]
  for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    if v.name in AVOID_LOAD_PARAMS:
      print '\033[91mSkipping\033[00m loading variable',v.name,'from input weight...'
      continue
    print '\033[95mLoading\033[00m variable',v.name,'from',LOAD_FILE
    vlist.append(v)
  reader=tf.train.Saver(var_list=vlist)
  reader.restore(sess,LOAD_FILE)
 
# Run training loop
for i in range(ITERATIONS):
  
  # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
  batch_data   = proc.fetch_data(KEYWORD_DATA).data()
  batch_label  = proc.fetch_data(KEYWORD_LABEL).data()
  batch_weight = None
  # Start IO thread for the next batch while we train the network
  if TRAIN:
    batch_weight = proc.fetch_data(KEYWORD_WEIGHT).data()
    proc.next()
    # perform per-event normalization
    if NORMALIZE_WEIGHTS:
      batch_weight /= np.mean(batch_weight,axis=1).reshape([batch_weight.shape[0],1])
    _,loss,acc_all,acc_nonzero = net.train(sess         = sess, 
                                           input_image  = batch_data,
                                           input_label  = batch_label,
                                           input_weight = batch_weight)
    sys.stdout.write('Training in progress @ step %d loss %g accuracy %g / %g           \r' % (i,loss,acc_all,acc_nonzero))
  else:
    proc.next()
    softmax,acc_all,acc_nonzero = net.train(sess        = sess,
                                            input_image = batch_data,
                                            input_label = batch_label)
    sys.stdout.write('Training in progress @ step %d accuracy %g / %g                   \r' % (i,acc_all,acc_nonzero))

  sys.stdout.flush()

  # Save log every 20 steps
  if TRAIN and (i+1)%SUMMARY_STEPS == 0:
    # Run summary
    feed_dict = net.feed_dict(input_image  = batch_data,
                              input_label  = batch_label,
                              input_weight = batch_weight)
    writer.add_summary(sess.run(merged_summary,feed_dict=feed_dict),i)
  
  # If configured to save summary + snapshot, do so here.
  if TRAIN and (i+1)%CHECKPOINT_STEPS == 0:
    # Save snapshot
    ssf_path = saver.save(sess,SAVE_FILE,global_step=i)
    print
    print 'saved @',ssf_path

proc.reset()
