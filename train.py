from libs.trainval import train_ssnet
import tensorflow as tf
import sys

t = train_ssnet()
for argv in sys.argv:
    if argv.endswith('.cfg'): t.override_config(argv)

t.initialize()

sess = tf.InteractiveSession() 

t.run(sess=sess)
