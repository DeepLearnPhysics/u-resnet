import sys,os
lib_path = os.path.join(os.environ['URESNETDIR'],'lib')
if not lib_path in sys.path:
    print lib_path
    sys.path.insert(0,lib_path)

import ssnet_trainval as api
import tensorflow as tf
import sys

t = api.ssnet_trainval()
for argv in sys.argv:
    if argv.endswith('.cfg'): t.override_config(argv)

t.initialize()

sess = tf.InteractiveSession() 

t.run(sess=sess)
