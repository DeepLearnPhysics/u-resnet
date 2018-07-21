import pandas as pd
import numpy as np
import sys

# Read the input CSV file and create DataFrame -- table formatted data representation.
# DataFrame is an extremely useful tool for data analysis, a standard in data science in industries.
# Think of it as something like TTree with which we can apply cuts, draw a plot, etc.
#
# You can google many example usages.
# Also find an example in our image classification tutorial where we used pandas:
# http://deeplearnphysics.org/Blog/tutorials/tutorial-06.html
#

# 0) Create DataFrame from CSV file
df = pd.read_csv(sys.argv[1])

# 1) Print average total, min, max accuracy for all pixels
print
print 'All pixel accuracy mean, min, max'
print df.acc_all.mean(), df.acc_all.min(), df.acc_all.max()

# 2) Print the same for non-zero pixels
print
print 'Nonzero pixel accuracy mean, min, max'
print df.acc_nonzero.mean(), df.acc_nonzero.min(), df.acc_nonzero.max()

# 3) Print per class
print
for i in np.arange(3):
    exec('acc_column = df.acc_class%d' % i)
    print 'Class',i,'accuracy mean',acc_column.mean(),'std',acc_column.std()

# 4) Plot softmax score mean distribution for class 1 & 2 pixels.
#    Panda's DataFrame column can be accessed as a numpy array.
#    So you can give it to other apps like matplotlib easily.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8),facecolor='w')
plt.hist(df.mean_softmax_class1,range=(0,1.0),bins=20,color='red',alpha=0.5,label='Class1')
plt.hist(df.mean_softmax_class2,range=(0,1.0),bins=20,color='blue',alpha=0.5,label='Class2')
plt.tick_params(labelsize=20)
plt.xlabel('Softmax Probability Mean',fontsize=20,fontweight='bold')
plt.ylabel('Number of Events',fontsize=20,fontweight='bold')
leg=plt.legend(fontsize=16,loc=2)
leg_frame=leg.get_frame()
leg_frame.set_facecolor('white')
plt.grid()
fig.savefig('plot.png')
