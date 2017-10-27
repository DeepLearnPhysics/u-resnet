import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import uresnet_layers as L
from resnet_module import double_resnet

def build(input_tensor, num_class=4, base_num_filters=16, debug=False):

    net = input_tensor
    if debug: print(net.shape, 'input shape')

    # assume zero padding in each layer (set as default in uresnet_layers.py)

    # downsampling path
    num_filters = base_num_filters

    # feature maps saved for later upsampling layer
    feature_map = {}

    # initial 7x7 convolution
    net = L.conv2d(input_tensor=net, name='conv7x7_begin', kernel=(7,7), stride=(1,1), num_filter=num_filters, activation_fn=tf.nn.relu)
    # save for concatenation with upsampling0 later
    feature_map[(net.get_shape()[-2].value,net.get_shape()[-3].value)] = net
    if debug: print(net.shape, "after conv7x7_begin")

    # downsampling path
    if debug: print(net.shape, "before downsampling")
    step = 0
    for _ in xrange(5):
        step += 1
        num_filters *= 2
        net = double_resnet(input_tensor=net, dim_factor=2, num_filters=num_filters,  step=step)
        fmap_key = (net.get_shape()[-2].value,net.get_shape()[-3].value)
        feature_map[fmap_key] = net
        if debug: print(net.shape, "after downsampling step", step)

    # upsampling path
    if debug: print(net.shape, "before upsampling")
    for _ in xrange(4):
        step += 1
        num_filters /= 2
        # deconvolution
        net = L.deconv2d(input_tensor=net, name='deconv_%d' % step, kernel=(3,3), stride=(2,2), output_num_filter=num_filters)
        if debug: print(net.shape, "after upsampling deconv", step)

        # concatenate feature map
        fmap_key = (net.get_shape()[-2].value,net.get_shape()[-3].value)
        upsampling   = net
        downsampling = feature_map[fmap_key]
        net = tf.concat([downsampling, upsampling], axis=3)
        if debug: print(net.shape, "after upsampling concat", step)    

        # 2x resnet indexed 5: input is 32 x 32 x (512+512)
        net = double_resnet(input_tensor=net, dim_factor=1, num_filters=num_filters,  step=step)
        if debug: print(net.shape, "after upsampling", step)
    
    # deconvolution 4
    net = L.deconv2d(input_tensor=net, name='deconv%d' % (step+1), kernel=(3,3), stride=(2,2), output_num_filter=base_num_filters)
    if debug: print(net.shape, "after upsampling deconv", (step+1))

    # concatenate downsampling0 and upsampling0
    upsampling = net
    fmap_key = (net.get_shape()[-2].value,net.get_shape()[-3].value)
    downsampling = feature_map[fmap_key]
    net = tf.concat([downsampling, upsampling], axis=3)

    # first of two  7x7 conv at the end
    net = L.conv2d(input_tensor=net, name='conv7x7_end1', kernel=(7,7), stride=(1,1), num_filter=base_num_filters, activation_fn=tf.nn.relu)
    if debug: print(net.shape, "after conv7x7_end1")

    # second of two 7x7 conv at the end
    net = L.conv2d(input_tensor=net, name='conv7x7_end2', kernel=(7,7), stride=(1,1), num_filter=num_class, activation_fn=tf.nn.relu)
    if debug: print(net.shape, "after conv7x7_end2")

    return net

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,512,512,1])
    net = build(input_tensor=x,debug=True)
    
