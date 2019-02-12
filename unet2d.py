import tensorflow as tf
import tensorlayer as tl



w_init = tf.truncated_normal_initializer(stddev=0.01)
b_init = tf.constant_initializer(value=0.1)



def DownBlock2D(pool_layer,scope,channel):
    
    conv2 = tl.layers.Conv2dLayer(pool_layer, act=tf.nn.relu,
                                  shape=[3, 3, channel, channel*2], strides=[1, 1, 1, 1], padding='SAME',
                                  W_init=w_init, b_init=b_init, name=scope+'conv')

    print conv2.outputs

    conv2_1 = tl.layers.Conv2dLayer(conv2, act=tf.nn.relu,
                                    shape=[3, 3, channel*2, channel*2], strides=[1, 1, 1, 1], padding='SAME',
                                    W_init=w_init, b_init=b_init, name=scope+'conv2_1')

    print conv2_1.outputs

    pool2 = tl.layers.PoolLayer(conv2_1, ksize=[1,  2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME',
                                pool=tf.nn.max_pool, name=scope+'pool2')

    return pool2, conv2_1




def UpBlock2D(bottom, conv, scope, channel):

    dev1 = tl.layers.DeConv2d(
        bottom, n_filter=channel, strides=(2, 2), padding='SAME', name=scope+'upsample')

    print dev1.outputs

    deconv1_2 = tl.layers.ConcatLayer(
        [conv, dev1], concat_dim=3, name=scope+'concat1_2')

    print deconv1_2.outputs

    deconv1_3 = tl.layers.Conv2dLayer(deconv1_2, act=tf.nn.relu, shape=(
        3, 3, channel*2, channel), strides=[1, 1, 1, 1], padding='SAME',
        W_init=w_init, b_init=b_init,
        name=scope+'deconv1')

    deconv1_4 = tl.layers.Conv2dLayer(deconv1_3, act=tf.nn.relu, shape=[
                                      3, 3, channel, channel], strides=[1,  1, 1, 1], padding='SAME',
                                      W_init=w_init, b_init=b_init,
                                      name=scope+'deconv1_1')

    return deconv1_4





def u_net2d(x, batch_size, channel, out_channel=2,input_channel=1):
    print x.shape[0]
    print x.shape[1]
    print x.shape[2]

    net_in = tl.layers.InputLayer(x, name='input')

    print net_in.outputs

    conv1 = tl.layers.Conv2dLayer(net_in, act=tf.nn.relu,
                                  shape=[3, 3,  input_channel, channel], strides=[1,  1, 1, 1], padding='SAME',
                                  W_init=w_init, b_init=b_init, name='conv1')

    print conv1.outputs

    conv1_1 = tl.layers.Conv2dLayer(conv1, act=tf.nn.relu,
                                    shape=[3,  3, channel, channel], strides=[1, 1, 1, 1], padding='SAME',
                                    W_init=w_init, b_init=b_init, name='conv1_1')

    print conv1_1.outputs

    pool1 = tl.layers.PoolLayer(conv1_1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME',
                                pool=tf.nn.max_pool, name='pool1')

    print pool1.outputs

    pool2, conv2_1 = DownBlock2D(pool1, 'l2', channel)
    pool3, conv3_1 = DownBlock2D(pool2, 'l3', channel*2)

    bottom = tl.layers.Conv2dLayer(pool3, act=tf.nn.relu,
                                   shape=[3,  3, 4*channel, channel*8], strides=[1, 1, 1, 1], padding='SAME',
                                   W_init=w_init, b_init=b_init, name='bottom')

    bottom = tl.layers.Conv2dLayer(bottom, act=tf.nn.relu,
                                   shape=[3, 3,  8*channel, channel*8], strides=[1,  1, 1, 1], padding='SAME',
                                   W_init=w_init, b_init=b_init, name='bottom_2')

    print bottom.outputs

    deconv1_4 = UpBlock2D(bottom, conv3_1, 'l3', channel*4)
    deconv2_4 = UpBlock2D(deconv1_4, conv2_1, 'l2', channel*2)
    deconv3_4 = UpBlock2D(deconv2_4, conv1_1, 'l1', channel)
    print deconv1_4.outputs
    print deconv2_4.outputs
    print deconv3_4.outputs

    network = tl.layers.Conv2dLayer(deconv3_4,
                                    act=tf.identity,
                                    # [0]:foreground prob; [1]:background prob
                                    shape=[1, 1, channel, out_channel],
                                    strides=[ 1, 1, 1, 1],
                                    padding='SAME',
                                    W_init=w_init, b_init=b_init, name='softmax')
    print network.outputs

    outputs = network.outputs

    return network, outputs


def model_fun_SegAndMultiLms(batch_size,grid_size,dev,channel):
    with tf.device(dev):

        x = tf.placeholder(tf.float32, shape=[
                           batch_size, grid_size, grid_size,  1])
        y_ = tf.placeholder(tf.float32, shape=[
                            batch_size, grid_size, grid_size,  5*4+1])

                            

        u2d, outputs = u_net2d(x, batch_size, channel, out_channel=5*4+2)

        print 'out.shape', outputs

   

        lm_loss=lambda x,y : tf.losses.mean_squared_error(x,y,reduction=tf.losses.Reduction.SUM)/tf.reduce_sum(x*x)

        m1=lm_loss(y_[:,:,:,0:4],     outputs[:,:,:,0:4])
        m2=lm_loss(y_[:,:,:,4:8],     outputs[:,:,:,4:8])
        m3=lm_loss(y_[:,:,:,8:12],    outputs[:,:,:,8:12])
        m4=lm_loss(y_[:,:,:,12:16],   outputs[:,:,:,12:16])
        m5=lm_loss(y_[:,:,:,16:20],   outputs[:,:,:,16:20])


        guess = tf.nn.softmax(outputs[:,:,:,20:22])

        seg_metric = tl.cost.dice_coe(
            y_[:, :, :, 20], guess[:, :, :, 0], axis=[1, 2])




        cost = (m1+m2+m3+m4+m5)/5+1.0-seg_metric

        train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)


        metrics={'lm1':m1,'lm2':m2,'lm3':m3,'lm4':m4,'lm5':m5,'seg_dice':seg_metric}

    return {'out': outputs, 'inputs': (x, y_), 'cost': cost, 'net': u2d, 'top': train_op,'metrics':metrics}


def model_fun_MultiLmks(batch_size,grid_size,dev,channel):
    with tf.device(dev):

        x = tf.placeholder(tf.float32, shape=[
                           batch_size, grid_size, grid_size,  1])
        y_ = tf.placeholder(tf.float32, shape=[
                            batch_size, grid_size, grid_size,  3*2])

                            

        u2d, outputs = u_net2d(x, batch_size, channel, out_channel=2*3)

        print 'out.shape', outputs

   

        lm_loss=lambda x,y : tf.losses.mean_squared_error(x,y,reduction=tf.losses.Reduction.SUM)/tf.reduce_sum(x*x)

        m1=lm_loss(y_[:,:,:,0:3],     outputs[:,:,:,0:3])
        m2=lm_loss(y_[:,:,:,3:6],     outputs[:,:,:,3:6])
        
        cost = (m1+m2)

        train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)


        metrics={'lm1':m1,'lm2':m2}

    return {'out': outputs, 'inputs': (x, y_), 'cost': cost, 'net': u2d, 'top': train_op,'metrics':metrics}

def model_fun_seg(batch_size,grid_size,dev,channel):
    with tf.device(dev):

        x = tf.placeholder(tf.float32, shape=[
                           batch_size, grid_size, grid_size,  3])
        y_ = tf.placeholder(tf.float32, shape=[
                            batch_size, grid_size, grid_size,  1])

                            

        u2d, outputs = u_net2d(x, batch_size, channel, out_channel=2,input_channel=3)

        print 'out.shape', outputs

        guess = tf.nn.softmax(outputs[:,:,:,0:2])

        seg_metric = tl.cost.dice_coe(
            y_[:, :, :, 0], guess[:, :, :, 0], axis=[1, 2])

        cost=1.0-seg_metric

        
        train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)


        metrics={'seg':seg_metric}

    return {'out': outputs, 'inputs': (x, y_), 'cost': cost, 'net': u2d, 'top': train_op,'metrics':metrics}



if __name__ == "__main__":
    import numpy as np
    import time

    batch_size = 64
    grid_size = 128
    channel = 16

    ml = model_fun_SegAndMultiLms(batch_size, grid_size, '/gpu:0', channel)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    data = np.zeros((batch_size, grid_size,
                    grid_size, 1), dtype=np.float32)
    label = np.ones((batch_size, grid_size, 
                      grid_size, 4*5+1), dtype=np.float32)

    losslist=[]
    if 'metrics' in ml:
        print 'metrics exist'


        metrics=ml['metrics']
        kl=list(metrics.iterkeys())
        vl=list(metrics.itervalues())



    
    while True:
        t1=time.time()
        dtop, dcos = sess.run([ml['top'], ml['cost']], feed_dict={
                              ml['inputs'][0]: data, ml['inputs'][1]: label})
        t2=time.time()

        dm=sess.run(vl,feed_dict={
                              ml['inputs'][0]: data, ml['inputs'][1]: label})

      

        print 'time',t2-t1
        print dcos

        print dm 
        for i,j in  enumerate(dm):
            print kl[i],j                   


    print "finished"
