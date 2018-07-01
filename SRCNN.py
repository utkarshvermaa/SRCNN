import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from utils import *

image_8x = "./IMG_8x"
image_8x_bi = "./IMG_8x_bi"
trnd_img = "./Trained_img"
result = "./result/"
log = "./logs/"

if not os.path.exists(log):
    os.mkdir(log)
if not os.path.exists(result):
    os.mkdir(result)
images = []
labels = []
test_im = []
overlap = 0

images = create_patch_normal(patchsize = 48, location = image_8x_bi, overlap = overlap)
labels = create_patch_normal(patchsize = 48, location = image_8x, overlap = overlap)
test_im = append_image(location = image_8x_bi)
print("test_im = ", np.array(test_im).shape)
print("images = ", np.array(images).shape)
print("labels = ", np.array(labels).shape)

#plt.imshow(images[1289])
#plt.show()
#plt.imshow(labels[1289])
#plt.show()
patch_height = patch_width = 48

totsize = np.array(images).shape[0]
images = np.array(images)
labels = np.array(labels)
test_im = np.array(test_im)
images = images/float(255)
labels = labels/float(255)
test_im = test_im/float(255)
images = np.reshape(images,(-1,patch_height,patch_width,1))
labels = np.reshape(labels,(-1,patch_height,patch_width,1))

epochs = 10
display_step = 10
learningrate = 0.0001
batch_size = 10
n1 = 64
n2 = 32
n3 = 1

weights = {
    'c1' : tf.Variable(tf.random_normal([9, 9, 1, n1], stddev = 0.1)),
    'c2' : tf.Variable(tf.random_normal([5, 5, n1, n2], stddev = 0.1)),
    'c3' : tf.Variable(tf.random_normal([5, 5, n2, n3], stddev = 0.1)),
}

biases = {
    'be1' : tf.Variable(tf.random_normal([n1], stddev = 0.1)),
    'be2' : tf.Variable(tf.random_normal([n2], stddev = 0.1)),
    'be3' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
}

def nextbatch(batch_i):
    global ll
    global hl
    
    ll = batch_i*batch_size
    ul = batch_i*batch_size + (batch_size)
    #print (ll)
    #print (ul)
    #if ul < totsize
    tempx = images[ll:ul].copy()
    #print('tempx_data shape:', np.array(tempx).shape)
    tempy = labels[ll:ul].copy()
    #print('tempy_data shape:', np.array(tempy).shape)
    """
    test_tempx = np.reshape(tempx[25],(patch_size,patch_size))
    test_tempy = np.reshape(tempy[25],(patch_size,patch_size))
    plt.imshow(test_tempx)
    plt.show()
    plt.imshow(test_tempy)
    plt.show()
    """

    #print tempnoisy.shape
    #tempx = tempx.reshape(batch_size, patch_width*patch_height)
    #tempy = tempy.reshape(batch_size, patch_width*patch_height)
    #print(tempy)
    #print(tempx)
    #ll = ll+incr
    return tempy, tempx

inputs = tf.placeholder(tf.float32, shape = (None,None,None,1), name = 'inputs')
targets = tf.placeholder(tf.float32, shape = (batch_size,patch_height,patch_width,1), name = 'targets')
keepprob = tf.placeholder(tf.float32)

def cae (_X, _W, _b, _keepprob, reuse=False):
    _ce1 = tf.nn.relu(tf.add(tf.nn.conv2d(_X, _W['c1'], strides = [1,1,1,1], padding='SAME'), _b['be1']))
    _ce1 = tf.nn.dropout(_ce1, _keepprob)
    _ce2 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce1, _W['c2'], strides = [1,1,1,1], padding='SAME'), _b['be2']))
    _ce2 = tf.nn.dropout(_ce2, _keepprob)
    _ce3 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce2, _W['c3'], strides = [1,1,1,1], padding='SAME'), _b['be3']))
    _ce3 = tf.nn.dropout(_ce3, _keepprob)
    return _ce3

pred = cae(inputs,weights,biases,keepprob)
cost = tf.reduce_mean(tf.square(pred - targets)) #tf.losses.mean_squared_error(targets,pred)#    
opti = tf.train.AdamOptimizer(learningrate).minimize(cost)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("Start Training")
l = 0
for epoch_i in range(epochs):
    num_batch = int(totsize/(batch_size))
    for batch_i in range(num_batch):
        batch_y, batch_x = nextbatch(batch_i) #batch_y is for noise and batch_x is for image
        #cv2.imshow("Image", small_true_batch[0].reshape(dimension1,dimension1))
        #cv2.waitKey(0)
        #sess.run(optms, feed_dict = {x:masked_batch, y_mid:small_true_batch, y:true_batch, keepprob:1.})
        sess.run(opti, feed_dict = {inputs:batch_x, targets:batch_y, keepprob:1})
        #sess.run(optm, feed_dict = {x:masked_batch,y_mid:small_true_batch, y: true_batch, keepprob:1.})
    print("[%02d/%02d] cost: %.10f" % (epoch_i, epochs, sess.run(cost, feed_dict={inputs : batch_x, targets: batch_y, keepprob :1})))
    if epoch_i % display_step == 0 or epoch_i == epochs - 1:
        saver.save(sess, "./logs/latestmodel.ckpt")
        k = test_im.shape[0]
        print("k = ", k)
        for i in range (k):
            im_test = test_im[i]
            im_test_height = im_test.shape[0]
            print(im_test_height)
            im_test_width = im_test.shape[1]
            print(im_test_width)
            recon = sess.run(pred, feed_dict = {inputs:im_test.reshape(1,im_test_height,im_test_width,1), keepprob:1})
            reshape = np.reshape(recon,(im_test_height,im_test_width))
            #plt.imshow(reshape)
            #plt.show()
            #recreated = deprocess(reshape)
            recreated = reshape*255
            print(np.array(recreated).shape)
            cv2.imwrite(result+str(i)+"-im"+str(l)+".png", recreated)
        l = l+1



































