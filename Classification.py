import tensorflow as tf
import os
import math
import numpy
import numpy as np
import random
from PIL import Image
sess = tf.InteractiveSession()

IMAGE_SIZE = 32
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
train_images = []

def get_image_paths_in_folder(folder_name):
  image_paths = [os.path.join(folder, pic)
      for folder, subs, pics, in os.walk(".")
      for pic in pics if pic.endswith(".jpg") and folder.startswith(folder_name)]
  return image_paths

cat_image_paths = get_image_paths_in_folder("./cats")
dog_image_paths = get_image_paths_in_folder("./dogs")
flower_image_paths = get_image_paths_in_folder("./flowers")
test_image_paths = get_image_paths_in_folder("./test")

## Load images ##
print("loading cat images...")
for filename in cat_image_paths:  
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  normalize_img = np.array(image)
  normalize_img = normalize_img / 255.0
  train_images.append(normalize_img)
print("loading dog images...")
for filename in dog_image_paths:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  normalize_img = np.array(image)
  normalize_img = normalize_img / 255.0
  train_images.append(normalize_img)
print("loading flower images...")
for filename in flower_image_paths:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  normalize_img = np.array(image)
  normalize_img = normalize_img / 255.0
  train_images.append(normalize_img)

NUMBER_OF_CAT_IMAGES = len(cat_image_paths)
NUMBER_OF_DOG_IMAGES = len(dog_image_paths)
NUMBER_OF_FLOWER_IMAGES = len(flower_image_paths)
NUMBER_OF_INPUTS = NUMBER_OF_CAT_IMAGES + NUMBER_OF_DOG_IMAGES + NUMBER_OF_FLOWER_IMAGES
train_images = np.array(train_images,dtype=np.float32)
train_images = train_images.reshape(NUMBER_OF_INPUTS,IMAGE_PIXELS)

# Define labels of training data ##
train_labels = np.zeros(shape=(NUMBER_OF_INPUTS,3))
for i in range(NUMBER_OF_INPUTS):
    if i < NUMBER_OF_CAT_IMAGES:
        train_labels[i] = [1,0,0] #cats
    elif i < NUMBER_OF_CAT_IMAGES + NUMBER_OF_DOG_IMAGES:
        train_labels[i] = [0,1,0] #dogs
    else:
        train_labels[i] = [0,0,1] #flowers

## Shuffle the train_data ##
for i in train_images:
    index1 = random.randint(0,NUMBER_OF_INPUTS-1)
    index2 = random.randint(0,NUMBER_OF_INPUTS-1)
    tmp = train_images[index1]
    train_images[index1] = train_images[index2]
    train_images[index2] = tmp
    tmp = train_labels[index1]
    train_labels[index1] = train_labels[index2]
    train_labels[index2] = tmp

np.save("cat_dog_flower_data", train_images)
np.save("cat_dog_flower_label", train_labels)
# train_images = np.load("cat_dog_data.npy")
# train_labels = np.load("cat_dog_label.npy")
print("train_labels shape: ")
print(train_labels.shape)
print("train_images shape: ")
print(train_images.shape)

## Training data loading complete ##

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #stride [1,x_movement, y_movement,1]
    #Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    #stride[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


## Define placeholder for inputs to network ##
xs = tf.placeholder(tf.float32,[None,1024*3])  #32*32*3 pixels (features)
ys = tf.placeholder(tf.float32,[None,3]) #output 0~1
x_image = tf.reshape(xs,[-1,32,32,3])

## Conv1 layer ##
W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## Conv2 layer ##
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## Fully connect layer ##
W_fc1 = weight_variable([8*8*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

## Drop out ##
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## Output layer ##
W_fc2 = weight_variable([1024,3])
b_fc2 = bias_variable([3])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

## A method to compute loss, so does the mean square error method ##
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init = tf.initialize_all_variables()
sess.run(init)


## Training start ##
batch_start = 0
batch_end = 0
batch_size = 64
for i in range(1000):
    batch_end = batch_start+batch_size
    if batch_end > NUMBER_OF_INPUTS:
        batch_end = NUMBER_OF_INPUTS
    batch_xs = train_images[batch_start:batch_end]
    batch_ys = train_labels[batch_start:batch_end]
    if i%50 == 0:
        print("%d  to %d"%(batch_start,batch_end))
        train_accuracy = accuracy.eval(feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1.0})
        print("step %d, training accuracy %g"%(i,train_accuracy))
    batch_start = (batch_end + 1)%NUMBER_OF_INPUTS    
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.6})

## Save Network ##
saver = tf.train.Saver()
save_path = saver.save(sess,"my_net/cat_dog_one_loop.ckpt")
print("Save to path:",save_path)


# Define testing data
test_images = []
for filename in test_image_paths:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  normalize_img = np.array(image)
  normalize_img = normalize_img / 255.0
  test_images.append(normalize_img)
test_images = np.array(test_images,dtype=np.float32)
test_images = test_images.reshape(20,IMAGE_PIXELS)
# test_labels = np.zeros(shape=(NUMBER_OF_INPUTS,3))
# test_labels = [[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
test_predict = tf.argmax(y_conv,1)
test_ans = tf.argmax(ys,1)
test_result = test_predict.eval(feed_dict={xs:test_images,keep_prob:1.0})
for i in test_result:
    print i
# test_accuracy = accuracy.eval(feed_dict={xs:test_images,ys:test_labels,keep_prob:1.0})
# print("test accuracy %g"%(test_accuracy))









