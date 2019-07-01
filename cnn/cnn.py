import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

logPath = "logs/"

def generate_summaries(var):
    """
    Generate summaries for a given variable, including mean, standard deviation, max, min and histogram
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# get MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# get input placeholders
with tf.name_scope("MNIST_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")  

# reshape images to 28X28X1
with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

# initialize weights
def init_weights(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# initialize bias
def init_bias(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# convolution
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)

# max-pool
def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

# Model: 2 Conv-Max Pool layers followed by a FC layer

with tf.name_scope('Conv1'):
    with tf.name_scope('weights'):
        # 32 convolutions of size 5x5
        W_conv1 = init_weights([5, 5, 1, 32], name="weight")
        generate_summaries(W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = init_bias([32], name="bias")
        generate_summaries(b_conv1)
    # Convolution + Bias -> RELU -> MAX Pool
    conv1_wx_b = conv2d(x_image, W_conv1,name="conv2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name="relu")
    tf.summary.histogram('h_conv1', h_conv1)
    h_pool1 = max_pool_2x2(h_conv1, name="pool")

with tf.name_scope('Conv2'):
    with tf.name_scope('weights'):
        # 64 convolutions
        W_conv2 = init_weights([5, 5, 32, 64], name="weight")
        generate_summaries(W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = init_bias([64], name="bias")
        generate_summaries(b_conv2)
    # Convolution + Bias -> RELU -> MAX Pool
    conv2_wx_b = conv2d(h_pool1, W_conv2, name="conv2d") + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2_wx_b, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2, name="pool")

with tf.name_scope('FC'):
    W_fc1 = init_weights([7 * 7 * 64, 1024], name="weight")
    b_fc1 = init_bias([1024], name="bias")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

# Dropout on FC
keep_prob = tf.placeholder(tf.float32, name="keep_prob") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("Output"):
    W_fc2 = init_weights([1024, 10], name="weight")
    b_fc2 = init_bias([10], name="bias")

# Y = WX + b
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Cross Entropy loss
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Adam Optimizer
with tf.name_scope("adam_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scalar", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

# tensorboard - merge summaries
summarize_all = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

# write the default graph
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Train the model
steps = 2000
counter = 100

# Start timer
start_time = time.time()
end_time = time.time()
for i in range(steps):
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})


    # Display status
    if i % counter == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
        # write summary to log
        tbWriter.add_summary(summary,i)


end_time = time.time()
print("Total training time for {0} steps: {1:.2f} seconds".format(i+1, end_time-start_time))
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))
