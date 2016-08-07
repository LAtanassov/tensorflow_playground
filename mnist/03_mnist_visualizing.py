# following this
# https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
# launch tensorboard --logdir=path/to/log-directory

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/mnist_data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# need to much memory
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)



def nn_layer(input_tensor, input_dim, output_dim, layer_name, act_fn=tf.nn.relu):

    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act_fn(preactivate, 'activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return activations


def train():

    mnist = input_data.read_data_sets('/tmp/mnist_data', one_hot=True)
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y")

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', x_image, 10)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act_fn=tf.nn.softmax)


    with tf.name_scope('cross_entropy'):
        diff = y_ * tf.log(y)
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()

    for i in range(FLAGS.max_steps):
        xs, ys = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_summary, _, train_acc = sess.run([merged, train_step, accuracy], feed_dict={x: xs, y_: ys, keep_prob: 0.5})
            test_summary, test_acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

            test_writer.add_summary(test_summary, i)
            train_writer.add_summary(train_summary, i)

            print('accuracy at step %s: (test) %s (training) %s' % (i, test_acc, train_acc))

        _ = sess.run([train_step], feed_dict={x: xs, y_: ys, keep_prob: 0.5})

    train_writer.close()
    test_writer.close()



def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()


if __name__ == '__main__':
  tf.app.run()


