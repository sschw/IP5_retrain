
import re
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
                            
tf.app.flags.DEFINE_string('retrain_processed_data_dir', 
                            os.path.join(os.path.dirname(__file__),
                              os.pardir, os.pardir,
                              'data', 'retrain', 
                              'processed'), 
                            """Data Directory containing the category folders""")

tf.app.flags.DEFINE_string('retrain_bottleneck_data_dir', 
                            os.path.join(os.path.dirname(__file__),
                              os.pardir, os.pardir,
                              'data', 'retrain', 
                              'tfrecords'),
                            """Data Directory containing the category folders""")
                           
tf.app.flags.DEFINE_string('train_dir', '/tmp/ip5wke_retrain',
                           """Directory where to write event logs """
                           """and checkpoint.""")
                           
tf.app.flags.DEFINE_string('output_dir', '/tmp/ip5wke_retrain_output',
                           """Directory where to export inference model.""")

tf.app.flags.DEFINE_string('eval_dir', '/tmp/model_reeval',
                           """Directory where to write event logs.""")
                            
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('batch_size', 22,
                            """Number of images to process in a batch.""")
                            
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20  # Epochs after which learning rate decays.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000 # Examples that are at least take of a batch.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.000046 # Initial learning rate. 0.00007
WEIGHT_DECAY = 0.0000049
ADAM_EPSILON = 0.0001

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Tensor where we split the graph into 2 subgraphes.
# The second subgraph will be removed, retrained and readded.
BOTTLENECK_TENSOR_NAME = 'local4/local4:0'

# The input tensor of the graph that we use to get the bottleneck values.
BOTTLENECK_INPUT_TENSOR_NAME = 'map/TensorArrayStack/TensorArrayGatherV3:0'

# The input in the saved graph. Needed for providing a working saved graph.
INPUT_TENSOR_NAME = 'tf_example:0'
# The jpg parser in the saved graph. Needed for providing a working saved graph.
JPEGS_TENSOR_NAME = 'ParseExample/ParseExample:0'

def num_of_classes():
    """Returns the number of groups that are available.
    Returns the highest folder number + 1
    Returns:
      Integer value"""
    reference_dir = os.path.join(FLAGS.retrain_processed_data_dir, 'train')
    
    dir_list = [d for d in os.listdir(reference_dir) 
                  if os.path.isdir(os.path.join(reference_dir, d))]
    dir_int_list = [int(d) for d in dir_list]
    return max(dir_int_list) + 1
    

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, connections, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer(dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
    

def softmax(input, output_size):
    """Build the softmax layer for ip5wke model and retraining
    Args:
      input: Matrix. Output of local5 layer.
    Returns:
      softmax_linear: matrix
    """

    # local5
    with tf.variable_scope('local5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 100],
                                              connections=4096 + 100,
                                              wd=WEIGHT_DECAY)
        bias = batch_norm_wrapper(tf.matmul(input, weights))
        local5 = tf.nn.elu(bias, name=scope.name)
        local5 = tf.nn.dropout(local5, FLAGS.dropout_keep_probability)
        _activation_summary(local5)
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [100, output_size],
                                              connections=100 + output_size,
                                              wd=0.0)
        softmax_linear = batch_norm_wrapper(tf.matmul(local5, weights))

        _activation_summary(softmax_linear)
        
    return softmax_linear
    

def loss_f(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)

    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection('accuracies', accuracy)

    curr_conf_matrix = tf.cast(
        tf.contrib.metrics.confusion_matrix(tf.argmax(logits, 1), labels,
                                            num_classes=num_of_classes()),
        tf.float32)
    conf_matrix = tf.get_variable('conf_matrix', dtype=tf.float32,
                                  initializer=tf.zeros(
                                      [num_of_classes(), num_of_classes()],
                                      tf.float32),
                                  trainable=False)

    # make old values decay so early errors don't distort the confusion matrix
    conf_matrix.assign(tf.multiply(conf_matrix, 0.97))

    conf_matrix = conf_matrix.assign_add(curr_conf_matrix)

    tf.summary.image('Confusion Matrix',
                     tf.reshape(tf.clip_by_norm(conf_matrix, 1, axes=[0]),
                                [1, num_of_classes(), num_of_classes(), 1]))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in ip5wke model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    accuracies = tf.get_collection('accuracies')
    for a in accuracies:
        tf.summary.scalar('accuracy', a)

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the
        # loss as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """Train ip5wke model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr, epsilon=ADAM_EPSILON)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
    
def batch_norm_wrapper(inputs, decay=0.999, shape=[0]):
    """ batchnormalization layer """
    epsilon = 1e-3
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if FLAGS.is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, shape, name="moments")
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale,
                                             epsilon, name="batch_norm")
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale,
                                         epsilon, name="batch_norm")