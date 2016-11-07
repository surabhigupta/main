import numpy as np
import tensorflow as tf

class AddTwo(object):
    def __init__(self):
        # If you are constructing more than one graph within a Python kernel
        # you can either tf.reset_default_graph() each time, or you can
        # instantiate a tf.Graph() object and construct the graph within it.

        # Hint: Recall from live sessions that TensorFlow
        # splits its models into two chunks of code:
        # - construct and keep around a graph of ops
        # - execute ops in the graph
        #
        # Construct your graph in __init__ and run the ops in Add.
        #
        # We make the separation explicit in this first subpart to
        # drive the point home.  Usually you will just do them all
        # in one place, including throughout the rest of this assignment.
        #
        # Hint:  You'll want to look at tf.placeholder and sess.run.

        # START YOUR CODE
        self.X1 = tf.placeholder(tf.float64)
        self.Y1 = tf.placeholder(tf.float64)
        
        self.graph = self.X1 + self.Y1
        # END YOUR CODE

    def Add(self, x, y):
        # START YOUR CODE
        sess = tf.Session()
        return sess.run(self.graph, feed_dict={self.X1: x, self.Y1: y})
        # END YOUR CODE

def affine_layer(hidden_dim, x, seed=0):
    # x: a [batch_size x # features] shaped tensor.
    # hidden_dim: a scalar representing the # of nodes.
    # seed: use this seed for xavier initialization.
    
    W = tf.get_variable("W", shape=[x.get_shape()[1],hidden_dim], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    graph = tf.matmul(x,W)
    b = tf.zeros_like(graph)
    return graph + b

def fully_connected_layers(hidden_dims, x):
    # hidden_dims: A list of the width of the hidden layer.
    # x: the initial input with arbitrary dimension.
    # To get the tests to pass, you must use relu(.) as your element-wise nonlinearity.
    #
    # Hint: see tf.variable_scope - you'll want to use this to make each layer 
    # unique.
    # Hint: a fully connected layer is a nonlinearity of an affine of its input.
    #       your answer here only be a couple of lines long (mine is 4).

    # START YOUR CODE
    print 'Hidden dims: %r, x: %r' % (hidden_dims, x)
    layer = x
    # hidden_dims: [10, 20, 100, 1]
    # x : 1 X 3, W : 3 X 10, layer : 1 X 10
    # x : 10 X 3, W : 3 X 20, layer : 10 X 20
    # x : 20 X 3, W : 3 X 100, layer : 20 X 100
    # x : 100 X 3, W : 3 X 1, layer : 100 X 1   
    for index, hidden_dim in enumerate(hidden_dims):
        with tf.variable_scope("MyLayer" + str(index)):
            layer = tf.nn.relu(affine_layer(hidden_dim, layer))

    return layer
    # END YOUR CODE

def train_nn(X, y, X_test, hidden_dims, batch_size, num_epochs, learning_rate,
             verbose=False):
    # Train a neural network consisting of fully_connected_layers
    # to predict y.  Use sigmoid_cross_entropy_with_logits loss between the
    # prediction and the label.
    # Return the predictions for X_test.
    # X: train features
    # Y: train labels
    # X_test: test features
    # hidden_dims: same as in fully_connected_layers
    # learning_rate: the learning rate for your GradientDescentOptimizer.

    print "hidden_dims: %r, batch_size: %r, num_epochs: %r, learning_rate: %r" % (hidden_dims, batch_size, num_epochs, learning_rate)
    
    # Construct the placeholders.
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=[None, X.shape[-1]])
    y_ph = tf.placeholder(tf.float32, shape=[None])
    global_step = tf.Variable(0, trainable=False)
    
    # Construct the neural network, store the batch loss in a variable called `loss`.
    # At the end of this block, you'll want to have these ops:
    # - y_hat: probability of the positive class
    # - loss: the average cross entropy loss across the batch
    #   (hint: see tf.sigmoid_cross_entropy_with_logits)
    #   (hint 2: see tf.reduce_mean)
    # - train_op: the training operation resulting from minimizing the loss
    #             with a GradientDescentOptimizer
    #
    # Hint:  Remember that a neural network has the form
    #        <affine-nonlinear>* -> affine -> sigmoid -> y_hat
    #        Double check your code works for 0..n affine-nonlinears.
    #
    # START YOUR CODE

    y_hat = fully_connected_layers(hidden_dims, x_ph)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.reduce_mean(y_hat, 1), y_ph)
    train_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, )
                                                                           
    # END YOUR CODE


    # Output some initial statistics.
    # You should see about a 0.6 initial loss (-ln 2).
    sess = tf.Session(config=tf.ConfigProto(device_filters="/cpu:0"))
    sess.run(tf.initialize_all_variables())
    
    print 'Initial loss:', sess.run(loss, feed_dict={x_ph: X, y_ph: y})

    if verbose:
        for var in tf.trainable_variables():
            print 'Variable: ', var.name, var.get_shape()
            print 'dJ/dVar: ', sess.run(
                  tf.gradients(loss, var), feed_dict={x_ph: X, y_ph: y})

    for epoch_num in xrange(num_epochs):
        print 'Epoch num: ', epoch_num,

        for batch in xrange(0, X.shape[0], batch_size):
            X_batch = X[batch : batch + batch_size]
            y_batch = y[batch : batch + batch_size]

            # Feed a batch to your network using sess.run.
            # Populate loss_value with the current value of loss.
            # Populate global_value with the current value of global_step.
            # You'll also want to run your training op.
            # START YOUR CODE
            
            loss_value = sess.run(loss, feed_dict={x_ph: X_batch, y_ph: y_batch})
            global_step_value = sess.run(global_step)
            opt_op = train_opt.minimize(loss, global_step=global_step, var_list=tf.trainable_variables())
            opt_op.run(session=sess, feed_dict={x_ph: X_batch, y_ph: y_batch})
            
            # END YOUR CODE
        if epoch_num % 100 == 0:
            print 'Step: ', global_step_value,'Loss:', loss_value
            if verbose:
                for var in tf.trainable_variables():
                    print var.name, sess.run(var)
                print ''
    # Return your predictions.
    # START YOUR CODE
    pass
    # END YOUR CODE
