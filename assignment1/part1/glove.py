import tensorflow as tf

def wordids_to_tensors(wordids, embedding_dim, vocab_size, seed=0):
    '''Convert a list of wordids into embeddings and biases.

    This function creates a variable w for the embedding matrix, dimension |E x V|
    and a variable b to hold the biases, dimension |V|.

    It returns an op that will accept the output of "wordids" op and lookup
    the corresponding embedding vector and bias in the table.

    Args:
      - wordids |IDs|: a tensor of wordids
      - embedding_dim, |E|: a scalar value of the # of dimensions in which to embed words
      - vocab_size |V|: # of terms in the vocabulary

    Returns:
      A tuple (ws, bs, m), all with tf.float32s:
        - ws |IDs x E| is a tensor of word embeddings (looked up from w)
        - bs |IDs| is a vector of biases (looked up from b)
        - w |V x E| is the full embedding matrix (from which you looked up ws)
          (we don't strictly need this, but it comes in handy in the "Play!" section.)

    HINT: To get the tests to pass, initialize w with a random_uniform [-1, 1] using
          the provided seed.  As usual the "b" vector should be initialized to 0.

    HINT: Look at tf.embedding_lookup(.).
    '''
    # START YOUR CODE
    ws = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0, seed=seed), dtype=tf.float32)
    bs = tf.zeros_like(wordids, dtype=tf.float32)
    w = tf.nn.embedding_lookup([ws], wordids)
    
    return w, bs, ws

    # END YOUR CODE


def example_weight(Xij, x_max, alpha):
    '''Scale the count according to Equation (9) in the Glove paper.

    This runs as part of the TensorFlow graph.  You must do this with
    TensorFlow ops.

    Args:
      - Xij: a |batch| tensor of counts.
      - x_max: a scalar, see paper.
      - alpha: a scalar, see paper.

    Returns:
      - A vector of corresponding weights.
    '''
    # START YOUR CODE

    if x_max == 0:
        return
    
    condition = Xij < x_max
    product = Xij/x_max

    case1 = pow(product, alpha)
    case2 = tf.ones_like(condition, dtype=tf.float32)
    return tf.select(condition, case1, case2)
        
    # END YOUR CODE


def loss(w, b, w_c, b_c, c):
    '''Compute the loss for each of training examples.

    Args:
      - w |batch_size x embedding_dim|: word vectors for the batch
      - b |batch_size|: biases for these words
      - w_c |batch_size x embedding_dim|: context word vectors for the batch
      - b_c |batch_size|: biases for context words
      - c |batch_size|: # of times context word appeared in context of word

    Returns:
      - loss |batch_size|: the loss of each example in the batch
    '''
    # START YOUR CODE
    
    product = tf.matmul(w, tf.transpose(w_c))
    loss = tf.square(product + b + b_c - tf.log(c))
    weight = example_weight(c, 100.0, 0.75)
    weighted_loss = weight * loss
    reduced_loss = tf.reduce_min(weighted_loss, 1)

    return reduced_loss
        
    # END YOUR CODE
