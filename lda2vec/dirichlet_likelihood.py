import tensorflow as tf

def dirichlet_likelihood(weights, alpha=None):
    n_topics = weights.get_shape()[1].value
    if alpha is None:
        alpha = 1.0 / n_topics
    log_proportions = tf.nn.log_softmax(weights)
    loss = (alpha - 1.0) * log_proportions
    return tf.reduce_sum(loss)


def pseudo_dirichlet(weights, alpha=None):
    n_topics = weights.get_shape()[1].value
    n_docs = weights.get_shape()[0].value
    if alpha is None:
        alpha = n_docs / n_topics
    proportions = tf.reduce_sum(tf.nn.softmax(weights),axis=0)
    loss = tf.norm(proportions - alpha, ord=2)
    return loss