import tensorflow as tf
from util.layers.conventional_layers import fc_layer


def sequential_attention(tensor_in, attention_size, time_ordered=True):
    """
    Attention mechanism layer which reduces RNN outputs with Attention vector.
    :param tensor_in: hehe
    :param attention_size: output feature length
    :param time_ordered: True=[T,N,D], False=[N,T,D]
    :return: attended feature
    """
    with tf.variable_scope('s_attention'):
        if time_ordered:
            tensor_in = tf.transpose(tensor_in, perm=[1, 0, 2])

        input_size = tensor_in.shape
        sequence_length = input_size[1].value
        feature_length = input_size[2].value

        squeezed_data = tf.reshape(tensor_in, [-1, feature_length])
        fc_1 = tf.tanh(fc_layer('fc', squeezed_data, attention_size))
        u_variable = tf.get_variable('u', [attention_size], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.1))
        fc_u = tf.matmul(fc_1, tf.reshape(u_variable, [-1, 1]))

        softmax_numerator = tf.reshape(tf.exp(fc_u), [-1, sequence_length])
        softmax_denominator = tf.reshape(tf.reduce_sum(softmax_numerator, 1), [-1, 1])
        attentions = softmax_numerator / softmax_denominator

        return tf.reduce_sum(tensor_in * tf.reshape(attentions, [-1, sequence_length, 1]), 1)


if __name__ == '__main__':
    aa = tf.zeros([5, 10, 10])
    a = sequential_attention(aa, 62)
    print(str(a))
