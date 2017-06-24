import tensorflow as tf

from util.layers.conventional_layers import fc_layer


def sequential_attention(name, tensor_in, attention_size, time_ordered=True):
    """
    Attention mechanism layer which reduces RNN outputs with Attention vector.
    :param name:
    :param tensor_in: hehe
    :param attention_size: output feature length
    :param time_ordered: True=[T,N,D], False=[N,T,D]
    :return: attended feature
    """
    with tf.variable_scope(name):
        if time_ordered:
            tensor_in = tf.transpose(tensor_in, perm=[1, 0, 2])

        input_size = tensor_in.shape
        sequence_length = input_size[1].value
        feature_length = input_size[2].value

        squeezed_data = tf.reshape(tensor_in, [-1, feature_length])
        fc_1 = tf.tanh(fc_layer('fc', squeezed_data, attention_size))
        u_variable = tf.get_variable('kernel_u', [attention_size], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.1))
        fc_u = tf.matmul(fc_1, tf.reshape(u_variable, [-1, 1]))

        softmax_numerator = tf.reshape(tf.exp(fc_u), [-1, sequence_length])
        softmax_denominator = tf.reshape(tf.reduce_sum(softmax_numerator, 1), [-1, 1])
        attentions = softmax_numerator / softmax_denominator

        return tf.reduce_sum(tensor_in * tf.reshape(attentions, [-1, sequence_length, 1]), 1), attentions


def sequential_attention_real(name, tensor_in, tensor_desc, out_length=150):
    """
    Attention mechanism layer which reduces RNN outputs with Attention vector.
    :param name:
    :param tensor_in: [T,N,D]
    :param tensor_desc: [T,N]
    :param out_length:
    :return: attended feature
    """
    with tf.variable_scope(name):
        input_size = tensor_in.shape
        batch_size = input_size[1].value
        sequence_length = input_size[0].value
        feature_length = input_size[2].value

        squeezed_data = tf.reshape(tensor_in, [batch_size * sequence_length, feature_length])
        fc_1 = tf.nn.tanh(fc_layer('fc_1', squeezed_data, output_dim=1))
        # fc_2 = fc_layer('fc_2', fc_1, output_dim=1)
        scores = tf.reshape(fc_1, [sequence_length, batch_size, 1])
        is_not_pad = tf.cast(tf.not_equal(tensor_desc, 0)[..., tf.newaxis], tf.float32)
        prob = tf.nn.softmax(scores, dim=0) * is_not_pad
        attentions = prob / tf.reduce_sum(prob, 0, keep_dims=True)

        # text_feat = tf.reduce_sum(attentions * tf.reshape(fc_1, [sequence_length, batch_size, out_length]), 0)
        text_feat = tf.reduce_sum(attentions * tensor_in, 0)

        return text_feat, tf.transpose(tf.squeeze(attentions, axis=2), perm=[1, 0])


if __name__ == '__main__':
    aa = tf.zeros([5, 10, 10])
    a = sequential_attention('hehe', aa, 62)
    print(str(a))
