import tensorflow as tf


def lstm_layer(name, tensor_in, out_length, num_layers, ):
    def build_lstm(_out_length):
        lstm_cell = tf.contrib.rnn.LSTMCell(_out_length, initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                            forget_bias=1.)
        return lstm_cell

    with tf.variable_scope(name):
        if num_layers > 1:
            grouped_lstm_cell = tf.contrib.rnn.MultiRNNCell([build_lstm(out_length) for _ in range(num_layers)])
        else:
            grouped_lstm_cell = build_lstm(out_length)
        lstm_out, _ = tf.nn.dynamic_rnn(grouped_lstm_cell, tensor_in, dtype=tf.float32, time_major=True)
        lstm_out = tf.concat(lstm_out, axis=2)
    return lstm_out


if __name__ == '__main__':
    # test script
    aa = tf.zeros([5, 10, 10])
    a1 = lstm_layer('hehe', aa, 20, 1)
    print('hehe')
