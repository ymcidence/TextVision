import tensorflow as tf

from util.layers import conventional_layers as layers


def net_generator(input_tensor):
    weights_initializer = tf.random_normal_initializer(stddev=0.02)
    biases_initializer = tf.constant_initializer(0.)
    fc_1 = layers.fc_layer('fc_0', input_tensor, 4 * 2 * 256, weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
    fc_1 = tf.reshape(fc_1, [-1, 4, 2, 256])
    fc_1 = tf.nn.relu(tf.layers.batch_normalization(fc_1, momentum=0.9, epsilon=1e-5))

    t_conv_1 = tf.layers.conv2d_transpose(fc_1, 512, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_1 = tf.nn.relu(tf.layers.batch_normalization(t_conv_1, momentum=0.9, epsilon=1e-5))
    t_conv_2 = tf.layers.conv2d_transpose(t_conv_1, 256, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_2 = tf.nn.relu(tf.layers.batch_normalization(t_conv_2, momentum=0.9, epsilon=1e-5))
    t_conv_3 = tf.layers.conv2d_transpose(t_conv_2, 128, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_3 = tf.nn.relu(tf.layers.batch_normalization(t_conv_3, momentum=0.9, epsilon=1e-5))
    t_conv_4 = tf.layers.conv2d_transpose(t_conv_3, 64, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_4 = tf.nn.relu(tf.layers.batch_normalization(t_conv_4, momentum=0.9, epsilon=1e-5))

    t_conv_5 = tf.layers.conv2d_transpose(t_conv_4, 3, 5, (2, 2), padding='SAME', activation=tf.nn.tanh,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    return t_conv_5


def net_discriminator(input_tensor, output_dim=1):
    def bn(in_tensor):
        return tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5)

    def leaky_relu(in_tensor, leak=0.2):
        return tf.maximum(in_tensor, leak * in_tensor)

    starting_out_dim = 64
    kernel_size = 5
    stride = 2
    conv_1 = layers.conv_relu_layer('conv_1', input_tensor, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim)
    conv_1 = leaky_relu(conv_1)
    conv_2 = layers.conv_relu_layer('conv_2', conv_1, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 2)
    conv_2 = leaky_relu(bn(conv_2))
    conv_3 = layers.conv_relu_layer('conv_3', conv_2, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 4)
    conv_3 = leaky_relu(bn(conv_3))
    conv_4 = layers.conv_relu_layer('conv_4', conv_3, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 6)
    conv_4 = leaky_relu(bn(conv_4))
    conv_5 = layers.conv_relu_layer('conv_5', conv_4, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 8)
    fc_d = layers.fc_layer('fc_d', conv_4, output_dim=output_dim)
    return fc_d


def net_fused_discriminator(input_image, input_text, output_dim=1):
    def bn(in_tensor):
        return tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5)

    def leaky_relu(in_tensor, leak=0.2):
        return tf.maximum(in_tensor, leak * in_tensor)

    starting_out_dim = 64
    kernel_size = 5
    stride = 2
    conv_1 = layers.conv_relu_layer('conv_1', input_image, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim)
    conv_1 = leaky_relu(conv_1)
    conv_2 = layers.conv_relu_layer('conv_2', conv_1, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 2)
    conv_2 = leaky_relu(bn(conv_2))
    conv_3 = layers.conv_relu_layer('conv_3', conv_2, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 4)
    conv_3 = leaky_relu(bn(conv_3))
    conv_4 = layers.conv_relu_layer('conv_4', conv_3, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 6)
    conv_4 = leaky_relu(bn(conv_4))

    conv_5 = layers.conv_relu_layer('conv_5', conv_4, kernel_size=3, stride=stride,
                                    output_dim=starting_out_dim * 8)
    conv_5 = leaky_relu(bn(conv_5))

    shape_5 = conv_5.shape
    batch_size = shape_5[0].value
    h_5 = shape_5[1].value
    w_5 = shape_5[2].value
    txt_rep_length = input_text.shape[1].value
    expend_text = tf.reshape(input_text, [batch_size, 1, 1, txt_rep_length])
    tiled_text = tf.tile(expend_text, [1, h_5, w_5, 1])

    conv_6_in = tf.concat([conv_5, tiled_text], axis=3)
    conv_6 = layers.conv_relu_layer('conv_6', conv_6_in, kernel_size=1, stride=1,
                                    output_dim=starting_out_dim * 8)
    conv_6 = leaky_relu(bn(conv_6))

    fc_d = layers.fc_layer('fc_d', conv_6, output_dim=output_dim)
    return fc_d
