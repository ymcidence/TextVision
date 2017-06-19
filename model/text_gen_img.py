import tensorflow as tf
import model.abc_net as an
from six.moves import xrange
from util.layers import conventional_layers as layers
from util.layers import attention
from util.layers import nets
from util.layers import lstm_layer as lstm

NAME_SCOPE_ATTENTION = 'attention_net'
SUB_SCOPE_TEXT = 'text'
SUB_SCOPE_JOINT = 'joint'


class TextGenImage(an.AbstractNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs.get('batch_size')
        self.log_path = kwargs.get('log_path')
        self.seq_length = kwargs.get('seq_length')
        self.dict_size = kwargs.get('dict_size')
        if kwargs.get('emb_size') is not None:
            self.emb_size = kwargs.get('emb_size')
        else:
            self.emb_size = 256
        if kwargs.get('emb_lstm') is not None:
            self.emb_lstm = kwargs.get('emb_lstm')
        else:
            self.emb_lstm = 256
        self.fused_discriminator = True
        self.sampled_variables = tf.placeholder(tf.float32, [self.batch_size, 100])
        self.batch_image = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])
        self.batch_sentence = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])  # [N,T]
        self.nets = self._build_net()
        self.loss = self._build_loss()

    def _build_net(self):
        # text attention
        with tf.variable_scope(NAME_SCOPE_ATTENTION + '/' + SUB_SCOPE_TEXT):
            emb_sentence = layers.emb_layer('word_emb', self.batch_sentence, self.dict_size, self.emb_size)
            lstm_sentence = lstm.lstm_layer('lstm', emb_sentence, out_length=self.emb_lstm, num_layers=2)
            feat_sbj = attention.sequential_attention('att_1', lstm_sentence, self.emb_size / 2)
            feat_rel = attention.sequential_attention('att_2', lstm_sentence, self.emb_size / 2)
            feat_obj = attention.sequential_attention('att_3', lstm_sentence, self.emb_size / 2)
            image_net_in = tf.concat([self.sampled_variables, feat_sbj, feat_rel, feat_obj], axis=1)
        # generator
        with tf.variable_scope(an.NAME_SCOPE_GENERATIVE_NET):
            fake_image = nets.net_generator(image_net_in)
        # discriminator
        if self.fused_discriminator:
            feat_sentence = tf.concat([feat_sbj, feat_rel, feat_obj], axis=1)
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET):
                fake_decision = nets.net_fused_discriminator(fake_image, feat_sentence)
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
                real_decision = nets.net_fused_discriminator(self.batch_image, feat_sentence)
        else:
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET):
                fake_decision = nets.net_discriminator(fake_image)
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
                real_decision = nets.net_discriminator(self.batch_image)

        tf.summary.image(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im', fake_image)
        tf.summary.image(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im', self.batch_image)
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_dec_hist', fake_decision)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/dis_dec_hist', real_decision)
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im_hist', (fake_image + 1) / 2 * 255.)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im_hist', self.batch_image)
        return fake_image, fake_decision, real_decision

    def _build_loss(self):
        loss_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[2]), logits=self.nets[2]))
        loss_dis = loss_dis_fake + loss_dis_real

        tf.summary.scalar(an.NAME_SCOPE_GENERATIVE_NET + '/loss', loss_gen)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss', loss_dis)

        return loss_gen, loss_dis

    def _build_opt(self):
        trainer1 = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        trainer2 = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=an.NAME_SCOPE_GENERATIVE_NET)
        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=an.NAME_SCOPE_DISCRIMINATIVE_NET)
        train_list_att_txt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=NAME_SCOPE_ATTENTION + '/' + SUB_SCOPE_TEXT)
        train_list_gen = train_list_gen + train_list_att_txt
        train_list_dis = train_list_dis + train_list_att_txt
        op_gen = trainer1.minimize(self.loss[0], var_list=train_list_gen, global_step=self.g_step)
        op_dis = trainer2.minimize(self.loss[1], var_list=train_list_dis, global_step=self.g_step)

        return op_gen, op_dis

    def train(self, max_iter, dataset):
        for i in xrange(max_iter):
            a = 1
