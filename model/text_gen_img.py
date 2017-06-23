import gc
import os

import numpy as np
import tensorflow as tf
from six.moves import xrange

import model.abc_net as an
from util.layers import attention
from util.layers import conventional_layers as layers
from util.layers import lstm_layer as lstm
from util.layers import nets

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
        self.emb_file = kwargs.get('emb_file')
        if kwargs.get('emb_size') is not None:
            self.emb_size = kwargs.get('emb_size')
        else:
            self.emb_size = 300
        if kwargs.get('emb_lstm') is not None:
            self.emb_lstm = kwargs.get('emb_lstm')
        else:
            self.emb_lstm = 300
        self.fused_discriminator = True
        self.sampled_variables = tf.placeholder(tf.float32, [self.batch_size, 50])
        self.batch_image = tf.placeholder(tf.float32, [self.batch_size, 64, 128, 3])
        self.batch_sentence = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])  # [N,T]
        self.subj_sup = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.obj_sup = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.rel_sup = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.nets = self._build_net()
        self.loss = self._build_loss()

    def _build_net(self):
        # text attention
        with tf.variable_scope(NAME_SCOPE_ATTENTION):
            emb_sentence = layers.emb_layer('word_emb', tf.transpose(self.batch_sentence), self.dict_size, self.emb_size)
            with tf.variable_scope(SUB_SCOPE_TEXT):
                lstm_sentence = lstm.lstm_layer('lstm', emb_sentence, out_length=self.emb_lstm, num_layers=2)
                feat_sbj, att_sbj = attention.sequential_attention('att_1', lstm_sentence, self.emb_size / 2)
                feat_rel, att_rel = attention.sequential_attention('att_2', lstm_sentence, self.emb_size / 2)
                feat_obj, att_obj = attention.sequential_attention('att_3', lstm_sentence, self.emb_size / 2)
                image_net_in = tf.concat([self.sampled_variables, feat_sbj, feat_rel, feat_obj], axis=1)
        # generator
        with tf.variable_scope(an.NAME_SCOPE_GENERATIVE_NET):
            fake_image = (nets.net_generator(image_net_in) + 1) / 2.
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
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im_hist', fake_image)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im_hist', self.batch_image)
        return fake_image, fake_decision, real_decision, att_sbj, att_rel, att_obj

    def _build_loss(self):
        loss_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[2]), logits=self.nets[2]))
        loss_att_sbj = tf.reduce_mean(tf.nn.l2_loss(self.subj_sup - self.nets[3]))
        loss_att_rel = tf.reduce_mean(tf.nn.l2_loss(self.rel_sup - self.nets[4]))
        loss_att_obj = tf.reduce_mean(tf.nn.l2_loss(self.obj_sup - self.nets[3]))
        loss_att = 0.1 * (loss_att_sbj + loss_att_rel + loss_att_obj)
        loss_dis = loss_dis_fake + loss_dis_real + loss_att

        tf.summary.scalar(an.NAME_SCOPE_GENERATIVE_NET + '/loss', loss_gen)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss', loss_dis)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss_att', loss_att)

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

    def _init_embeddings(self):
        emb_matrix = np.load(self.emb_file)
        with tf.variable_scope(NAME_SCOPE_ATTENTION, reuse=True):
            emb_var = tf.get_variable('word_emb/emb_kernel', [self.dict_size, self.emb_size])
            op_init_emb = tf.assign(emb_var, emb_matrix)
        self.sess.run(op_init_emb)

    def train(self, max_iter, dataset, restore_file=None):
        from time import gmtime, strftime
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        ops = self._build_opt()
        initial_op = tf.global_variables_initializer()
        self.sess.run(initial_op)
        summary_path = os.path.join(self.log_path, 'log', time_string) + os.sep
        save_path = os.path.join(self.log_path, 'model') + os.sep

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if restore_file is not None:
            self._restore(restore_file)
            print('Model restored.')
        else:
            self._init_embeddings()

        writer = tf.summary.FileWriter(summary_path)
        summary_gen = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=an.NAME_SCOPE_GENERATIVE_NET))
        summary_dis = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope=an.NAME_SCOPE_DISCRIMINATIVE_NET))
        d_loss = 0
        for i in xrange(max_iter):
            this_batch = dataset.next_batch()
            this_feed_dict = {self.batch_image: this_batch['batch_image'],
                              self.sampled_variables: this_batch['batch_noise'],
                              self.batch_sentence: this_batch['batch_text'],
                              self.obj_sup: this_batch['batch_obj_sup'],
                              self.subj_sup: this_batch['batch_subj_sup'],
                              self.rel_sup: this_batch['batch_rel_sup']}
            if i % 1 == 0:
                d_loss, _, dis_sum = self.sess.run([self.loss[1], ops[1], summary_dis], feed_dict=this_feed_dict)
                writer.add_summary(dis_sum, global_step=tf.train.global_step(self.sess, self.g_step))

            g_loss, _, gen_sum = self.sess.run([self.loss[0], ops[0], summary_gen], feed_dict=this_feed_dict)
            writer.add_summary(gen_sum, global_step=tf.train.global_step(self.sess, self.g_step))

            step = tf.train.global_step(self.sess, self.g_step)
            print('Batch ' + str(i) + '(Global Step: ' + str(step) + '): ' + str(g_loss) + '; ' + str(d_loss))

            gc.collect()

            if i % 2000 == 0 and i > 0:
                self._save(save_path, step)
