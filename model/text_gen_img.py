import gc
import os
from time import gmtime, strftime

import numpy as np
import tensorflow as tf
from six.moves import xrange

import model.abc_net as an
from util.layers import conventional_layers as layers
from util.layers import lstm_layer as lstm
from util.layers import nets
from util.layers.attention import sequential_attention_real as text_attention

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
            self.emb_lstm = 500
        self.fused_discriminator = True
        self.sampled_variables = tf.random_uniform([self.batch_size, 100], minval=-1, maxval=1)
        self.batch_image = tf.placeholder(tf.float32, [self.batch_size, 128, 64, 3])
        self.batch_sentence = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])  # [N,T]
        self.subj_sup = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.obj_sup = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.rel_sup = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.nets = self._build_net()
        self.loss = self._build_loss()

    def _build_net(self):
        def bn(_in_tensor):
            return tf.layers.batch_normalization(_in_tensor, momentum=0.9, epsilon=1e-5)

        # text attention
        with tf.variable_scope(NAME_SCOPE_ATTENTION):
            words = tf.transpose(self.batch_sentence)
            emb_sentence = layers.emb_layer('word_emb', words, self.dict_size,
                                            self.emb_size)
            with tf.variable_scope(SUB_SCOPE_TEXT):
                lstm_sentence = lstm.lstm_layer('lstm', emb_sentence, out_length=self.emb_lstm, num_layers=2)
                feat_sbj, att_sbj = text_attention('att_1', lstm_sentence, words)
                feat_rel, att_rel = text_attention('att_2', lstm_sentence, words)
                feat_obj, att_obj = text_attention('att_3', lstm_sentence, words)
                feat_sentence = tf.concat([feat_sbj, feat_rel, feat_obj], axis=1)
                feat_sentence = layers.leaky_relu(layers.fc_layer('fc_1', feat_sentence, 128))
                # image_net_in = tf.concat([self.sampled_variables, feat_sentence], axis=1)
                image_net_in = feat_sentence
        # generator
        with tf.variable_scope(an.NAME_SCOPE_GENERATIVE_NET):
            fake_image = nets.net_generator(image_net_in)
        # discriminator

        if self.fused_discriminator:
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET):
                fake_decision = nets.net_fused_discriminator(fake_image, feat_sentence)
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
                real_decision = nets.net_fused_discriminator(self.batch_image, feat_sentence)
        else:
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET):
                fake_decision = nets.net_discriminator(fake_image)
            with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
                real_decision = nets.net_discriminator(self.batch_image)

        rels = tf.matmul(feat_sentence, feat_sentence, transpose_b=True)
        rels = tf.expand_dims(rels, 0)
        rels = tf.expand_dims(rels, -1)
        tf.summary.image(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im', fake_image)
        tf.summary.image(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im', self.batch_image)
        tf.summary.image(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/rels', rels)
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_dec_hist', fake_decision)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/dis_dec_hist', real_decision)
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im_hist', fake_image)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im_hist', self.batch_image)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/rep_sbj', feat_sbj)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/rep_obj', feat_obj)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/att_sbj', tf.argmax(att_sbj, axis=1))
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/att_obj', tf.argmax(att_obj, axis=1))

        tf.summary.image(NAME_SCOPE_ATTENTION + '/rels', rels)
        tf.summary.histogram(NAME_SCOPE_ATTENTION + '/att_sbj', tf.argmax(att_sbj, axis=1))
        tf.summary.histogram(NAME_SCOPE_ATTENTION + '/att_obj', tf.argmax(att_obj, axis=1))
        return fake_image, fake_decision, real_decision, att_sbj, att_rel, att_obj

    def _build_loss(self):
        loss_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[2]), logits=self.nets[2]))

        # loss_att_sbj = tf.losses.sigmoid_cross_entropy(self.subj_sup, self.nets[3])
        # loss_att_rel = tf.losses.sigmoid_cross_entropy(self.rel_sup, self.nets[4])
        # loss_att_obj = tf.losses.sigmoid_cross_entropy(self.obj_sup, self.nets[5])

        loss_att_sbj = tf.nn.l2_loss(self.subj_sup - self.nets[3])
        loss_att_rel = tf.nn.l2_loss(self.rel_sup - self.nets[4])
        loss_att_obj = tf.nn.l2_loss(self.obj_sup - self.nets[5])

        loss_att = 0.01 * (loss_att_sbj + loss_att_rel + loss_att_obj)
        loss_dis = loss_dis_fake + loss_dis_real + loss_att

        # loss_hehe = tf.nn.l2_loss(self.nets[-1])
        # loss_gen = loss_gen + loss_hehe
        # tf.summary.scalar(an.NAME_SCOPE_GENERATIVE_NET + '/loss', loss_hehe)

        tf.summary.scalar(an.NAME_SCOPE_GENERATIVE_NET + '/hehe', loss_gen)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss', loss_dis)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss_att', loss_att)
        tf.summary.scalar(NAME_SCOPE_ATTENTION + '/loss_att', loss_att)

        return loss_gen, loss_dis

    def _build_opt(self):
        trainer1 = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        trainer2 = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=an.NAME_SCOPE_GENERATIVE_NET)
        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=an.NAME_SCOPE_DISCRIMINATIVE_NET)
        train_list_att_txt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=NAME_SCOPE_ATTENTION)

        train_list_gen = train_list_gen  # + train_list_att_txt
        train_list_dis = train_list_dis  # + train_list_att_txt
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
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        ops = self._build_opt()
        initial_op = tf.global_variables_initializer()
        self.sess.run(initial_op)
        summary_path = os.path.join(self.log_path, 'log', time_string) + os.sep
        save_path = os.path.join(self.log_path, 'model', time_string) + os.sep

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
        g_loss = 0
        for i in xrange(max_iter):
            this_batch = dataset.next_batch()
            this_feed_dict = {self.batch_image: this_batch['batch_image'],
                              self.batch_sentence: this_batch['batch_text'],
                              self.obj_sup: this_batch['batch_obj_sup'],
                              self.subj_sup: this_batch['batch_subj_sup'],
                              self.rel_sup: this_batch['batch_rel_sup']}

            d_loss, _, dis_sum = self.sess.run([self.loss[1], ops[1], summary_dis], feed_dict=this_feed_dict)
            writer.add_summary(dis_sum, global_step=tf.train.global_step(self.sess, self.g_step))

            if i % 2 == 0:
                g_loss, _, gen_sum = self.sess.run([self.loss[0], ops[0], summary_gen], feed_dict=this_feed_dict)
                writer.add_summary(gen_sum, global_step=tf.train.global_step(self.sess, self.g_step))

            step = tf.train.global_step(self.sess, self.g_step)
            print('Batch ' + str(i) + '(Global Step: ' + str(step) + '): ' + str(g_loss) + '; ' + str(d_loss))
            gc.collect()

            if i % 2000 == 0 and i > 0:
                self._save(save_path, step)

    def pre_train(self, max_iter, dataset):
        loss_att_sbj = tf.nn.l2_loss(self.subj_sup - self.nets[3])
        loss_att_rel = tf.nn.l2_loss(self.rel_sup - self.nets[4])
        loss_att_obj = tf.nn.l2_loss(self.obj_sup - self.nets[5])

        loss = 0.01 * (loss_att_sbj + loss_att_rel + loss_att_obj)

        trainer = tf.train.AdamOptimizer(0.00001)
        train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=NAME_SCOPE_ATTENTION)
        opt = trainer.minimize(loss, self.g_step, var_list=train_list)

        initial_op = tf.global_variables_initializer()
        self.sess.run(initial_op)
        self._init_embeddings()
        summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=NAME_SCOPE_ATTENTION))

        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join(self.log_path, 'log', time_string) + os.sep
        save_path = os.path.join(self.log_path, 'model', 'text_net') + os.sep
        writer = tf.summary.FileWriter(summary_path)

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in xrange(max_iter):
            this_batch = dataset.next_batch()
            this_feed_dict = {self.batch_sentence: this_batch['batch_text'],
                              self.obj_sup: this_batch['batch_obj_sup'],
                              self.subj_sup: this_batch['batch_subj_sup'],
                              self.rel_sup: this_batch['batch_rel_sup']}
            d_loss, _, dis_sum = self.sess.run([loss, opt, summary], feed_dict=this_feed_dict)
            writer.add_summary(dis_sum, global_step=tf.train.global_step(self.sess, self.g_step))
            print('Batch ' + str(i) + '(Global Step: ' + str(i) + '): ' + str(d_loss))
            gc.collect()

        self._save(save_path=save_path, step=max_iter, var_list=train_list)
