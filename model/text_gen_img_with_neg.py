import gc
import os
from time import gmtime, strftime

import tensorflow as tf
from six.moves import xrange

import model.abc_net as an
from model.text_gen_img import TextGenImage
from util.layers import conventional_layers as layers
from util.layers import lstm_layer as lstm
from util.layers import nets
from util.layers.attention import sequential_attention_real as text_attention

NAME_SCOPE_ATTENTION = 'attention_net'
SUB_SCOPE_TEXT = 'text'
SUB_SCOPE_JOINT = 'joint'


def bn(_in_tensor):
    return tf.layers.batch_normalization(_in_tensor, momentum=0.9, epsilon=1e-5)


class ConTextGenImage(TextGenImage):
    def __init__(self, **kwargs):
        self.batch_neg_sentence = tf.placeholder(tf.int32, [kwargs['batch_size'], kwargs['seq_length']])
        super().__init__(**kwargs)
        self.gradient_clip = kwargs.get('gradient_clip')

    def _build_text_net(self, tensor_in):
        words = tf.transpose(tensor_in)
        emb_sentence = layers.emb_layer('word_emb', words, self.dict_size,
                                        self.emb_size)
        with tf.variable_scope(SUB_SCOPE_TEXT):
            lstm_sentence = lstm.lstm_layer('lstm', emb_sentence, out_length=self.emb_lstm, num_layers=2)
            feat_sbj, att_sbj = text_attention('att_1', lstm_sentence, words)
            feat_rel, att_rel = text_attention('att_2', lstm_sentence, words)
            feat_obj, att_obj = text_attention('att_3', lstm_sentence, words)
            feat_sentence = tf.concat([bn(feat_sbj), bn(feat_rel), bn(feat_obj)], axis=1)
            feat_sentence = layers.leaky_relu(layers.fc_layer('fc_1', feat_sentence, 128))

        return feat_sentence

    def _build_net(self):
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
                feat_sentence = tf.concat([bn(feat_sbj), bn(feat_rel), bn(feat_obj)], axis=1)
                feat_sentence = layers.leaky_relu(layers.fc_layer('fc_1', feat_sentence, 128))
                image_net_in = tf.concat([self.sampled_variables, feat_sentence], axis=1)

        with tf.variable_scope(NAME_SCOPE_ATTENTION, reuse=True):
            feat_neg_sentence = self._build_text_net(self.batch_neg_sentence)

        # generator
        with tf.variable_scope(an.NAME_SCOPE_GENERATIVE_NET):
            fake_image = nets.net_generator(image_net_in)
        # discriminator
        with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET):
            fake_decision, fake_rep = nets.net_scored_discriminator(fake_image, feat_sentence.shape[1].value)
            fake_score = tf.multiply(fake_rep, feat_sentence)
        with tf.variable_scope(an.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
            real_decision, real_rep = nets.net_scored_discriminator(self.batch_image, feat_sentence.shape[1].value)
            real_score = tf.multiply(real_rep, feat_sentence)
            neg_score = tf.multiply(real_rep, feat_neg_sentence)

        rels = tf.matmul(feat_sentence, feat_sentence, transpose_b=True)
        rels = tf.expand_dims(rels, 0)
        rels = tf.expand_dims(rels, -1)
        tf.summary.image(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im', fake_image, max_outputs=10)
        tf.summary.image(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im', self.batch_image, max_outputs=10)
        tf.summary.image(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/rels', rels)
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_dec_hist', fake_decision)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/dis_dec_hist', real_decision)
        tf.summary.histogram(an.NAME_SCOPE_GENERATIVE_NET + '/gen_im_hist', fake_image)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im_hist', self.batch_image)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/rep_sbj', feat_sbj)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/rep_obj', feat_obj)
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/att_sbj', tf.argmax(att_sbj, axis=1))
        tf.summary.histogram(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/att_obj', tf.argmax(att_obj, axis=1))

        to_return = [fake_image, fake_decision, real_decision, att_sbj, att_rel, att_obj, fake_score, real_score,
                     feat_sentence, neg_score]
        return to_return

    def _build_loss(self):
        loss_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[2]), logits=self.nets[2]))

        loss_gen_score = -1 * tf.reduce_mean(self.nets[6])

        loss_dis_score_fake = tf.reduce_mean(self.nets[6])
        loss_dis_score_real = -1 * tf.reduce_mean(self.nets[7])
        loss_dis_score_neg = tf.reduce_mean(self.nets[9])
        loss_w = (loss_dis_score_real + loss_dis_score_neg) / 2

        loss_att_sbj = tf.nn.l2_loss(self.subj_sup - self.nets[3])
        loss_att_rel = tf.nn.l2_loss(self.rel_sup - self.nets[4])
        loss_att_obj = tf.nn.l2_loss(self.obj_sup - self.nets[5])
        loss_rel = tf.nn.l2_loss(
            tf.matmul(self.nets[8], self.nets[8], transpose_b=True) - tf.eye(self.batch_size, self.batch_size))
        loss_att = 0.01 * (loss_att_sbj + loss_att_rel + loss_att_obj + loss_rel)

        loss_gen = loss_gen + 0.5 * loss_gen_score
        loss_dis = loss_dis_fake + loss_dis_real + 0.5 * (loss_dis_score_fake + loss_w)  # + loss_att

        tf.summary.scalar(an.NAME_SCOPE_GENERATIVE_NET + '/hehe', loss_gen)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss', loss_dis)
        tf.summary.scalar(an.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss_att', loss_att)

        return loss_gen, loss_dis

    def _build_opt(self):
        trainer1 = tf.train.RMSPropOptimizer(0.00005)
        trainer2 = tf.train.RMSPropOptimizer(0.00005)
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=an.NAME_SCOPE_GENERATIVE_NET)
        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=an.NAME_SCOPE_DISCRIMINATIVE_NET)
        # train_list_att_txt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=NAME_SCOPE_ATTENTION)

        train_list_gen = train_list_gen  # + train_list_att_txt
        train_list_dis = train_list_dis  # + train_list_att_txt
        op_gen = trainer1.minimize(self.loss[0], var_list=train_list_gen, global_step=self.g_step)
        op_dis_provisional = trainer2.minimize(self.loss[1], var_list=train_list_dis, global_step=self.g_step)

        with tf.control_dependencies([op_dis_provisional]):
            op_dis = [tf.assign(var, tf.clip_by_value(var, -1 * self.gradient_clip, self.gradient_clip)) for var in
                      train_list_dis]

        return op_gen, op_dis

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
                              self.rel_sup: this_batch['batch_rel_sup'],
                              self.batch_neg_sentence: this_batch['batch_ntext']}

            d_loss, _, dis_sum = self.sess.run([self.loss[1], ops[1], summary_dis], feed_dict=this_feed_dict)
            writer.add_summary(dis_sum, global_step=tf.train.global_step(self.sess, self.g_step))

            if i % 1 == 0:
                g_loss, _, gen_sum = self.sess.run([self.loss[0], ops[0], summary_gen], feed_dict=this_feed_dict)
                writer.add_summary(gen_sum, global_step=tf.train.global_step(self.sess, self.g_step))

            step = tf.train.global_step(self.sess, self.g_step)
            print('Batch ' + str(i) + '(Global Step: ' + str(step) + '): ' + str(g_loss) + '; ' + str(d_loss))
            gc.collect()

            if i % 2000 == 0 and i > 0:
                self._save(save_path, step)
