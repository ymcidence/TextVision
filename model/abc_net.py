from abc import ABCMeta, abstractmethod

import tensorflow as tf

MODE_FLAG_TRAIN = 'train'
MODE_FLAG_TEST = 'test'
NAME_SCOPE_GENERATIVE_NET = 'generative_net'
NAME_SCOPE_DISCRIMINATIVE_NET = 'discriminative_net'


class AbstractNet(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess')
        self.g_step = tf.Variable(0, trainable=False, name='global_step')

    @abstractmethod
    def _build_net(self):
        pass

    @abstractmethod
    def _build_loss(self):
        pass

    @abstractmethod
    def _build_opt(self):
        pass

    def train(self, max_iter, dataset, restore_file=None):
        pass

    def _restore(self, restore_file, var_list=None):
        if var_list is None:
            save_list = tf.trainable_variables()
        else:
            save_list = var_list
        saver = tf.train.Saver(var_list=save_list)
        saver.restore(self.sess, save_path=restore_file)

    def _save(self, save_path, step, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, save_path + 'YMModel', step)
        print('Saved!')


class NetFactory(object):
    @staticmethod
    def get_net(**kwargs):
        from model.text_gen_img import TextGenImage
        from model.text_gen_img_with_score import ScoredTextGenImage
        cases = {
            'abstract': AbstractNet,
            't2i': TextGenImage,
            'st2i': ScoredTextGenImage
        }
        model_name = kwargs.get('model')
        model = cases.get(model_name)
        return model(**kwargs)
