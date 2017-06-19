import tensorflow as tf
from abc import ABCMeta, abstractmethod

MODE_FLAG_TRAIN = 'train'
MODE_FLAG_TEST = 'test'
NAME_SCOPE_GENERATIVE_NET = 'generative_net'
NAME_SCOPE_DISCRIMINATIVE_NET = 'discriminative_net'


class AbstractNet(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess')
        self.g_step = tf.Variable(0, trainable=False, name='global_step')

    @abstractmethod
    def _build_net(self): pass

    @abstractmethod
    def _build_loss(self): pass

    @abstractmethod
    def _build_opt(self): pass

    def train(self, max_iter, dataset): pass

    def _restore(self, restore_file):
        save_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list=save_list)
        saver.restore(self.sess, save_path=restore_file)

    def _save(self, save_path, step):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path + 'YMModel', step)


class NetFactory(object):
    @staticmethod
    def get_net(**kwargs):
        from model.text_gen_img import TextGenImage
        cases = {
            'abstract': AbstractNet,
            't2i': TextGenImage
        }
        model_name = kwargs.get('model')
        model = cases.get(model_name)
        return model(**kwargs)