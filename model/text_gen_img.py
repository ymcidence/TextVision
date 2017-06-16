import tensorflow as tf
import model.abc_net as an

NAME_SCOPE_ATTENTION = 'attention_net'
SUB_SCOPE_TEXT = 'text'
SUB_SCOPE_JOINT = 'joint'


class TextGenImage(an.AbstractNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = kwargs.get('batch_size')
        self.log_path = kwargs.get('log_path')
        self.sampled_variables = tf.placeholder(tf.float32, [self.batch_size, 100])
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])
        self.nets = self._build_net()
        self.loss = self._build_loss()

    def _build_net(self):
        pass

    def _build_loss(self):
        pass

    def _build_opt(self):
        pass
