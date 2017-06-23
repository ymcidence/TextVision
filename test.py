import tensorflow as tf

from model.abc_net import NetFactory
from util.data import dataset


def run():
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)
    settings = dict(
        model='t2i',
        batch_size=32,
        sess=sess,
        log_path='/home/ymcidence/WorkSpace/Loggings/t2i/',
        seq_length=10,
        dict_size=72704,
        emb_file='./util/data/embed_matrix.npy',
        set_size=58016,
        data_path='/home/ymcidence/WorkSpace/Data/VisualGenome/CropMan/',
        meta_file='./util/data/visg_man_10.npy'
    )
    model = NetFactory.get_net(**settings)
    data = dataset.SimpleDataset(**settings)
    model.train(6006, data)


if __name__ == '__main__':
    run()
