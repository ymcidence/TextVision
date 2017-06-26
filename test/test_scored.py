import tensorflow as tf

from model.abc_net import NetFactory
from util.data import dataset


def run():
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)
    settings = dict(
        model='st2i',
        batch_size=64,
        sess=sess,
        log_path='/home/ymcidence/WorkSpace/Loggings/st2i/',
        seq_length=10,
        dict_size=72704,
        emb_file='../util/data/embed_matrix.npy',
        set_size=58016,
        data_path='/home/ymcidence/WorkSpace/Data/VisualGenome/CropMan/',
        meta_file='../util/data/visg_man_10.npy',
        gradient_clip=0.5
    )
    model = NetFactory.get_net(**settings)
    data = dataset.SimpleDataset(**settings)
    model.train(30000, data)


if __name__ == '__main__':
    run()
