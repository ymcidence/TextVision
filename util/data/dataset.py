import numpy as np
from skimage.io import imread


def read_image(file_name):
    this_image = imread(file_name)
    if this_image.ndim == 2:
        this_image = np.tile(this_image[..., np.newaxis], (1, 1, 3))
    return this_image


class SimpleDataset(object):
    def __init__(self, **kwargs):
        self.set_size = kwargs.get('set_size')
        self.dict_size = kwargs.get('dict_size')
        self.seq_length = kwargs.get('seq_length')
        self.batch_size = kwargs.get('batch_size')
        self.data_path = kwargs.get('data_path')
        self.meta_file = kwargs.get('meta_file')
        self.meta = np.load(self.meta_file)[()]
        self.batch_num = self.set_size // self.batch_size
        self.batch_count = 0

    def next_batch(self):
        if self.batch_count == 0:
            self._shuffle()
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        batch_image = [read_image(self.data_path + v + '.jpg') for v in self.meta['id'][batch_start:batch_end]]
        batch_text = self.meta.get('p_desc')[batch_start:batch_end, ...]
        batch_desc = self.meta.get('desc')[batch_start:batch_end, ...]
        batch_subj_sup = self.meta.get('subj_sup')[batch_start:batch_end, ...]
        batch_obj_sup = self.meta.get('obj_sup')[batch_start:batch_end, ...]
        batch_rel_sup = self.meta.get('rel_sup')[batch_start:batch_end, ...]
        this_batch = dict(batch_image=np.asarray(batch_image, dtype=np.float32) / 255.,
                          batch_desc=batch_desc,
                          batch_text=batch_text,
                          batch_obj_sup=batch_obj_sup,
                          batch_rel_sup=batch_rel_sup,
                          batch_subj_sup=batch_subj_sup,
                          batch_noise=np.random.uniform(-1, 1, size=(self.batch_size, 450)))
        self.batch_count = (self.batch_count + 1) % self.batch_num
        return this_batch

    def _shuffle(self):
        inds = np.random.choice(self.set_size, self.set_size)
        new_meta = dict(
            id=self.meta['id'][inds],
            name=self.meta['name'][inds],
            desc=self.meta['desc'][inds],
            p_desc=self.meta['p_desc'][inds, ...],
            subj_sup=self.meta['subj_sup'][inds, ...],
            obj_sup=self.meta['obj_sup'][inds, ...],
            rel_sup=self.meta['rel_sup'][inds, ...]
        )
        self.meta = new_meta

    def _sample(self):
        inds = np.random.choice(58016, self.batch_size)
        new_meta = dict(
            id=self.meta['id'][inds],
            name=self.meta['name'][inds],
            desc=self.meta['desc'][inds],
            p_desc=self.meta['p_desc'][inds, ...],
            subj_sup=self.meta['subj_sup'][inds, ...],
            obj_sup=self.meta['obj_sup'][inds, ...],
            rel_sup=self.meta['rel_sup'][inds, ...]
        )
        self.meta = new_meta


if __name__ == '__main__':
    a = np.load('../data/visg_man_10.npy')[()]
    print(len(a['id']))
