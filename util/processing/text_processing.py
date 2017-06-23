import re

import numpy as np

INDICATOR_UNKNOWN = '<unk>'
INDICATOR_PADDING = '<pad>'


def load_dict(dict_file, pad_at_first=True):
    with open(dict_file, encoding='utf-8') as f:
        words = [w.strip() for w in f.readlines()]
    if pad_at_first and words[0] != '<pad>':
        raise Exception("The first word needs to be <pad> in the word list.")
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict


def word_index(sentence, dictionary, padding_length=10):
    reg_compiler = re.compile(r'(\W+)')
    words = reg_compiler.split(str(sentence).strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    if len(words) > 0 and (words[-1] == '.' or words[-1] == '?'):
        words = words[:-1]
    if len(words) < padding_length:
        words = [INDICATOR_PADDING] * (padding_length - len(words)) + words
    if len(words) > padding_length:
        words = words[:padding_length]
    vocab_indices = [(dictionary[w] if w in dictionary else dictionary[INDICATOR_UNKNOWN]) for w in words]
    return vocab_indices


def get_vector(start, end):
    sup = [0 for _ in xrange(length)]
    for i in xrange(start, end):
        sup[i] = 1
    return sup


if __name__ == '__main__':
    from six.moves import xrange

    data = np.load('../data/visg_man.npz')
    my_dict = load_dict('../data/vocabulary_72700.txt')
    length = 10
    p_desc = []
    subj_sup = []
    obj_sup = []
    rel_sup = []
    for i in xrange(len(data['id'])):
        print(str(i))
        this_sentence = data['desc'][i]
        processed_sentence = word_index(this_sentence, my_dict)
        p_desc.append(processed_sentence)

        if data['txt_pos'][i][0] + data['txt_pos'][i][1] + data['txt_pos'][i][2] > length:
            this_subj_sup = get_vector(0, data['txt_pos'][i][0])
            this_obj_sup = get_vector(data['txt_pos'][i][0] + data['txt_pos'][i][1], length)
            this_rel_sup = get_vector(data['txt_pos'][i][0], data['txt_pos'][i][0] + data['txt_pos'][i][1])
        else:
            pos1 = length - data['txt_pos'][i][2] - data['txt_pos'][i][1] - data['txt_pos'][i][0]
            pos2 = length - data['txt_pos'][i][2] - data['txt_pos'][i][1]
            pos3 = length - data['txt_pos'][i][2]
            this_subj_sup = get_vector(pos1, pos2)
            this_obj_sup = get_vector(pos3, length)
            this_rel_sup = get_vector(pos2, pos3)

        subj_sup.append(this_subj_sup)
        obj_sup.append(this_obj_sup)
        rel_sup.append(this_rel_sup)

    p_desc = np.asarray(p_desc, dtype=np.int32)
    subj_sup = np.asarray(subj_sup, dtype=np.int32)
    obj_sup = np.asarray(obj_sup, dtype=np.int32)
    rel_sup = np.asarray(rel_sup, dtype=np.int32)

    to_save = dict(id=data['id'],
             name=data['name'],
             desc=data['desc'],
             # box=data['box'],
             # subj=data['subj'],
             # obj=data['obj'],
             # txt_pos=data['txt_pos'],
             p_desc=p_desc,
             subj_sup=subj_sup,
             obj_sup=obj_sup,
             rel_sup=rel_sup)
    np.save('../data/visg_man_10.npy', to_save)
