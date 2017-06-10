import sys

from seq2seq import shi_generator
from shi_gen_util import read_shi
import pickle
import operator
from random import randint

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_dev_data_ratio = 0.01
_buckets = shi_generator.buckets






# def load_w2v_vocab():
#     model = word2vec.load_model()
#     return model.wv.vocab.keys()


def load_shi_vocab_mapping():
    """load (w2i,i2w) mappings from file, or create new"""

    try:
        with open('./shi_gen_data/w2i.dat', 'rb') as f:
            # load the object from the file into var b
            print('Loading w2i mapping from file: ./shi_gen_data/w2i.dat')
            data = pickle.load(f)
            f.close()
            return data

    except FileNotFoundError:
        print('Building new w2i mapping data ...')
        w2i = dict()
        i2w = dict()

        # starting from 4 to avoid pre-allocated numbers
        i = 4
        vocab = load_shi_vocab()
        for w in vocab:
            w2i[w] = i
            i2w[i] = w
            i += 1
        w2i['3'] = str(UNK_ID)
        i2w['3'] = str(3)
        data = (w2i, i2w)

        with open('./shi_gen_data/w2i.dat', 'wb') as f:
            pickle.dump(data, f)
            f.close()
        return data


def sentence_to_int_list(sentence, w2i):
    """convert a sentence into a list of integers"""
    ids = []
    for w in sentence:
        ids.append(UNK_ID if w not in w2i else w2i[w])
    return ids


def load_shi_vocab(max_vocab_size=10000):

    try:
        with open('./shi_gen_data/vocab.dat', 'rb') as f:
            # load the object from the file into var b
            print('Loading vocabulary from file: ./shi_gen_data/vocab.dat')
            data = pickle.load(f)
            f.close()
            return data

    except FileNotFoundError:
        print('Building new vocabulary  data ...')

        vocab = dict()
        all_shi = read_shi.load_shi()

        for author in all_shi:
            for _, one_shi in enumerate(all_shi[author]):
                for w in one_shi:
                    vocab[w] = 1 if w not in vocab else vocab[w] + 1

        if max_vocab_size < len(vocab):
            sorted_x = sorted(vocab.items(), key=operator.itemgetter(1))
            data = dict(sorted_x[:max_vocab_size-1])
        else:
            data = vocab
        with open('./shi_gen_data/vocab.dat', 'wb') as f:
            pickle.dump(data, f)
            f.close()

        return data


def load_data(max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      is_dev_set: is generating dev data.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """

    shi_list = read_shi.load_shi()

    # target are just the next sentence of the source

    (w2i, i2w) = load_shi_vocab_mapping()

    data_set = [[] for _ in _buckets]
    data_set_dev = [[] for _ in _buckets]

    counter = 0
    for author in shi_list:
        for _, shi in enumerate(shi_list[author]):
            shi = shi.replace("，", "。")
            sentences = shi.split('。')

            for i in range(0, len(sentences)-2):
                if sentences[i] == "":
                    continue
                source = sentences[i]
                target = sentences[i+1]
                counter += 1
                if counter % 10000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [(UNK_ID if w not in w2i else w2i[w]) for w in source]
                target_ids = [(UNK_ID if w not in w2i else w2i[w]) for w in target]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        if randint(1, 100) <= (100*_dev_data_ratio):
                            data_set_dev[bucket_id].append([source_ids, target_ids])
                        else:
                            data_set[bucket_id].append([source_ids, target_ids])
                        break

    return [data_set, data_set_dev]


def read_data(is_dev_set=False, max_size=None):
    try:
        with open('./shi_gen_data/data_set.dat', 'rb') as f:
            # load the object from the file into var b
            print('Loading data set from file: ./shi_gen_data/data_set.dat')
            data = pickle.load(f)
            f.close()
            if is_dev_set:
                return data[1]
            return data[0]

    except FileNotFoundError:
        print('Building new data set  data ...')

        data = load_data(max_size)

        with open('./shi_gen_data/data_set.dat', 'wb') as f:
            pickle.dump(data, f)
            f.close()

            if is_dev_set:
                return data[1]
            return data[0]

