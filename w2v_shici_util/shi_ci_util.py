from w2v_shici_util import read_shi
from w2v_shici_util import read_ci
from w2v_shici_util import word2vec
import pickle
import operator

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


def load_shi_and_ci_sentence_stream(is_dev_set):
    ci_sentence_stream = read_ci.build_ci_sentences()
    shi_sentence_stream = read_shi.build_shi_sentences()
    sentences = ci_sentence_stream + shi_sentence_stream

    if is_dev_set:
        sentences = sentences[(-int(len(sentences) * _dev_data_ratio)):]
    else:
        sentences = sentences[:int(len(sentences) * (1 - _dev_data_ratio))-1]
    for s in enumerate(sentences):
        yield s

    yield None


def load_shi_sentence_stream(is_dev_set):
    sentences = read_shi.build_shi_sentences()
    if is_dev_set:
        sentences = sentences[(-int(len(sentences) * _dev_data_ratio)):]
    else:
        sentences = sentences[:(int(len(sentences) * (1 - _dev_data_ratio))-1)]
    for s in enumerate(sentences):
        if s is None:
            print("")
            continue
        yield s

    yield None


def load_ci_sentence_stream(is_dev_set):
    sentences = read_ci.build_ci_sentences()

    if is_dev_set:
        sentences = sentences[-int(len(sentences) * _dev_data_ratio):]
    else:
        sentences = sentences[:int(len(sentences) * (1 - _dev_data_ratio))-1]
    for s in enumerate(sentences):
        yield s

    yield None


# def load_w2v_vocab():
#     model = word2vec.load_model()
#     return model.wv.vocab.keys()


def load_shi_vocab_mapping():
    """load (w2i,i2w) mappings from file, or create new"""

    try:
        with open('./data/w2i.dat', 'rb') as f:
            # load the object from the file into var b
            print('Loading w2i mapping from file: ./data/w2i.dat')
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
        data = (w2i, i2w)

        with open('./data/w2i.dat', 'wb') as f:
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
        with open('./data/vocab.dat', 'rb') as f:
            # load the object from the file into var b
            print('Loading vocabulary from file: ./data/vocab.dat')
            data = pickle.load(f)
            f.close()
            return data

    except FileNotFoundError:
        print('Building new vocabulary  data ...')

        vocab = dict()
        sentence_stream = load_shi_sentence_stream(True)
        _, s = next(sentence_stream)
        while s:
            for w in s:
                vocab[w] = 1 if w not in vocab else vocab[w] + 1
            _, s = next(sentence_stream)

        sentence_stream = load_shi_sentence_stream(False)
        _, s = next(sentence_stream)
        while s:
            for w in s:
                vocab[w] = 1 if w not in vocab else vocab[w] + 1
            _, s = next(sentence_stream)

        if max_vocab_size < len(vocab):
            x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
            sorted_x = sorted(x.items(), key=operator.itemgetter(1))
            data = dict(sorted_x[:max_vocab_size-1])
        else:
            data = vocab
        with open('./data/vocab.dat', 'wb') as f:
            pickle.dump(data, f)
            f.close()

        return data

