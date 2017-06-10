import codecs
import logging

import gensim
import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from w2v_shici_util import shi_ci_util

#zhfont1 = matplotlib.font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def build_phrases():

    sentence_stream = shi_ci_util.load_data()

    phrases = gensim.models.Phrases(sentence_stream, min_count=5, threshold=100)
    bigram = gensim.models.phrases.Phraser(phrases)

    # Following code for printing out the tokenized vocab
    vocab = dict()
    ssss = 0
    for phrase, score in phrases.export_phrases(sentence_stream):
        vocab[phrase.decode('utf-8')] = score
        ssss += score

    sorted_dic = [(k, vocab[k]) for k in sorted(vocab, key=vocab.get, reverse=True)]

    print(sorted_dic)
    print(len(sorted_dic))
    print('sum:' + str(ssss))

    bigram.save('./data/phrases.dat')

    return bigram, sentence_stream


def load_phraser_and_sentence_stream():
    try:
        phraser = gensim.models.Phrases.load('./data/phrases.dat')
        print('Load phraser from file')
        return phraser, shi_ci_util.load_data()
    except FileNotFoundError:
        return build_phrases()


def load_model():
    try:
        model = gensim.models.Word2Vec.load('./data/vec.mdl')
        return model

    except FileNotFoundError:
        print('Building new model...\n')
        bigram, sentence_stream = load_phraser_and_sentence_stream()
        corpus = list(bigram[sentence_stream])

        # better result from testing data
        model = gensim.models.Word2Vec(corpus, min_count=5, size=100, workers=4, window=5,
                                       sg=0, sample=1e-3, alpha=0.025,)
        model.save('./data/vec.mdl')

    return model

#
# def plot_vocab_with_tsne(model):
#
#     wv = model.wv
#     random_vocab = (model.wv.vocab.keys())
#
#     X = None
#     vocabulary = None
#     try:
#         vocabulary, X = zip(*[(it, wv[it]) for it in random_vocab if wv.__contains__(it)])
#     except KeyError:
#         pass
#
#     tsne = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     Y = tsne.fit_transform(X)
#
#     plt.scatter(Y[:, 0], Y[:, 1])
#     for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontproperties=zhfont1)
#
#     plt.show()
#
#     return


def test_model(model, arg_name, arg_value, words=('玉', '云', '马', '日', '天', '绿', '竹')):
    s = '\n\nTesting: ' + str(arg_name) + '=' + str(arg_value) + '.............\n'

    for w in words:
        _, ww, ws = zip(*[(_, ww, ws) for _, (ww, ws) in enumerate(model.wv.most_similar(w, topn=10))])
        s += '和（' + w + '）接近的10个词为： ' + str(ww) + '\n'
        print(s)
    try:
        f = codecs.open('test.txt', 'a', encoding='utf-8')
        f.write(s)
        f.close()
    except IOError as e:
        f.close()
        print(e)

    print(s)

    return

#model = load_model()

def optimaze_model():

    #  try with and without phrase
    bigram, sentence_stream = load_phraser_and_sentence_stream()
    corpus = list(sentence_stream)
    # corpus = list(bigram[sentence_stream])

    p = dict()
    p['min_count'] = (2, 5, 10)
    p['size'] = (80, 100)

    # learning rate by default = 0.025
    p['alpha'] = (0.025, 0.5)

    p['window'] = (3, 7, 10)
    p['iter'] = (5, 15)

    # sg = 1 for skip_gram
    p['sg'] = (0, 1)

    # number of nagtive words
    p['negative'] = (5, 10)

    # threshold for configuring which higher-frequency words are randomly downsampled;
    p['sample'] = (1e-3, 1e-1)

    for para_name in p.keys():
        for para_value in p[para_name]:
            args = {'sentences': corpus, para_name: para_value}

            print('Building Model: ' + str(para_name) + '=' + str(para_value) + '.............')

            try:
                model = load_test_model(para_value,para_value)
                if model is None:
                    model = gensim.models.Word2Vec(**args)
                save_test_model(model,para_name,para_value)

                test_model(model, para_name, para_value)
                del model
            except Exception as e:
                print(e)

    return


def save_test_model(model,para_name, para_value):
    try:
        f = './temp/model_' + str(para_name) + '_' + str(para_value) + '.dat'
        model.save(f)
        print('Model saved to file:' + f)
        return
    except IOError:
        return


def load_test_model(para_name, para_value):
    try:
        f = './temp/model_' + str(para_name) + '_' + str(para_value) + '.dat'
        model = gensim.models.Word2Vec.load(f)
        print('Model load from file:' + f)
        return model
    except FileNotFoundError:
        return None

# print(model.wv.vocab.keys())


#plot_vocab_with_tsne(model)
#
# model = load_model()
# print(model.wv.most_similar(positive=['日', '天'], negative=['月']))
# print(model.wv.most_similar(positive=['东', '日'], negative=['西']))
# print(model.wv.most_similar(positive=['绿', '竹'], negative=['杨']))

# optimaze_model()

# build_phrases()
# shi = read_shi.load_data_from_file()
#
# print(shi['李显'])
