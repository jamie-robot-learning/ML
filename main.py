from w2v_shici_util import shi_ci_util
from shi_gen_util import shi_util

from seq2seq import shi_generator
import numpy as np

# vocab = data_util.load_w2i_mapping()
# print(vocab[0]['昆'])


# shi_sentence_stream = read_shi.build_shi_sentences()
# ci_sentence_stream = read_ci.build_ci_sentences()
# print(len(ci_sentence_stream))
# print(ci_sentence_stream[:20])

# s = data_util.convert_into_int_list()
# print(s[:200])
#
# shi_sentence_stream = read_shi.build_shi_sentences()
# w2i, i2w = data_util.load_w2i_mapping()

# source_gen = data_util.load_shi_sentence_stream(False)
# target_gen = data_util.load_shi_sentence_stream(False)
# next(target_gen)
#
# (w2i, i2w) = data_util.load_vocab_mapping()
#
# (_, source), (_, target) = next(source_gen), next(target_gen)
# print(source)
# print(target)

# translate.read_data(shi_sentence_stream, w2i)

shi_generator.train()

# shi_generator.decode()
# d = shi_generator.read_data()
# shi = read_shi.build_shi_sentences()

# shi = shi_util.read_data()
# print(shi[0])
# shi_generator.decode()

#
# b = np.arange(24).reshape(4,6)
# print(b)
# print(np.argmax(b,axis=1))