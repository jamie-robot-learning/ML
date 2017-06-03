from util import read_shi
from util import read_ci


def load_data():
    ci_sentence_stream = read_ci.build_ci_sentences()
    shi_sentence_stream = read_shi.build_shi_sentences()
    sentence_stream = ci_sentence_stream + shi_sentence_stream

    return sentence_stream
