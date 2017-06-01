import codecs
import gensim, logging
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_names(file_name, coding):

    names = set()

    with codecs.open(file_name, 'r', encoding=coding) as f:

        for line in f:
            line = line.replace("\r\n", "")
            if line == "" or line is None:
                continue

            names.add(line)

    return names


def build_data(file_name, coding, names):

    data = dict()
    str_list = []
    name = ''
    i = 0
    is_author_des = False

    with codecs.open(file_name, mode='r', encoding=coding) as f:

        for line in f:

            line = line.replace("\r\n", "")
            line = line.replace("\u3000", "")

            # skip empty lines
            if line == "" or line is None:
                continue

            # read author des
            if is_author_des:
                is_author_des = False
                continue

            # read author name
            if line in names:
                data[name] = ''.join(str_list)
                name = line
                str_list = []
                is_author_des = True
                continue

            # read titles
            if line[-1:] != '。':
                continue

            str_list.append("\u3000" + line + '\n')

        # for the last author
        data[name] = ''.join(str_list)

    return data


def load_and_save_data(author_file='n.txt', data_file='ci.txt', save_file='ci.dat'):
    """load data from text file, store in dic and serialize in binary file"""

    author_names = read_names(author_file, 'gb18030')
    data = build_data(data_file, 'gb18030', author_names)

    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
        f.close()

    return data


def load_data_from_file(file_name='ci.dat'):
    """load dic data from file"""

    with open(file_name, 'rb') as f:
        # load the object from the file into var b
        data = pickle.load(f)
        f.close()

        return data


def combine_sentence_save(data, file_name='one_sentence.dat'):
    """combine list of string to one string, and serialize to binary file"""

    str_b = ''
    for s in data.values():
        str_b += s

    with open(file_name, 'wb') as f:
        pickle.dump(str_b, f)
        f.close()

    return str_b


def load_sentence(file_name='one_sentence.dat'):
    s = ''

    with open(file_name, 'rb') as f:
        # load the object from the file into var b
        s = pickle.load(f)
        f.close()

        return s


def create_sent_stream(sentences, sp_over=False):

    sentence_stream = list()
    for para in sentences:
        for line in para.splitlines():

            if not sp_over:
                sentence_stream.append(list(line))
                continue

            for sen in line.split("。"):
                sentence_stream.append(list(sen))

    return sentence_stream




""""
with open('ci.dat', 'rb') as f:
    # load the object from the file into var b
    d = pickle.load(f)

print(d)
"""

#phrasal_data = gensim.models.Phrases(data)

#model = gensim.models.Word2Vec(phrasal_data[data], min_count=5, size=100, workers=4)
#model.save(fname)

#build_data('ci.txt', 'gb18030')