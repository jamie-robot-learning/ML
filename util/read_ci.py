import codecs
import pickle


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
    is_author_des = False

    with codecs.open(file_name, mode='r', encoding=coding) as f:

        for line in f:

            line = line.replace("\r\n", "")
            line = line.replace("\u3000", "")
            line = line.strip()

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

            # to detect authors not in name list
            if is_auth_des(line):
                data[name] = ''.join(str_list)
                name = line
                str_list = []
                continue
            # read titles
            if line[-1] != '。':
                continue

            str_list.append(line)

        # for the last author
        data[name] = ''.join(str_list)

    return data


def is_auth_des(line):
    if line.find("（") > 0 or line.find("）") > 0:
        return True
    return False


def load_and_save_data(author_file='./data/names.txt', data_file='./data/ci.txt', save_file='./data/ci.dat'):
    """load data from text file, store in dic and serialize in binary file"""

    author_names = read_names(author_file, 'gb18030')
    data = build_data(data_file, 'gb18030', author_names)

    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
        f.close()

    return data


def load_data_from_file(file_name='./data/ci.dat'):
    """load dic data from file"""

    try:
        with open(file_name, 'rb') as f:
            # load the object from the file into var b
            data = pickle.load(f)
            f.close()

            return data
    except FileNotFoundError:
        return load_and_save_data()


def create_sent_stream(text, sp_over=False, sp_coma=False):

    sentence_stream = list()
    for para in text:
        if sp_coma:
            para = para.replace("，", "。")

        for line in para.splitlines():

            if not sp_over:
                sentence_stream.append(list(line))
                continue

            for sen in line.split("。"):
                sentence_stream.append(list(sen))

    return sentence_stream


def build_ci_sentences():
    """return a list of sentences"""
    data = list(load_data_from_file().values())

    sentence_stream = create_sent_stream(data, True, True)
    return sentence_stream






#
# def combine_sentence_save(data, file_name='../one_sentence.dat'):
#     """combine list of string to one string, and serialize to binary file"""
#
#     str_b = ''
#     for s in data.values():
#         str_b += s
#
#     with open(file_name, 'wb') as f:
#         pickle.dump(str_b, f)
#         f.close()
#
#     return str_b
#
#
# def load_sentence(file_name='../data/one_sentence.dat'):
#     with open(file_name, 'rb') as f:
#         # load the object from the file into var b
#         s = pickle.load(f)
#         f.close()
#
#         return s
