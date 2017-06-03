import codecs
import pickle


def build_data(file_name, coding):

    data = dict()
    str_list = ''
    name = ''

    with codecs.open(file_name, mode='r', encoding=coding) as f:

        for line in f:

            line = line.replace("\r\n", "")
            line = line.replace("\u3000", "")
            line = line.strip()

            # skip empty lines
            if line == "" or line is None:
                continue

            # if the line contains author
            n = line.find("】")
            if n > 0:

                data[name] = data[name] + str_list if name in data else str_list
                name = line[n+1:]
                str_list = ''

                continue

            # assume all shi ju ends with . ,
            if line[-1] == '。' or line[-1] == '，':
                str_list = str_list + line

        # for the last author
        data[name] = str_list

    return data


def load_and_save_data(data_file='./data/shi.txt', save_file='./data/shi.dat'):
    """load data from text file, store in dic and serialize in binary file"""

    data = build_data(data_file, 'gb18030')

    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
        f.close()

    return data


def load_data_from_file(file_name='./data/shi.dat'):
    """load dic data from file"""

    try:
        with open(file_name, 'rb') as f:
            # load the object from the file into var b
            print('Loading shi from file: ' + file_name)
            data = pickle.load(f)
            f.close()

            return data
    except FileNotFoundError:
        print('Creating new shi data list...')
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


def build_shi_sentences():
    """return a list of sentences"""
    data = list(load_data_from_file().values())

    sentence_stream = create_sent_stream(data, True, True)
    return sentence_stream


