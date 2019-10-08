import os
import numpy as np

def creat_vocabulary(data_paths, save_path):
    if not os.path.exists(save_path):
        print('create vocabulary file...')
        vocab = {}
        vocab['UNK'] = 0
        vocab['PAD'] = 1
        idx = 2
        for data_path in data_paths:
            with open(data_path, 'r', encoding='utf-8') as f_r:
                for line in f_r:
                    text = line.split('\t')[1]
                    for word in text.strip().split(' '):
                        word = word.lower()
                        if word not in vocab.keys() and len(word) != 0 :
                            vocab[word] = idx
                            idx += 1

        with open(save_path, 'w', encoding='utf-8') as f_w:
            for word, i in vocab.items():
                f_w.write(word + '\t' + str(i) + '\n')
    else:
        print('vocabulary file is already exists...')

def load_vocabulary(vocab_path):
    if os.path.exists(vocab_path):
        print('load vocabulary file from {}'.format(vocab_path))
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f_r:
            for line in f_r:
                word, idx = line.strip().split('\t')[0], line.strip().split('\t')[1]
                vocab[word] = idx
    else:
        raise ValueError('Vocabulary file {} does not found'.format(vocab_path))

    return vocab

def creat_embedding_mat(vocab, glove_path, embed_mat_path):
    if not os.path.exists(embed_mat_path):
        print('create word embedding mat...')
        embed_mat = np.zeros((len(vocab), 300), dtype=float)
        count = 0
        with open(glove_path, 'r', encoding='utf-8') as f_r:
            for line in f_r:
                line_ls = line.lstrip().rstrip().split(' ')
                word = line_ls[0]
                value = list(map(float, line_ls[1:]))
                if word in vocab.keys():
                    count += 1
                    embed_mat[int(vocab[word]), :] = value

        np.savez_compressed(embed_mat_path, glove=embed_mat)
        print('{}/{} words are found in {}.'.format(count, len(vocab), glove_path))
    else:
        print('word embedding mat is already exists...')

def load_embedding_mat(file_path):
    print('loading embedding mat from file {}'.format(file_path))
    return np.load(file_path)['glove']


def word_to_ids(vocab, data_path, target_path):
    if not os.path.exists(target_path):
        print('transform word to ids and save data to file {}'.format(target_path))
        with open(data_path, 'r', encoding='utf-8') as f_r:
            with open(target_path, 'w', encoding='utf-8') as f_w:
                for line in f_r:
                    label, sentence = line.strip().split('\t')[0], line.strip().split('\t')[1]
                    f_w.write(label + '\t')

                    words_list = sentence.strip().split(' ')
                    for word in words_list:
                        f_w.write(str(vocab.get(word, 0)) + ' ')

                    f_w.write('\n')
    else:
        print('word2ids file {} is already exists...'.format(target_path))

def preprocess(data_paths=[], vocab_path='', glove_path='', embed_mat_path=''):
    creat_vocabulary(data_paths, vocab_path)
    vocab = load_vocabulary(vocab_path)
    print('vocabulary size is {}.'.format(len(vocab)))

    word_to_ids(vocab, data_paths[0], target_path='./data/train_data.ids')
    word_to_ids(vocab, data_paths[1], target_path='./data/valid_data.ids')

    creat_embedding_mat(vocab, glove_path, embed_mat_path)

    print('preprocess data already.')


def pad_sentence(sentence, max_len, PAD_ID):
    new_sentence = sentence
    if len(sentence) < max_len:
        for i in range(max_len-len(sentence)):
            new_sentence.append(int(PAD_ID))

    return new_sentence


def generate_data(data_path, sent_len, vocab):
    x = []
    y = []
    with open(data_path, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            line = line.strip().split('\t')
            label = int(line[0])
            sentence = [int(value) for value in line[1].split(' ')]
            new_sentence = pad_sentence(sentence, sent_len, vocab['PAD'])
            if label == 1:
                y.append([1, 0])
            else:
                y.append([0, 1])

            x.append(new_sentence)

    return (x, y)

if __name__ == '__main__':

    data_paths = ['./data/train_data.txt', './data/valid_data.txt']
    save_path = './data/vocabulary.txt'
    glove_path = './data/embed/glove.42B.300d.txt'
    embed_mat_path = './data/embed/embedding_mat.npz'

    preprocess(data_paths=data_paths, vocab_path=save_path, glove_path=glove_path, embed_mat_path=embed_mat_path)











