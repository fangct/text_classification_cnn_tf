import data_utils
import tensorflow as tf
import cnn_model
import numpy as np
FLAGS = tf.flags.FLAGS

#parameters

# data parameters
tf.flags.DEFINE_string("train_data_file", "./data/train_data.txt", "Data source for the train data.")
tf.flags.DEFINE_string("valid_data_file", "./data/valid_data.txt", "Data source for the valid data.")
tf.flags.DEFINE_string("save_embedding_file", "./data/embed/embed_mat.npz", "Embeddings which contains the word from data")
tf.flags.DEFINE_string("glove_file", "./data/embed/glove.42B.300d.txt", "glove embedding downloaded.")
tf.flags.DEFINE_string("vocabulary_file", "./data/vocabulary.txt", "Words in data")
tf.flags.DEFINE_string("model_path", "./model/cnn.model", "save model")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding(default: 300)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("filter_nums", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate (default: 0.001)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_sentence_len", 100, "The max length of sentence (default: 100)")
tf.flags.DEFINE_integer("label_nums", 2, "Number of categories")

# tf.flags.DEFINE_string("input_layer_type", "CNN-non-static", "Type of input layer,CNN-rand,CNN-static,CNN-non-static,CNN-multichannel	 (default: 'CNN-rand')")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 16)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")


def train():
    embedding_mat = data_utils.load_embedding_mat(FLAGS.save_embedding_file)
    vocab = data_utils.load_vocabulary(FLAGS.vocabulary_file)
    model = cnn_model.CNN_MODEL(embed_dim=FLAGS.embedding_dim,
                                filter_sizes=FLAGS.filter_sizes,
                                max_sent_len=FLAGS.max_sentence_len,
                                embedding_mat=embedding_mat,
                                word_nums=len(vocab),
                                filter_nums=FLAGS.filter_nums,
                                label_nums=FLAGS.label_nums,
                                learning_rate=FLAGS.learning_rate,
                                model_path=FLAGS.model_path,
                                epoch=FLAGS.num_epochs,
                                batch_size=FLAGS.batch_size,
                                dropout_prob=FLAGS.dropout_keep_prob)

    train_data = data_utils.generate_data('./data/train_data.ids', FLAGS.max_sentence_len, vocab)
    valid_data = data_utils.generate_data('./data/valid_data.ids', FLAGS.max_sentence_len, vocab)
    print('train data size is {}, valid data size is {}.'.format(len(train_data[0]), len(valid_data[0])))

    model.train(train_data, valid_data)


if __name__ == '__main__':

    # preprocess data
    data_utils.preprocess(data_paths=[FLAGS.train_data_file, FLAGS.valid_data_file],
                          vocab_path=FLAGS.vocabulary_file,
                          glove_path=FLAGS.glove_file,
                          embed_mat_path=FLAGS.save_embedding_file)
    train()