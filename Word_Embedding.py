import fasttext
import re


def train_model():
    model_train = fasttext.train_unsupervised('./code-fall2019-a3/NLP_class/data5/training-data/training-data.1m',
                                              "skipgram",
                                              thread=4,
                                              epoch=10,
                                              lr=.05)
    model_train.save_model("./output/skip.bin")
    return model_train


def generate_output(model):
    (output_dim, feature_dim) = model.get_output_matrix().shape
    word_list = model.get_words()

    with open('./output/word_vecs.txt', 'w', encoding='utf-8') as fp:
        print("writing to output...")
        fp.write("%d %d\n" % (output_dim, feature_dim))
        for word in word_list:
            s = str(word).encode('utf8')
            fp.write('%s ' % s.decode('utf8'))

            s_list = str(list(model.get_word_vector(word))).strip("[]")
            r_sub = re.sub(",", "", s_list)
            fp.write(r_sub)
            fp.write('\n')
        print("writing done.")
        fp.close()


model = train_model()
# model = fasttext.load_model("./output/cbow.bin")

generate_output(model)
