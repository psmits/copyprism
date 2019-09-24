import pandas as pd
import random
from pickle import load
from sklearn.metrics.pairwise import cosine_similarity
from copyprisim_utilities import generate_seq, clean_text, load_doc
from keras.models import load_model


random.seed(952)


# load the model
model = load_model('../results/ikea_word_model.h5')

# load the tokenizer
tokenizer = load(open('../results/word_tokenizer.pkl', 'rb'))

# load the testing data
in_filename = '../results/ikea_word_test_sequences.txt'
test_sequences = load_doc(in_filename)
test_lines = test_sequences.split('\n')

# move on to processing the test set into the right shape
# i've split train/test by objects, not sequences
# make the testing data the right shape to test with
ikea_test = pd.read_csv('../results/ikea_word_test.csv')

# get word tokens from corpus
test_desc_single = ' '.join(ikea_test.description)
test_tokens = clean_text(test_desc_single)

# seq length for text
seq_length = len(test_lines[0].split()) - 1

test_long = []
for item in ikea_test.description:
    if len(item.split()) > 60:
        test_long.append(item)


opens = []
refs = []
gens = []
rans = []
distances = []
for tl in test_long:
    exam = clean_text(tl)

    opener = exam[:50]  # seed text
    opens.append(' '.join(opener))

    closer = exam[50:]  # reference text
    refs.append(' '.join(closer))

    # fill in rest of description
    res = generate_seq(model,
                       tokenizer,
                       seq_length,
                       ' '.join(opener),
                       len(closer))
    # print(res)

    #
    rand_tokens = test_tokens
    random.shuffle(rand_tokens)
    rand_out = ' '.join(rand_tokens[:len(closer)])
    gens.append(res)

    # keep text snippets

    # to liked format
    ref = tokenizer.texts_to_matrix([' '.join(closer)], mode='tfidf')[0]
    gen = tokenizer.texts_to_matrix([res], mode='tfidf')[0]
    ran = tokenizer.texts_to_matrix([rand_out], mode='tfidf')[0]
    rans.append(rand_out)

    ref_a = ref.reshape(1, len(ref))
    gen_a = gen.reshape(1, len(gen))
    ran_a = ran.reshape(1, len(ran))

    # just using cosine similarity
    # each vector is the tf-idf embedding
    ref2gen = cosine_similarity(ref_a, gen_a)[0][0]

    ref2ran = cosine_similarity(ref_a, ran_a)[0][0]

    # how much closer is gen to ref than ran is to ref
    distances.append(ref2gen - ref2ran)


output = pd.DataFrame(list(zip(opens, refs, gens, rans, distances)),
                      columns=['start',
                               'reference',
                               'generated',
                               'random',
                               'distance'])
output.to_csv('../results/text_comparison.csv')
