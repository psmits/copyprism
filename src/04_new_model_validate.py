import pandas as pd
import random
from copyprism_utilities import sequence_gen, clean_text, load_doc
import gpt_2_simple as gpt2


random.seed(952)


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

# load gpt2 model stuff
sess = gpt2.start_tf_sess()
chp_dir = '/home/peter/Documents/projects/insight/checkpoint/'
gpt2.load_gpt2(sess,
               run_name='run1',
               checkpoint_dir=chp_dir)


opens = []
refs = []
gens = []
rans = []
for tl in test_long:
    exam = tl.split()

    opener = exam[:50]  # seed text
    opens.append(' '.join(opener))

    closer = exam[50:]  # reference text
    refs.append(' '.join(closer))

    # fill in rest of description
    gen = sequence_gen(sess,
                       prefix=' '.join(opener),
                       checkpoint_dir=chp_dir,
                       length=len(closer),
                       temperature=0.7,
                       nsamples=1,
                       batch_size=1)

    gen = gen[0].replace('\n', '')
    gen = gen.replace(' '.join(opener), '')
    gens.append(gen)

    rand_tokens = test_tokens
    random.shuffle(rand_tokens)
    rand_out = ' '.join(rand_tokens[:len(closer)])
    rans.append(rand_out)


output = pd.DataFrame(list(zip(opens, refs, gens, rans)),
                      columns=['start',
                               'reference',
                               'generated',
                               'random'])

output.to_csv('../results/new_model_text_comparison.csv')
