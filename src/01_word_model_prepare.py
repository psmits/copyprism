import pandas as pd
from sklearn.model_selection import train_test_split
from copyprisim_utilities import clean_text, save_doc

ikea_items = pd.read_csv('ikea_2.csv')

# some items do not have descriptions from the specific box
ikea_items = ikea_items.dropna()

# some descriptions are identical
desc_uni = ikea_items.drop_duplicates(subset='description')

# split train and test
desc_train, desc_test = train_test_split(desc_uni, test_size=0.2)
pd.DataFrame(desc_train).to_csv('ikea_word_train.csv')
pd.DataFrame(desc_test).to_csv('ikea_word_test.csv')

# make one corpus
desc_single = ' '.join(desc_train.description)

# tokenize that shit
tokens = clean_text(desc_single)

print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# make sequences of words from the full corpus
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)

print('Total Sequences: %d' % len(sequences))

out_filename = 'ikea_word_train_sequences.txt'
save_doc(sequences, out_filename)
