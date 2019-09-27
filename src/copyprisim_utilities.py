import string
import nltk
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from itertools import compress, islice, cycle
from google.cloud import vision
import io

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def clean_text(input):
    '''clean text prior to analysis'''
    # tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove non alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # make lower case
    tokens = [word.lower() for word in tokens]

    # remove tokens of length 1
    tokens_len = [len(i) > 1 for i in tokens]
    tokens_filter = list(compress(tokens, tokens_len))
    tokens = tokens_filter

    return tokens


def replace_nouns(text, replace):
    '''replace nouns in a string with other nouns'''
    tokenized = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokenized)

    tt = []
    for ii in range(0, len(tagged)):
        tt.append(tagged[ii][1][0] == 'N')

    replacements = list(islice(cycle(replace), sum(tt)))

    jj = 0
    for ii in range(0, len(tagged)):
        if tt[ii]:
            tokenized[ii] = replacements[jj]
            jj = jj + 1

    return ' '.join(tokenized)


def save_doc(lines, filename):
    '''save tokens to file, one sequence per line'''
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_doc(filename):
    '''open sequence file'''
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    '''generate a sequence from a language model'''
    result = list()
    in_text = seed_text

    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]

        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')

        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break

        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


def detect_labels(path):
    """Detects labels in LOCAL file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations

    # list of labels (ignoring uncertainty)
    labels = [x.description for x in labels]
    return labels
