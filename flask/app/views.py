# views.py

import os
from app import app
from flask import request, render_template
from werkzeug.utils import secure_filename
from google.cloud import vision
import io
import pickle
from random import randint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer
import string
from itertools import compress


# auth so the whole thing runs
my_dir = "/home/peter/Documents/projects/insight/copyprisim"
js = "/auth/copyprisim-20edfacb40ce.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = my_dir + js


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


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


def clean_text(input):
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


# load the pickle-d model
# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
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

    # send it all out
    return ' '.join(result)


# load model
model = load_model(my_dir + '/results/ikea_word_model.h5')
# load tokenizer
tokenizer = pickle.load(open(my_dir + '/results/word_tokenizer.pkl', 'rb'))

# bring in testing sequences
in_filename = '/results/ikea_word_test_sequences.txt'
doc = load_doc(my_dir + in_filename)
lines = doc.split('\n')

seq_length = len(lines[0].split()) - 1


@app.route('/', methods=['GET', 'POST'])
def upload_form():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # put the file someplace safe
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        lab = detect_labels(os.path.join(app.config['UPLOAD_FOLDER'],
                                         filename))
        lab = ' '.join(lab)
        res = clean_text(lab)

        # combine res with random seed text

        # generate text from seed
        seed_text = lines[randint(0, len(lines))]

        seed_split = seed_text.split()
        new_seed = seed_split[:-len(res)] + res
        # new_seed = seed_text[:-len(res)] + ' ' + res_string
        new_seed = ' '.join(new_seed)

        # put into generator
        generated = generate_seq(model, tokenizer, seq_length, new_seed, 50)

        full_filename = os.path.join(app.config['UPLOAD_FOLDER'],
                                     filename)
        return render_template('result.html',
                               uimg=full_filename,
                               lab=lab,
                               res=new_seed,
                               gen=generated)


if __name__ == '__main__':
    app.run(debug=True)
