# views.py

import os
# import urllib
# import sys
# import io
# import string
import pickle
from random import randint
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import flash, request, redirect, render_template
from flask import url_for, send_from_directory
from app import app
from copyprisim_utilities import detect_labels, load_doc, clean_text
from copyprisim_utilities import replace_nouns, generate_seq


# auth so the whole thing runs
js = "./auth/copyprisim-20edfacb40ce.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = js


# fail well on images
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# load model
model = load_model('./app/static/model/ikea_word_model.h5')
model._make_predict_function()
# load tokenizer
tokenizer = pickle.load(open('./app/static/model/word_tokenizer.pkl', 'rb'))

# bring in testing sequences
in_filename = './app/static/model/ikea_word_test_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

seq_length = len(lines[0].split()) - 1


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def upload_form():
    return render_template('upload.html')


@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('no file part')
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, brower also submit an empty part w/o
        # filename
        if file.filename == '':
            flash('no selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            lab = detect_labels(filepath)
            lab = ' '.join(lab)
            res = clean_text(lab)

            # combine res with random seed text
            # generate text from seed
            seed_text = lines[randint(0, len(lines))]

            # replace nouns of seed text with image tags
            new_seed = replace_nouns(seed_text, res)

            # put into generator
            generated = generate_seq(model=model,
                                     tokenizer=tokenizer,
                                     seq_length=seq_length,
                                     seed_text=new_seed,
                                     n_words=50)

            return render_template('result.html',
                                   lab=lab,
                                   res=new_seed,
                                   gen=generated)
    return render_template('upload.html')
