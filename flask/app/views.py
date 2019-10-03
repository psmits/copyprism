# views.py

import os
# import urllib
# import sys
# import io
# import string
# from random import randint
from werkzeug.utils import secure_filename
from flask import flash, request, redirect, render_template
from flask import url_for, send_from_directory
from app import app
import nltk
from copyprism_utilities import detect_labels, load_doc, clean_seq_gen
from copyprism_utilities import sequence_gen  # , replace_nouns
import tensorflow as tf
import gpt_2_simple as gpt2
from itertools import compress


# auth so the whole thing runs
js = "./auth/copyprisim-20edfacb40ce.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = js


# fail well on images
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# load model
sess = gpt2.start_tf_sess()

chp_dir = './app/static/model/checkpoint'
gpt2.load_gpt2(sess,
               run_name='run1',
               checkpoint_dir=chp_dir)
graph = tf.get_default_graph()


# bring in testing sequences
in_filename = './app/static/model/ikea_word_test_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# # bring in ikea product descriptions
# in_filename = './app/static/model/ikea_2.csv'
# ikea_cat = pd.read_csv(in_filename)
# ikds = ikea_cat.description


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@app.route('/copyprism', methods=['GET', 'POST'])
def upload_form():
    return render_template('upload.html')


@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('no file part')
                return redirect(request.url)

            file = request.files['file']

            # if user does not select file, browser submits an empty part w/o
            # filename
            if file.filename == '':
                flash('no selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # google vision image labels
                lab = detect_labels(filepath)
                lab = [x.lower() for x in lab]

                # only nounds from lab
                pos = nltk.pos_tag(lab)
                tt = []
                for ii in range(0, len(pos)):
                    tt.append(pos[ii][1][0] == 'N')

                lab = list(compress(lab, tt))

                # given best lines...
                # seed_text = lines[randint(0, len(lines))]
                # new_seed = replace_nouns(seed_text, lab)
                new_seed = 'This is a ' + lab[0] + '.'

                # generate some text
                text = sequence_gen(sess=sess,
                                    prefix=new_seed,
                                    checkpoint_dir=chp_dir,
                                    length=50,
                                    temperature=0.7,
                                    nsamples=3,
                                    batch_size=1)

                text = clean_seq_gen(text=text, to_rm=new_seed)

                filename = 'upload/' + filename

                return render_template('result.html',
                                       filepath=filename,
                                       lab=lab[0],
                                       seed=new_seed,
                                       gen=text)
        return render_template('upload.html')
