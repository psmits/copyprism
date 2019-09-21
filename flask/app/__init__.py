# app/__init__.py

from flask import Flask
import os

# initialize the app
app = Flask(__name__, instance_relative_config=True)

app.config['UPLOAD_FOLDER'] = os.getcwd() + '/app/upload'

# load the views
from app import views

# load the config file
app.config.from_object('config')
