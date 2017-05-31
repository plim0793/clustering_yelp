# -*- coding: utf-8 -*-
import logging
import json

from flask import Flask, jsonify, render_template, redirect, url_for, send_from_directory, request, send_file
from flask_wtf import FlaskForm
from wtforms import fields
from wtforms.validators import Required, InputRequired

from sklearn.externals import joblib
import numpy as np
import pandas as pd

WTF_CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

debug = True


# create the application object
app = Flask(__name__)
app.config.from_object("config")

# unpickle my model


class PredictForm(FlaskForm):
    """Fields for Predict"""
    preference = fields.TextField('Preference:', [InputRequired()])

    submit = fields.SubmitField('Submit')



@app.route('/', methods=['GET','POST'])
def index():

	if request.method == 'POST':
		form = PredictForm(request.form)
		print('FORM: ', form)
		data = json.loads(request.data.decode('utf-8'), strict=False)

	else:
		form = PredictForm()
		data = {'preference': 'sample preference'}

	print("PREFERENCE: ", data)
	top_10 = ''
	preferences = ''


	if request.method == 'POST':
		return jsonify(
					top_10=top_10
				)


	return render_template('index.html',
							form=form,
							top_10=top_10)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)



