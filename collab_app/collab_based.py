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

# unpickle objects
df_dists = joblib.load('df_dists.pkl')

class PredictForm(FlaskForm):
    """Fields for Predict"""
    preference = fields.TextField('Business Name:', [InputRequired()])

    submit = fields.SubmitField('Submit')



@app.route('/', methods=['GET','POST'])
def index():

	if request.method == 'POST':
		form = PredictForm(request.form)
		print('FORM: ', form)
		req = request.data.decode("utf-8")
		print(bool(req))
		if not req:
			req = "sample"
		data = json.loads(request.data.decode('utf-8'), strict=False)
		print(data)

	else:
		form = PredictForm()
		data = {'preference': 'sample preference'}

	print("PREFERENCE: ", data)

	input_pref = data['preference']
	input_pref = input_pref.split(', ')
	for inp in input_pref:
		inp = inp.replace(',', '')

	print("INPUT PREF: ", input_pref)
	top_10_l = ["Invalid Business Name.  Try Again."]

	try:
		pref_sum = df_dists[input_pref].apply(lambda row: np.sum(row), axis=1)
		pref_sum_10 = pref_sum.sort_values(ascending=False)[:10]
		top_10_list = list(pref_sum_10.index)
		top_1 = top_10_list[0]
		top_2 = top_10_list[1]
		top_3 = top_10_list[2]
		top_4 = top_10_list[3]
		top_5 = top_10_list[4]
		top_6 = top_10_list[5]
		top_7 = top_10_list[6]
		top_8 = top_10_list[7]
		top_9 = top_10_list[8]
		top_10 = top_10_list[9]
		top_10_l = ''
		print("TOP 10: ", top_10_list)

	except:
		top_10_l = "Invalid Business Name.  Try Again."
		top_1 = ''
		top_2 = ''
		top_3 = ''
		top_4 = ''
		top_5 = ''
		top_6 = ''
		top_7 = ''
		top_8 = ''
		top_9 = ''
		top_10 = ''

	

	if request.method == 'POST':
		return jsonify(
					top_1 = top_1,
					top_2 = top_2,
					top_3 = top_3,
					top_4 = top_4,
					top_5 = top_5,
					top_6 = top_6,
					top_7 = top_7,
					top_8 = top_8,
					top_9 = top_9,
					top_10 = top_10,
					top_10_l = top_10_l
				)


	return render_template('index.html',
							form=form,
							top_1 = top_1,
							top_2 = top_2,
							top_3 = top_3,
							top_4 = top_4,
							top_5 = top_5,
							top_6 = top_6,
							top_7 = top_7,
							top_8 = top_8,
							top_9 = top_9,
							top_10 = top_10,
							top_10_l = top_10_l)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)



