# Flask
from flask import Flask, render_template, request
# Data manipulation
import pandas as pd
# Matrices manipulation
import numpy as np
# Script logging
import logging
# ML model
import joblib
# JSON manipulation
import json
# Utilities
import sys
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder = 'static', template_folder = 'template')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# Function
def ValuePredictor(data = pd.DataFrame):
	# Model name
	model_name = 'bin/Logistic_Regression_model.pkl'
	# Directory where the model is stored
	model_dir = os.path.join(current_dir, model_name)
	# Load the model
	loaded_model = joblib.load(open(model_dir, 'rb'))
	# Predict the data
	result = loaded_model.predict(data)
	
	return result[0]

# Home page
@app.route('/')
def home():
	return render_template('index.html')

# Prediction page
@app.route('/prediction', methods = ['POST'])
def predict():
	if request.method == 'POST':
		# Get the data from form
		# client_id = request.form['client_id']
		name = request.form['name']
		#clerk_type = request.form['clerk_type'] # NO SE TOMO EN CUENTA
		
		# Categorical columns
		application_submission_type = request.form['application_submission_type'] # WEB o Carga
		residential_state = request.form['residential_state']
		marital_status = request.form['marital_status']
		residence_type = request.form['residence_type']
		monthly_income_tot = request.form['monthly_income_tot']

		# Numerical columns
		sex = request.form['sex']
		flag_residencial_phone = request.form['flag_residencial_phone']
		flag_professional_phone = request.form['flag_professional_phone']
		payment_day = request.form['payment_day']
		nacionality = request.form['nacionality']
		flags_cards = request.form['flags_cards']
		quant_banking_accounts_tot = request.form['quant_banking_accounts_tot']
		personal_assets_value = request.form['personal_assets_value']
		quant_cards = request.form['quant_cards']
		quant_dependants = request.form['quant_dependants']
		months_in_residence = request.form['months_in_residence']


		# Load template of JSON file containing columns name
		# Schema name
		schema_name = 'data/columns_set.json'
		# Directory where the schema is stored
		schema_dir = os.path.join(current_dir, schema_name)
		with open(schema_dir, 'r') as f:
			cols =  json.loads(f.read())
		schema_cols = cols['data_columns']

		# Parse the Categorical columns
		# APPLICATION_SUBMISSION_TYPE
		try:
			col = ('APPLICATION_SUBMISSION_TYPE_' + str(application_submission_type))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass

		# RESIDENCIAL_STATE
		try:
			col = ('RESIDENCIAL_STATE_' + str(residential_state))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass
	

		# MARITAL_STATUS
		try:
			col = ('MARITAL_STATUS_' + str(marital_status))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass
	

		# RESIDENCE_TYPE
		try:
			col = ('RESIDENCE_TYPE_' + str(residence_type))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass
	
		# MONTHLY_INCOMES_TOT
		try:
			col = ('MONTHLY_INCOMES_TOT_' + str(monthly_income_tot))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass

		# Parse the Numerical columns
		schema_cols['SEX'] = sex
		schema_cols['FLAG_RESIDENCIAL_PHONE'] = flag_residencial_phone
		schema_cols['COMPANY'] = flag_professional_phone
		schema_cols['PAYMENT_DAY'] = payment_day
		schema_cols['NACIONALITY'] = nacionality
		schema_cols['FLAG_CARDS'] = flags_cards
		schema_cols['QUANT_BANKING_ACCOUNTS_TOT'] = quant_banking_accounts_tot
		schema_cols['PERSONAL_ASSETS_VALUE'] = personal_assets_value
		schema_cols['QUANT_CARS'] = quant_cards
		schema_cols['QUANT_DEPENDANTS'] = quant_dependants
		schema_cols['MONTHS_IN_RESIDENCE'] = months_in_residence

		# Convert the JSON into data frame
		df = pd.DataFrame(
				data = {k: [v] for k, v in schema_cols.items()},
				dtype = float
			)

		# Create a prediction
		print('ACA LLEGO')
		print(df.dtypes)
		result = ValuePredictor(data = df)

		# Determine the output
		if int(result) == 1:
			prediction = 'Dear Mr/Mrs/Ms {name}, your loan is approved!'.format(name = name)
		else:
			prediction = 'Sorry Mr/Mrs/Ms {name}, your loan is rejected!'.format(name = name)

		# Return the prediction
		return render_template('prediction.html', prediction = prediction)
	
	# Something error
	else:
		# Return error
		return render_template('error.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug = True)