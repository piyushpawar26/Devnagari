# imports
from flask import Flask, render_template, url_for, request
import numpy as np
import json
from createData import createData
from neuralNet import train, load_model, predict

# initializing the flask app
app = Flask(__name__)
app.secret_key = 'my_unobvious_secret_key'

# testing input which is received by http request
X_test = None

# ensuring getting data first and then testing on that data
has_got_image = False

# index routing
@app.route('/', methods=['GET', 'POST'])
def index():
	global X_test
	global has_got_image
	output = ''
	if request.method == 'POST':
		if request.form['submit'] == 'Train':
			train()
			output = 'Training done!'
		elif request.form['submit'] == 'Test':
			if has_got_image:
				model = load_model()
				output = predict(model, X_test)
				has_got_image = False
			else:
				output = 'please click on the Send Image button first!'
	return render_template("index.html", output=output)

# receiving the image here
@app.route('/req', methods=['GET', 'POST'])
def req():
	global X_test
	global has_got_image
	request_data = request.data.decode('utf-8')
	json_data = json.loads(request_data)
	string_array = json_data['input']
	main_array = np.array(np.array([[0 for _ in range(36)]]), dtype=np.float64)
	for row in string_array.split(';'):
		main_array = np.append(main_array, [np.array([float(col) for col in row.split(',')])], axis=0)
	X_test = np.reshape(np.array(main_array[1:]), [1, 36, 36, 1])
	has_got_image = True
	return "processed"

# ensuring the call from same module
if __name__=='__main__':
	app.run(debug=True)
	
