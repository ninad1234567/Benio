from flask import Flask, jsonify
import jinja2
from flask import render_template,request
from chatbot import *

app = Flask(__name__)

import json

# try again
@app.route('/', methods=['GET', 'POST'])
def process_input():

    if request.method == 'POST':

        input_text = request.form.get('input_text')

        processed_output = process_text(input_text)

        #return json.dumps(processed_output)
        return render_template('main.html',processed_output = processed_output, input_text = input_text)

    return render_template('main.html')


if __name__ == "__main__":
    app.run(debug=True)



