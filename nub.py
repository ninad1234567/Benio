from flask import Flask, render_template,request
import jinja2
app = Flask(__name__)

import chatbot
from chatbot import *
@app.route('/')
def home():
   return render_template('wow.html')

@app.route("/get")
def get_response():
    textInput = request.args.get('msg')
    return str(chatbot.get_response())



if __name__ == '__main__':
   app.run()