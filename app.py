#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app = Flask(__name__, template_folder='templates')

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)