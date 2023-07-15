from flask import *
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
app = Flask(__name__)

model = pickle.load(open("bcc_classification.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def predict_api():
    data = request.json['data']
    text = [data]
    tokenizer = Tokenizer(num_words=17727, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True)
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=3000)
    pred = model.predict(padded)
    labels = ['Business','Entertainment','Politics','Sports','Tech']

@app.route("/predictions", methods=["POST"])
def predict():
    data = [request.form.values()]
    tokenizer = Tokenizer(num_words=17727, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True)
    seq = tokenizer.texts_to_sequences(data)
    padded = pad_sequences(seq, maxlen=3000)
    pred = model.predict(padded)
    labels = ['Business','Entertainment','Politics','Sports','Tech']
    output = labels[np.argmax(pred)]
    return render_template("home.html",prediction=output)


if __name__ == '__main__':
    app.run(debug=True)