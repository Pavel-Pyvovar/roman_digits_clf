from flask import Flask, render_template, url_for, request
import os, sys, json
import numpy as np
app = Flask(__name__)

sys.path.append(os.path.abspath(os.path.join(__file__, '../../data')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../model/utils')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../model/scr')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../model')))

from prepare_data import resize_grey_and_save
from utils import get_args
from config import process_config
from src.model import Model

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html', title='Today')

@app.route("/wiki")
def wiki():
    return render_template('wiki.html')

@app.route("/data")
def data():
    return render_template('data.html')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/data/upload", methods=['POST'])
def classify():
    uploaded = os.path.join(APP_ROOT, '../uploaded')
    uploaded_clean = os.path.join(APP_ROOT, '../uploaded_clean')
    print(uploaded_clean)
    config_json = os.path.abspath(os.path.join(APP_ROOT, '../model/configs/roman.json'))
    config = process_config(config_json)
    img_size = config.image_size
    m = Model(config)
    if not os.path.isdir(uploaded) and \
            not os.path.isdir(uploaded_clean):
        try:
            os.mkdir(uploaded)
            os.mkdir(uploaded_clean)
        except Exception as e:
            print("Directory already exists")

    for file in request.files.getlist("file"):
        file_path = file.filename
        destination = "/".join([uploaded, file_path])
        file.save(destination)
        file_name = destination.split('/')[-1]
        path_to_img = os.path.abspath(os.path.join(uploaded_clean, file_name))
        array = resize_grey_and_save(destination, path_to_img, img_size)
        array = np.stack((array, ) , axis=0)
        pred_arr = m.predict_proba(array, batch=1)
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        for item in pred_arr: print(item)
        pred_arr.shape
        print('\n\n\n\n\n\n\n\n\n\n\n\n')
        predicted = np.argmax(pred_arr) + 1

    return render_template("complete.html", pred_class=predicted)


@app.route("/model")
def model():
    return render_template('model.html')


if __name__ == "__main__":
    app.run(debug=True)
