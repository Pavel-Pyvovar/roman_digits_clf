from flask import Flask, render_template, url_for, request
import os, sys, json
import numpy as np

sys.path.append(os.path.abspath\
    (os.path.join(__file__, '../../data')))
sys.path.append(os.path.abspath(os.path.join\
    (__file__, '../../model/utils')))
sys.path.append(os.path.abspath(os.path.join\
    (__file__, '../../model/scr')))
sys.path.append(os.path.abspath(os.path.join\
    (__file__, '../../model')))

from prepare_data import resize_grey_and_save
from utils import get_args
from config import process_config
from src.model import Model

app = Flask(__name__)

@app.route("/")
def model():
    return render_template('model.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/model", methods=['POST'])
def classify():
    uploaded = os.path.join(APP_ROOT, '../uploaded')
    uploaded_clean = os.path.join(APP_ROOT, '../uploaded_clean')
    predicted = 0
    args = get_args()
    config = process_config(args.config)
    m = Model(config)
    m.load_model()
    if not os.path.isdir(uploaded) and \
            not os.path.isdir(uploaded_clean):
        try:
            os.mkdir(uploaded)
            os.mkdir(uploaded_clean)
        except Exception as e:
            print("Directory already exists")

    for file in request.files.getlist("file"):
        file_path = file.filename
        print(file_path)
        destination = "/".join([uploaded, file_path])
        file.save(destination)
        file_name = destination.split('/')[-1]
        print(file_name)
        path_to_img = os.path.abspath(os.path.join(uploaded_clean, file_name))
        array = resize_grey_and_save(destination, path_to_img, config.image_size)
        array = np.stack((array, ) , axis=0)
        pred_arr = m.predict_proba(array)
        predicted = np.argmax(pred_arr) + 1
        print(predicted)
    

    return render_template("model.html", pred_class=predicted)


if __name__ == "__main__":
    app.run(debug=True)