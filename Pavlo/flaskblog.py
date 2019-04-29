from flask import Flask, render_template, url_for, request
import os
app = Flask(__name__)

posts = [
    {
    'author': 'Pavlo Pyvovar',
    'title': 'Blog Post 1',
    'content': 'First post content',
    'date_posted': 'April 28, 2019'
    },
    {
    'author': 'Olya Pashnieva',
    'title': 'Blog Post 2',
    'content': 'Second post content',
    'date_posted': 'April 28, 2019'
    }

]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/blog")
def blog():
    return  render_template('blog.html', posts=posts)

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
def upload():
    target = os.path.join(APP_ROOT, 'data/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    return render_template("complete.html")

@app.route("/model")
def model():
    return render_template('model.html')

if __name__ == "__main__":
    app.run(debug=True)