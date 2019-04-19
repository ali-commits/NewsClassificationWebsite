from flask import Flask, render_template, url_for, flash, redirect, request
from classify import classifier

model = classifier()

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


@app.route("/",  methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        return render_template('result.html', results=model.classify(request.form['text']))
    return render_template('home.html')
    

@app.route("/about")
def about():
    return render_template('help.html')


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html', title='404'), 404


if __name__ == '__main__':
    app.run(debug=True)
