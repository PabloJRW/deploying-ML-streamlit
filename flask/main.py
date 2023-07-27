import os
from flask import Flask, render_template



template_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
template_dir = os.path.join(template_dir, 'flask', 'templates')


app = Flask(__name__, template_folder=template_dir)


@app.route("/")
def predict():
    return render_template("predict.html")    



if __name__=='__main__':
    app.run(debug=True)