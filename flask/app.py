from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np


app = Flask(__name__)


with open('trained_models/rf_model.pkl', 'rb') as model: # Random Forest
    rf_clf = pickle.load(model) 


@app.route("/", methods=["GET", "POST"])
def get_variables():
    if request.method == 'POST':
        lensep = request.form.get("lensep")
        widsep = request.form.get("widsep")
        lenpet = request.form.get("lenpet")
        widpet = request.form.get("widpet")

        # Verificar que los valores sean números válidos antes de continuar
        try:
            lensep = float(lensep)
            widsep = float(widsep)
            lenpet = float(lenpet)
            widpet = float(widpet)
        except ValueError:
            return "Por favor, ingrese valores numéricos válidos."

        return redirect(url_for('retorno', lensep=lensep, widsep=widsep, lenpet=lenpet, widpet=widpet))
    
    return render_template("home.html")


@app.route("/retorno")
def retorno(): 
    lensep = float(request.args.get('lensep'))
    widsep = float(request.args.get('widsep'))
    lenpet = float(request.args.get('lenpet'))
    widpet = float(request.args.get('widpet'))

    # Arreglo de los valores introducidos
    inputs_to_pred = np.array([lensep, widsep, lenpet, widpet]).reshape(1, -1)


    # Predicción
    rf_pred = rf_clf.predict(inputs_to_pred)

    return render_template("home.html", rf_pred=rf_pred[0])


if __name__=='__main__':
    app.run(debug=True)