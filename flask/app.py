from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

with open('trained_models/rf_model.pkl', 'rb') as model: # Random Forest
    rf_clf = pickle.load(model) 

app = Flask('Iris')

def validate_number(value):
    try:
        return float(value)
    except ValueError:
        return None

def validate_input(lensep, widsep, lenpet, widpet):
    lensep = validate_number(lensep)
    widsep = validate_number(widsep)
    lenpet = validate_number(lenpet)
    widpet = validate_number(widpet)

    return lensep, widsep, lenpet, widpet

@app.route("/", methods=["GET", "POST"])
def get_variables():
    if request.method == 'POST':
        lensep = request.form.get("lensep")
        widsep = request.form.get("widsep")
        lenpet = request.form.get("lenpet")
        widpet = request.form.get("widpet")

        lensep, widsep, lenpet, widpet = validate_input(lensep, widsep, lenpet, widpet)

        # Comprobamos si alguna de las entradas no es válida (None)
        if None in [lensep, widsep, lenpet, widpet]:
            return "Por favor, ingrese valores numéricos válidos."

        return redirect(url_for('retorno', lensep=lensep, widsep=widsep, lenpet=lenpet, widpet=widpet))
    
    return render_template("home.html")


@app.route("/retorno")
def retorno(): 
    lensep = request.args.get('lensep')
    widsep = request.args.get('widsep')
    lenpet = request.args.get('lenpet')
    widpet = request.args.get('widpet')

    lensep, widsep, lenpet, widpet = validate_input(lensep, widsep, lenpet, widpet)

    # Comprobamos si alguna de las entradas no es válida (None)
    if None in [lensep, widsep, lenpet, widpet]:
        return "Por favor, ingrese valores numéricos válidos."

    # Arreglo de los valores introducidos
    inputs_to_pred = np.array([lensep, widsep, lenpet, widpet]).reshape(1, -1)

    # Predicción
    rf_pred = rf_clf.predict(inputs_to_pred)

    return render_template("home.html", rf_pred=rf_pred[0])


if __name__=='__main__':
    app.run(debug=True)
