from flask import Blueprint, render_template, request, flash, redirect, url_for
from .knn_predict import predict_knn, loading
from .calc_corr import calc_corr

model, scaler, encode_dict = loading()
feature_name = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC",
    "FAF", "TUE", "CALC", "MTRANS"
]

views = Blueprint("views", __name__)


@views.route("/")
@views.route("/home")
def home():
    return render_template("home.html", name="h2ome")


@views.route('/submit-predict', methods=['POST'])
def submit_predict():
    features = [
        float(request.form['gender']),
        float(request.form['age']),
        float(request.form['height']),
        float(request.form['weight']),
        float(request.form['family_history']),
        float(request.form['favc']),
        float(request.form['fcvc']),
        float(request.form['ncp']),
        float(request.form['caec']),
        float(request.form['smoke']),
        float(request.form['ch20']),
        float(request.form['scc']),
        float(request.form['faf']),
        float(request.form['tue']),
        float(request.form['calc']),
        float(request.form['mtrans']),
    ]

    prediction = predict_knn(model, scaler, features, feature_name)

    prediction_label = encode_dict['NObeyesdad'][prediction]  # str

    return render_template('result.html', prediction=prediction_label)


@views.route("/data-exploration")
def explore_data():
    return render_template("data_explorations.html")


@views.route("/data-visulization")
def visualize_data():
    corr_json = calc_corr()
    return render_template("data_visualizations.html", corr_json=corr_json)


@views.route("/predict")
def predict():
    return render_template("predict.html")
