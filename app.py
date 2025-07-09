from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("best_xgb.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/webpredict", methods=["POST"])
def webpredict():
    try:
        age = int(request.form["Age"])
        vehicle_age_raw = request.form["Vehicle_Age"]
        annual_premium = float(request.form["Annual_Premium"])
        gender = request.form["Gender"]
        damage = request.form["Damage"]
        driving_license = int("Driving_License" in request.form)  
        
        vintage_days = int(request.form["Vintage_days"])
        vintage_months = round(vintage_days / 30.0, 1)
  
        region = int(request.form["Region_Code"])
        previously_insured = int(request.form["Previously_Insured"])

        vehicle_age = {
            "less_than_1": 0,
            "between_1_2": 1,
            "more_than_2": 2
        }.get(vehicle_age_raw, 0)

        gender_male = 1 if gender == "male" else 0

        damage_yes = 1 if damage == "yes" else 0

        annual_premium_log = np.log1p(annual_premium)

        features = [
            age, vehicle_age, annual_premium_log, gender_male,
            damage_yes, driving_license, vintage_months, region, previously_insured
        ]
        input_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(input_array)[0])
        probability = float(model.predict_proba(input_array)[0][1])

        return render_template("result.html", prediction=prediction, probability=round(probability, 4))

    except Exception as e:
        return f"Error: {e}", 500

@app.route("/moreinfo", methods=["GET"])
def more_info():
    return render_template("moreinfo.html")


if __name__ == "__main__":
    app.run(debug=True)
