from flask import Flask, render_template,request
import pandas as pd
from xgboost import XGBClassifier
app=Flask(__name__)



model=XGBClassifier()
model.load_model("Notebook and Dataset/xgboost_model.pkl")


def predict_survival(passenger_data):
  print("=========================")
  print(passenger_data['Pclass'])
  print(passenger_data['Sex'])
  print(passenger_data['Age'])
  print(passenger_data['Embarked'])
  prediction = model.predict([[passenger_data["Pclass"], passenger_data["Sex"], passenger_data["Age"],passenger_data['Embarked']]])[0]
  survival_label = "Survived" if prediction > 0.5 else "Not Survived"
  return survival_label

@app.route("/", methods=["GET", "POST"])
def index():
    
    if request.method == "POST":
        []
        passenger_data = {

        "Pclass": int(request.form["Pclass"]),
        "Sex": request.form["Sex"],
        "Age": float(request.form["Age"]),
        "Embarked": request.form["Embarked"]
        }
        if passenger_data["Embarked"]=="S":
            passenger_data["Embarked"]=0
        elif passenger_data["Embarked"]=="C":
            passenger_data["Embarked"]=1
        else:
            passenger_data["Embarked"]=2


        if passenger_data["Sex"]=="male":
            passenger_data["Sex"]=0
        else:
            passenger_data['Sex']=1


        print(passenger_data)
        prediction = predict_survival(passenger_data)
        return render_template("index.html", prediction=prediction)
    else:
        return render_template("index.html")


if __name__ == "__main__":
  app.run(debug=True)