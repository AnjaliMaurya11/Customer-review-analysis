from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


#@app.route("/predict", methods=["POST"])
#def predict():

#    review = request.form["review"]

#    # temporary logic (we will replace with ML model later)
#    if "good" in review.lower():
#        result = "Positive Review"
#    else:
#        result = "Negative Review"

#    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)