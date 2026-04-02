from flask import Flask, render_template
import os

from app.preprocessing import preprocess_data


app = Flask(__name__)

TEMPLATE_FILE = "index.html"


@app.route("/", methods=["GET"])
def home():
    return render_template(TEMPLATE_FILE)


@app.route("/predict", methods=["POST"])
def predict():

    try:
        file_path = os.path.join(
            "app",
            "dataset",
            "Womens Clothing E-Commerce Reviews.csv"
        )

        df = preprocess_data(file_path)

        if df is None:
            return render_template(
                TEMPLATE_FILE,
                prediction="❌ Preprocessing failed"
            )

        sample_output = df["Cleaned Review"].head(5).tolist()

        return render_template(
            TEMPLATE_FILE,
            prediction="✅ Data cleaned successfully",
            reviews=sample_output
        )

    except Exception as e:
        return render_template(
            TEMPLATE_FILE,
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)