from flask import Flask, render_template, request
import pandas as pd
import os
import plotly.express as px
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- GEMINI CLIENT ----------------
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

uploaded_df = None
# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload():

    global uploaded_df

    if request.method == "POST":

        file = request.files["dataset"]
        chart_type = request.form.get("chart_type", "bar")

        if file:

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # READ FILE
            if file.filename.endswith(".csv"):
                df = pd.read_csv(filepath)

            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(filepath)

            else:
                return "Only CSV and Excel files are allowed"

            uploaded_df = df

            # CLEAN COLUMN NAMES
            df.columns = df.columns.str.strip()

            # BASIC INFORMATION
            preview = df.head().to_html(classes="table table-bordered")

            missing_values = df.isnull().sum().sum()

            columns = list(df.columns)

            dtypes = df.dtypes.astype(str).to_dict()

            summary = df.describe(include='all').to_html()

            # CHART GENERATION
            chart_html = ""

            numeric_cols = df.select_dtypes(
                include=['int64', 'float64']
            ).columns

            text_cols = df.select_dtypes(
                include=['object']
            ).columns

            if len(numeric_cols) > 0 and len(text_cols) > 0:

                if chart_type == "bar":

                    fig = px.bar(
                        df,
                        x=text_cols[0],
                        y=numeric_cols[0],
                        title="Bar Chart"
                    )

                elif chart_type == "line":

                    fig = px.line(
                        df,
                        x=text_cols[0],
                        y=numeric_cols[0],
                        title="Line Chart"
                    )

                elif chart_type == "pie":

                    fig = px.pie(
                        df,
                        names=text_cols[0],
                        values=numeric_cols[0],
                        title="Pie Chart"
                    )

                elif chart_type == "scatter":

                    fig = px.scatter(
                        df,
                        x=text_cols[0],
                        y=numeric_cols[0],
                        title="Scatter Plot"
                    )

                chart_html = fig.to_html(full_html=False)

            return render_template(
                "dashboard.html",
                tables=preview,
                rows=df.shape[0],
                cols=df.shape[1],
                missing=missing_values,
                columns=columns,
                dtypes=dtypes,
                summary=summary,
                chart=chart_html
            )

    return render_template("upload.html")
# ---------------- AI CHAT ----------------
@app.route("/ask", methods=["POST"])
def ask():

    global uploaded_df

    question = request.form["question"]

    if uploaded_df is None:
        return "Please upload a dataset first."

    dataset_preview = uploaded_df.head(20).to_string()

    prompt = f"""
    You are an expert data analyst.

    Dataset:
    {dataset_preview}

    User Question:
    {question}

    Give a clear answer based only on the dataset.
    """

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"AI Error: {str(e)}"
# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)