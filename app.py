import io
import os
import pandas as pd
from flask import (
    Flask, render_template, request,
    send_file, flash, redirect, url_for, session
)
from preprocess import preprocess_dataset

app = Flask(__name__)
app.secret_key = "smart-preprocessing-secret-key"

# In-memory store for the cleaned CSV (avoids writing to disk)
_cleaned_csv_buffer = {}


@app.route("/", methods=["GET", "POST"])
def home():
    report = None

    if request.method == "POST":

        # ── Validate file presence ──────────────────────────
        if "file" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(url_for("home"))

        file = request.files["file"]

        if file.filename == "":
            flash("Please select a CSV file before submitting.", "error")
            return redirect(url_for("home"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "error")
            return redirect(url_for("home"))

        # ── Read & preprocess ───────────────────────────────
        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f"Could not read the file: {e}", "error")
            return redirect(url_for("home"))

        if df.empty:
            flash("The uploaded CSV is empty.", "error")
            return redirect(url_for("home"))

        try:
            cleaned_df, report = preprocess_dataset(df)
        except Exception as e:
            flash(f"Preprocessing failed: {e}", "error")
            return redirect(url_for("home"))

        # ── Store cleaned CSV in memory for download ────────
        buf = io.StringIO()
        cleaned_df.to_csv(buf, index=False)
        _cleaned_csv_buffer["data"] = buf.getvalue()

        flash("Dataset preprocessed successfully!", "success")

    return render_template("index.html", report=report)


@app.route("/download")
def download():
    csv_data = _cleaned_csv_buffer.get("data")
    if not csv_data:
        flash("No cleaned dataset available. Please preprocess a file first.", "error")
        return redirect(url_for("home"))

    buf = io.BytesIO(csv_data.encode("utf-8"))
    buf.seek(0)

    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="cleaned_dataset.csv"
    )


if __name__ == "__main__":
    app.run(debug=True)
