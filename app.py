from flask import Flask, render_template, redirect, request, flash
import subprocess
import csv
import os
import threading
import sys

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"

training_running = False  # prevent duplicate train runs

def get_stats():
    """Return dashboard stats: total_students and total_records."""
    total_students = 0
    roster_csv = os.path.join(os.path.dirname(__file__), "students.csv")
    if os.path.exists(roster_csv):
        with open(roster_csv, "r", encoding="utf-8") as f:
            total_students = sum(1 for _ in csv.DictReader(f))

    total_records = 0
    att_csv = os.path.join(os.path.dirname(__file__), "attendance.csv")
    if os.path.exists(att_csv):
        with open(att_csv, "r", encoding="utf-8") as f:
            total_records = sum(1 for _ in f)

    return total_students, total_records

@app.route("/")
def home():
    total_students, total_records = get_stats()
    return render_template("index.html",
                           total_students=total_students,
                           total_records=total_records)

@app.route("/capture", methods=["GET","POST"])
def capture():
    # show a simple form to enter the name on GET
    if request.method == "GET":
        return render_template("capture_form.html")

    # on POST start capture in background
    name = request.form.get("name", "").strip()
    if not name:
        flash("Name is required to capture dataset.")
        return redirect("/capture")

    def run_capture():
        subprocess.run([sys.executable, "capture_dataset.py", "--name", name])
    threading.Thread(target=run_capture, daemon=True).start()
    flash(f"Started dataset capture for {name}. A camera window will appear.")
    return redirect("/")

@app.route("/train", methods=["POST"])
def train():
    global training_running
    if training_running:
        flash("⚠️ Training is already running. Please wait for it to finish.")
        return redirect("/")
    def run_training():
        global training_running
        training_running = True
        subprocess.run([sys.executable, "train_model.py"])
        training_running = False
    threading.Thread(target=run_training, daemon=True).start()
    flash("🧠 Training started. It will finish in the background.")
    return redirect("/")

@app.route("/recognize")
def recognize():
    def run_recog():
        subprocess.run([sys.executable, "recognize_attendance.py"])
    threading.Thread(target=run_recog, daemon=True).start()
    flash("Recognition started; you will be redirected to attendance when finished.")
    return redirect("/attendance")

@app.route("/session")
def session():
    def run_sess():
        subprocess.run([sys.executable, "run_session.py"])
    threading.Thread(target=run_sess, daemon=True).start()
    flash("Full session started in background.")
    return redirect("/attendance")

@app.route("/attendance")
def attendance():
    records = []
    att_csv = os.path.join(os.path.dirname(__file__), "attendance.csv")
    if os.path.exists(att_csv):
        with open(att_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            records = list(reader)
    # Filter valid rows (at least 2 columns, skip header-like rows)
    valid_records = [r for r in records if len(r) >= 2]
    return render_template("attendance.html", records=valid_records)

if __name__ == "__main__":
    app.run(debug=True)