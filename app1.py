from flask import Flask, render_template, request, redirect, session
import os
import mysql.connector
from werkzeug.utils import secure_filename
from PIL import Image

# 🔥 NEW IMPORTS (Retinal Logic)
import cv2
import numpy as np

# ---------------- DB CONNECTION ----------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Kits@12345",
    database="heart_db"
)

cursor = db.cursor()

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = 'mysecret123'

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---------------- RETINAL FUNCTION ----------------
# -------- RETINAL LOGIC --------

def analyze_retina(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)

    if img is None:
        return 0, "Retina not readable"

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- FEATURES ---
    brightness = np.mean(gray)

    edges = cv2.Canny(gray, 50, 150)
    vessel_density = np.sum(edges > 0) / (256 * 256)

    red_intensity = np.mean(img[:, :, 2])

    # --- NORMAL RANGES ---
    normal_brightness = (80, 160)
    normal_vessel = (0.05, 0.25)
    normal_red = (120, 180)

    abnormal_score = 0

    if not (normal_brightness[0] <= brightness <= normal_brightness[1]):
        abnormal_score += 1

    if not (normal_vessel[0] <= vessel_density <= normal_vessel[1]):
        abnormal_score += 1

    if not (normal_red[0] <= red_intensity <= normal_red[1]):
        abnormal_score += 1

    # --- FINAL RESULT ---
    if abnormal_score >= 2:
        return 3, "Abnormal Retina"
    else:
        return 0, "Normal Retina"

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('login.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/first')
def first():
    cursor.execute("SELECT * FROM patient_data ORDER BY id DESC")
    data = cursor.fetchall()
    return render_template('index.html', data=data)


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    session['username'] = username
    return redirect('/index')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('index1.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/prevention')
def prevention():
    return render_template('prevention.html')


# ---------------- SAVE IMAGE ----------------
def save_img(img, filename):
    picture_path = os.path.join(app.root_path, 'static/images', filename)
    img = Image.open(img)
    img.save(picture_path)
    return 'images/' + filename


# ---------------- MAIN PREDICTION ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'GET':
        return redirect('/prediction')

    # -------- IMAGE PART --------
    file = request.files['file']
    filename = secure_filename(file.filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # -------- PARAMETERS --------
    age = int(request.form.get("age"))
    sbp = int(request.form.get("sbp"))
    dbp = int(request.form.get("dbp"))
    bmi = float(request.form.get("bmi"))
    hba1c = float(request.form.get("hba1c"))

    # -------- PARAMETER LOGIC --------
    risk_score = 0

    if age > 45:
        risk_score += 2
    elif age > 30:
        risk_score += 1

    if sbp > 150:
        risk_score += 2
    elif sbp > 130:
        risk_score += 1

    if dbp > 95:
        risk_score += 2
    elif dbp > 85:
        risk_score += 1

    if bmi > 30:
        risk_score += 2
    elif bmi > 25:
        risk_score += 1

    if hba1c > 6.5:
        risk_score += 2
    elif hba1c > 5.7:
        risk_score += 1

    # -------- PARAM RESULT --------
    if risk_score >= 7:
        param_result = "High Risk"
    elif risk_score >= 4:
        param_result = "Medium Risk"
    else:
        param_result = "Low Risk"

    # 🔥 -------- RETINAL LOGIC --------
    retinal_score, retinal_result = analyze_retina(file_path)

    # -------- COMBINE --------
    total_score = risk_score + retinal_score

    if total_score >= 8:
        final_result = "High Risk"
    elif total_score >= 4:
        final_result = "Medium Risk"
    else:
        final_result = "Low Risk"

    # -------- SAVE TO DB --------
    username = str(session.get('username'))

    cursor.execute(
        "INSERT INTO patient_data (username, age, systolic_bp, diastolic_bp, bmi, haemoglobin, image_path, retinal_result, param_result, final_result) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (username, age, sbp, dbp, bmi, hba1c, file_path, retinal_result, param_result, final_result)
    )

    db.commit()

    # -------- RETURN --------
    return render_template(
        'success.html',
        param_result=param_result,
        final_result=final_result,
        retinal_result=retinal_result
    )


@app.route('/success')
def success():
    return render_template('success.html')


@app.route('/prediction')
def prediction():
    cursor.execute("SELECT * FROM patient_data")
    data = cursor.fetchall()
    return render_template('predict.html', data=data)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)