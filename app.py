from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
from model import train_and_predict
from flask import Flask, render_template, request, redirect, url_for, session
from auth import init_db, register_user, validate_user
import sqlite3
from datetime import datetime


app = Flask(__name__)
init_db()
app.secret_key = "supersecret"  # secure this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

latest_predictions = []  # Store for download

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['user'])

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if register_user(request.form['username'], request.form['password']):
            return redirect(url_for('login'))
        return "Username already exists."
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if validate_user(request.form['username'], request.form['password']):
            session['user'] = request.form['username']
            return redirect(url_for('home'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, filename, uploaded_at FROM uploads WHERE username = ? ORDER BY uploaded_at DESC", (session['user'],))
    uploads = c.fetchall()
    conn.close()
    return render_template('history.html', uploads=uploads)
@app.route('/view/<int:upload_id>')
def view_forecast(upload_id):
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM uploads WHERE id = ? AND username = ?", (upload_id, session['user']))
    row = c.fetchone()
    conn.close()

    if row:
        filename = row[0]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            preview_df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore').head(5)
            predictions, historical_data, rmse, mape = train_and_predict(filepath)

            return render_template(
                'result.html',
                predictions=predictions,
                preview_data=preview_df.to_html(classes='table table-striped', index=False),
                historical_data=historical_data,
                rmse=rmse,
                mape=mape
            )
        except Exception as e:
            return f"<h3 style='color:red;'>Error loading file: {str(e)}</h3>"

    return "Upload not found"
@app.route('/download/<filename>')
def download_file(filename):
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict():
    global latest_predictions
    file = request.files['file']
    forecast_days = int(request.form.get('forecast_days', 10))  # get from user input

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            preview_df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore').head(5)
            predictions, historical_data, rmse, mape = train_and_predict(filepath, forecast_days)
            latest_predictions = predictions

            # ✅ Save upload info to DB
            if 'user' in session:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("INSERT INTO uploads (username, filename, uploaded_at) VALUES (?, ?, ?)",
                          (session['user'], filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()

            return render_template(
                'result.html',
                predictions=predictions,
                preview_data=preview_df.to_html(classes='table table-striped', index=False),
                historical_data=historical_data,
                rmse=rmse,
                mape=mape,
                forecast_days=forecast_days
            )
        except Exception as e:
            return f"<h3 style='color:red;'>Error: {str(e)}</h3>"

    return "No file uploaded"


@app.route('/download')
def download():
    global latest_predictions
    if not latest_predictions:
        return "No predictions to download."
    df = pd.DataFrame(latest_predictions, columns=['Date', 'Predicted Sales'])
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'forecast.csv')
    df.to_csv(output_path, index=False)
    return send_file(output_path, as_attachment=True)

@app.route('/sample')
def sample_csv():
    sample = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "Sales": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    })
    sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample.csv')
    sample.to_csv(sample_path, index=False)
    return send_file(sample_path, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
