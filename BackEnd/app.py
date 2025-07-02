# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__, template_folder="../FrontEnd", static_folder="../static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Init base SQLite
def init_db():
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS images') # Supprimer la table si elle existe déjà
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_date TEXT,
            annotation TEXT,
            width INTEGER,
            height INTEGER,
            filesize INTEGER,
            avg_color TEXT,
            contrast REAL,
            edges INTEGER,
            histogram TEXT,
            histogram_luminance TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance = extract_features(path)

            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("""INSERT INTO images 
                (filename, upload_date, width, height, filesize, avg_color, contrast, edges, histogram, histogram_luminance) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance))
            conn.commit()
            conn.close()

            return redirect(url_for('annotate', filename=filename))
    return render_template('upload.html')


@app.route('/annotate/<filename>', methods=['GET', 'POST'])
def annotate(filename):
    if request.method == 'POST':
        annotation = request.form['annotation']
        if annotation in ['pleine', 'vide']:
            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("UPDATE images SET annotation = ? WHERE filename = ?", (annotation, filename))
            conn.commit()
            conn.close()
        else:
            # Si l'annotation est "A determiner", on lance la labellisation automatique
            label_image(filename)
        return redirect(url_for('upload_image'))

    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM images WHERE filename = ?", (filename,))
    image_data = c.fetchone()
    conn.close()

    return render_template('annotate.html', filename=filename, image=image_data)

def extract_features(image_path):
    filesize_bytes = os.path.getsize(image_path)
    filesize_kb = round(filesize_bytes / 1024, 2)  # Convertir en Ko avec 2 décimales
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    img_array = np.array(img)
    avg_rgb = tuple(int(x) for x in np.mean(img_array.reshape(-1, 3), axis=0))

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    contrast = float(np.max(gray) - np.min(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_count = int(np.sum(edges > 0))

    # Histogrammes des couleurs RGB
    hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256]).flatten()
    hist_rgb_str = ','.join([f'{int(v)}' for v in np.concatenate([hist_r, hist_g, hist_b])])
    
    # Histogramme de luminance
    hist_luminance = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_luminance_str = ','.join([f'{int(v)}' for v in hist_luminance])

    return width, height, filesize_kb, str(avg_rgb), contrast, edge_count, hist_rgb_str, hist_luminance_str


def label_image(filename):
    """Cette fonction permet de labeliser l'image automatiquement.
    Si elle est "pleine" ou "vide".
    update la table images avec le label."""
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM images WHERE filename = ?", (filename,))
    image_data = c.fetchone()
    if image_data:
        # Logique de labellisation ici
        # Exemple simple : si la couleur moyenne est claire, on considère l'image comme "pleine", sinon "vide"
        avg_color = eval(image_data[7])  # Convertir la chaîne de caractères en
        label = 'pleine' if sum(avg_color) > 382 else 'vide'
        # Mettre à jour l'annotation dans la base de données
        c.execute("UPDATE images SET annotation = ? WHERE filename = ?", (label, filename))
        conn.commit()
    conn.close()


if __name__ == '__main__':
    app.run(debug=True)
