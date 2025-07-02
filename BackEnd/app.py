# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__, template_folder="../FrontEnd", static_folder="../static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Init base SQLite
def init_db():
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    #c.execute('DROP TABLE IF EXISTS images') # Supprimer la table si elle existe déjà
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

            # Classification automatique
            avg_rgb = eval(avg_color)  # Convertir la string en tuple
            auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance)

            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("""INSERT INTO images 
                (filename, upload_date, width, height, filesize, avg_color, contrast, edges, histogram, histogram_luminance, annotation) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, auto_classification))
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

@app.route('/stats')
def get_stats():
    """Route pour obtenir des statistiques sur les classifications"""
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    
    # Statistiques générales
    c.execute("SELECT COUNT(*) FROM images")
    total_images = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM images WHERE annotation = 'pleine'")
    pleines = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM images WHERE annotation = 'vide'")
    vides = c.fetchone()[0]
    
    # Moyennes des critères
    c.execute("SELECT AVG(contrast), AVG(edges) FROM images")
    avg_stats = c.fetchone()
    avg_contrast = avg_stats[0] if avg_stats[0] else 0
    avg_edges = avg_stats[1] if avg_stats[1] else 0
    
    conn.close()
    
    stats = {
        'total': total_images,
        'pleines': pleines,
        'vides': vides,
        'pourcentage_pleines': round((pleines / total_images * 100) if total_images > 0 else 0, 1),
        'avg_contrast': round(avg_contrast, 2),
        'avg_edges': round(avg_edges, 2)
    }
    
    return render_template('stats.html', stats=stats)

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

def classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, hist_luminance_str=None):
    """
    Algorithme de classification automatique amélioré pour déterminer si une poubelle est vide ou pleine
    
    Améliorations:
    - Correction de la logique de luminosité
    - Ajout de l'analyse de la distribution de luminance
    - Pondération des critères
    - Seuils adaptatifs selon la taille de l'image
    """
    
    # Paramètres de classification ajustables
    BRIGHTNESS_THRESHOLD = 135  # Seuil de luminosité moyenne (0-255)
    EDGE_DENSITY_BASE_THRESHOLD = 0.11  # Seuil de base pour la densité des contours
    CONTRAST_THRESHOLD = 245  # Seuil de contraste (abaissé pour plus de sensibilité)
    
    # Calculer la luminosité moyenne (moyenne pondérée RGB)
    r, g, b = avg_rgb
    # Utilisation de la formule de luminance perceptuelle
    avg_brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Calculer la densité des contours avec adaptation selon la taille
    total_pixels = width * height
    edge_density = edge_count / total_pixels if total_pixels > 0 else 0
    
    # Ajustement du seuil selon la résolution (images plus grandes = plus de détails naturels)
    resolution_factor = min(1.0, (width * height) / (640 * 480))  # Normalisation à 640x480
    adjusted_edge_threshold = EDGE_DENSITY_BASE_THRESHOLD * (1 + resolution_factor * 0.5)
    
    # Analyse de la distribution de luminance (si disponible)
    luminance_uniformity = 0.5  # Valeur par défaut
    if hist_luminance_str:
        hist_values = [float(x) for x in hist_luminance_str.split(',')]
        # Calculer l'entropie de la distribution (mesure de l'uniformité)
        total_pixels_hist = sum(hist_values)
        if total_pixels_hist > 0:
            probabilities = [x / total_pixels_hist for x in hist_values if x > 0]
            luminance_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            luminance_uniformity = luminance_entropy / 8.0  # Normalisation (max théorique = 8 bits)
    
    # Système de score pondéré
    criteria_scores = {}
    total_weight = 0
    weighted_score = 0
    
    # Critère 1: Luminosité (poids: 2.0) - CORRIGÉ: poubelle pleine = plus claire
    brightness_weight = 2.0
    if avg_brightness > BRIGHTNESS_THRESHOLD:  # Correction: < au lieu de >
        criteria_scores['brightness'] = 1.0
        weighted_score += brightness_weight
    else:
        criteria_scores['brightness'] = 0.0
    total_weight += brightness_weight
    
    # Critère 2: Densité des contours (poids: 2.5)
    edge_weight = 3.5
    if edge_density > adjusted_edge_threshold:
        criteria_scores['edges'] = 1.0
        weighted_score += edge_weight
    else:
        criteria_scores['edges'] = 0.0
    total_weight += edge_weight

    # Critère 3: Non-uniformité de la luminance (poids: 1.0)
    uniformity_weight = 1.0
    uniformity_threshold = 0.6
    if luminance_uniformity > uniformity_threshold:  # Plus de variation = plus plein
        criteria_scores['uniformity'] = 1.0
        weighted_score += uniformity_weight
    else:
        criteria_scores['uniformity'] = 0.0
    total_weight += uniformity_weight
    
    # Score final normalisé
    final_score = weighted_score / total_weight
    confidence_score = abs(final_score - 0.5) * 2  # Distance de 0.5, normalisée
    
    # Classification avec seuil ajustable
    classification_threshold = 0.45  # Seuil légèrement biaisé vers "vide"
    
    if final_score > classification_threshold:
        classification = "pleine"
        confidence = "haute" if confidence_score > 0.6 else "moyenne" if confidence_score > 0.3 else "faible"
    else:
        classification = "vide"
        confidence = "haute" if confidence_score > 0.6 else "moyenne" if confidence_score > 0.3 else "faible"
    
    # Informations détaillées de debug
    debug_info = {
        'avg_brightness': round(avg_brightness, 2),
        'brightness_threshold': BRIGHTNESS_THRESHOLD,
        'edge_density': round(edge_density, 4),
        'edge_threshold': round(adjusted_edge_threshold, 4),
        'contrast': round(contrast, 2),
        'contrast_threshold': CONTRAST_THRESHOLD,
        'luminance_uniformity': round(luminance_uniformity, 3),
        'final_score': round(final_score, 3),
        'confidence_score': round(confidence_score, 3),
        'criteria_scores': criteria_scores,
        'confidence': confidence,
        'resolution_factor': round(resolution_factor, 3)
    }
    
    return classification, debug_info

@app.route('/images')
def images():
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, annotation FROM images')
    images = cursor.fetchall()
    conn.close()
    return render_template('gallery.html', images=images)

@app.route('/image/<int:image_id>')
def image_detail(image_id):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images WHERE id = ?', (image_id,))
    image = cursor.fetchone()
    conn.close()
    if image:
        return render_template('detail.html', image=image)
    else:
        return "Image non trouvée", 404


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

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()

    # Récupère les données de base
    c.execute("SELECT COUNT(*) FROM images")
    total_images = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM images WHERE annotation = 'pleine'")
    full_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM images WHERE annotation = 'vide'")
    empty_count = c.fetchone()[0]

    c.execute("SELECT filesize FROM images")
    sizes = [row[0] / 1024 for row in c.fetchall()]  # en Ko

    c.execute("SELECT upload_date FROM images")
    # On ne garde que les dates valides (non nulles et non vides)
    dates = [row[0][:10] for row in c.fetchall() if row[0] and len(row[0]) >= 10]  # Juste la date (AAAA-MM-JJ)

    conn.close()

    # Pie chart annotation
    labels = ['Pleine', 'Vide']
    values = [full_count, empty_count]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    pie_buf = BytesIO()
    plt.savefig(pie_buf, format='png')
    pie_buf.seek(0)
    pie_png = base64.b64encode(pie_buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # Histogram taille des fichiers
    fig, ax = plt.subplots()
    ax.hist(sizes, bins=10, color='skyblue')
    ax.set_title('Distribution des tailles de fichiers (Ko)')
    ax.set_xlabel('Taille (Ko)')
    ax.set_ylabel('Fréquence')
    hist_buf = BytesIO()
    plt.savefig(hist_buf, format='png')
    hist_buf.seek(0)
    hist_png = base64.b64encode(hist_buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return render_template('dashboard.html',
                            total=total_images,
                           full=full_count,
                           empty=empty_count,
                           pie_chart=pie_png,
                           hist_chart=hist_png,
                           dates=dates)


@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    # Récupérer le nom du fichier pour supprimer le fichier physique
    c.execute("SELECT filename FROM images WHERE id = ?", (image_id,))
    row = c.fetchone()
    if row:
        filename = row[0]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(image_path):
            os.remove(image_path)
        c.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
    conn.close()
    return redirect(url_for('images'))


if __name__ == '__main__':
    app.run(debug=True)
