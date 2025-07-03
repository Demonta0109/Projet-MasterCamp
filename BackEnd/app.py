# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sqlite3
import hashlib
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
            histogram_luminance TEXT,
            bin_edges INTEGER,
            bin_area INTEGER,
            file_hash TEXT UNIQUE
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
        upload_type = request.form.get('upload_type', 'single')
        reanalyze = request.form.get('reanalyze') == 'true'
        processed_files = []
        duplicates = []
        reanalyzed = []
        errors = []
        
        if upload_type == 'single':
            # Mode single file (existant)
            file = request.files.get('image')
            if file and allowed_file(file.filename):
                result = process_single_file(file, reanalyze=reanalyze)
                if result['success']:
                    if result['type'] == 'reanalyzed':
                        reanalyzed.append(result)
                    else:
                        processed_files.append(result['filename'])
                else:
                    if result['type'] == 'duplicate':
                        duplicates.append(result)
                    else:
                        errors.append(result)
                
        elif upload_type == 'multiple':
            # Mode multiple files
            files = request.files.getlist('images')
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    result = process_single_file(file, reanalyze=reanalyze)
                    if result['success']:
                        if result['type'] == 'reanalyzed':
                            reanalyzed.append(result)
                        else:
                            processed_files.append(result['filename'])
                    else:
                        if result['type'] == 'duplicate':
                            duplicates.append(result)
                        else:
                            errors.append(result)
                    
        elif upload_type == 'folder':
            # Mode folder
            files = request.files.getlist('folder')
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    result = process_single_file(file, reanalyze=reanalyze)
                    if result['success']:
                        if result['type'] == 'reanalyzed':
                            reanalyzed.append(result)
                        else:
                            processed_files.append(result['filename'])
                    else:
                        if result['type'] == 'duplicate':
                            duplicates.append(result)
                        else:
                            errors.append(result)
        
        # Construire le message de résultat
        message_parts = []
        if processed_files:
            message_parts.append(f"{len(processed_files)} nouvelle(s) image(s) traitée(s)")
        if reanalyzed:
            message_parts.append(f"{len(reanalyzed)} image(s) ré-analysée(s)")
        if duplicates:
            message_parts.append(f"{len(duplicates)} doublon(s) ignoré(s)")
        if errors:
            message_parts.append(f"{len(errors)} erreur(s)")
        
        message = " • ".join(message_parts) if message_parts else "Aucun fichier traité"
        
        if processed_files or reanalyzed:
            if len(processed_files) == 1 and not reanalyzed and not duplicates and not errors:
                # Une seule nouvelle image traitée sans complications, rediriger vers la page d'annotation
                return redirect(url_for('annotate', filename=processed_files[0]))
            else:
                # Plusieurs images ou ré-analyses, rediriger vers la galerie avec un message détaillé
                return redirect(url_for('images', 
                                      message=message,
                                      duplicates=len(duplicates),
                                      reanalyzed=len(reanalyzed),
                                      errors=len(errors)))
        else:
            # Aucun fichier traité avec succès
            return redirect(url_for('upload_image', 
                                  error_message=message))
            
    # Afficher le message d'erreur s'il y en a un
    error_message = request.args.get('error_message')
    return render_template('upload.html', error_message=error_message)

def process_single_file(file, reanalyze=False):
    """
    Traite un seul fichier et l'insère en base de données
    
    Args:
        file: Le fichier à traiter
        reanalyze: Si True, ré-analyse les doublons au lieu de les ignorer
    
    Returns:
        dict: Résultat du traitement avec les clés :
        - success: bool
        - filename: str (si succès)
        - message: str
        - type: str ('success', 'duplicate', 'error', 'reanalyzed')
        - duplicate_info: dict (si doublon détecté)
    """
    filename = secure_filename(file.filename)
    
    # Créer un fichier temporaire pour vérifier les doublons
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
    file.save(temp_path)
    
    try:
        # Vérification des doublons
        is_duplicate, duplicate_info = is_duplicate_file(temp_path, filename)
        if is_duplicate and not reanalyze:
            # Si c'est un doublon et qu'on ne ré-analyse pas, supprimer le fichier temporaire
            os.remove(temp_path)
            return {
                'success': False,
                'message': duplicate_info['message'],
                'type': 'duplicate',
                'duplicate_info': duplicate_info,
                'original_filename': filename
            }
        elif is_duplicate and reanalyze:
            # Ré-analyser le doublon : utiliser le fichier existant et mettre à jour la base
            existing_filename = duplicate_info['filename']
            existing_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_filename)
            
            # Supprimer le fichier temporaire (on utilise l'existant)
            os.remove(temp_path)
            
            # Si le nouveau fichier a un contenu différent, remplacer l'ancien
            if duplicate_info['type'] == 'filename':  # Même nom mais contenu différent
                # Remplacer le fichier existant par le nouveau
                final_path = existing_path
                file.seek(0)  # Rembobiner le fichier
                file.save(final_path)
            else:
                # Même contenu, garder le fichier existant
                final_path = existing_path
            
            # Ré-analyser avec les nouvelles métriques
            width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, bin_edge_count, bin_area = extract_features(final_path)
            avg_rgb = eval(avg_color)
            auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance, bin_edge_count, bin_area)
            file_hash = calculate_file_hash(final_path)
            
            # Mettre à jour la base de données
            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("""UPDATE images SET 
                upload_date = ?, width = ?, height = ?, filesize = ?, avg_color = ?, 
                contrast = ?, edges = ?, histogram = ?, histogram_luminance = ?, 
                annotation = ?, bin_edges = ?, bin_area = ?, file_hash = ?
                WHERE id = ?""",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 width, height, filesize, avg_color, contrast, edge_count, histogram, 
                 histogram_luminance, auto_classification, bin_edge_count, bin_area, 
                 file_hash, duplicate_info['id']))
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'filename': existing_filename,
                'message': f"Image {existing_filename} ré-analysée avec succès",
                'type': 'reanalyzed',
                'original_id': duplicate_info['id']
            }
        
        # Si pas de doublon, traitement normal
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Éviter les conflits de noms dans le système de fichiers
        if os.path.exists(final_path):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{name}_{timestamp}{ext}"
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.rename(temp_path, final_path)
        
        # Extraction des caractéristiques
        width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, bin_edge_count, bin_area = extract_features(final_path)

        # Classification automatique
        avg_rgb = eval(avg_color)  # Convertir la string en tuple
        auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance, bin_edge_count, bin_area)

        # Calculer le hash pour la base de données
        file_hash = calculate_file_hash(final_path)

        # Insertion en base de données
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("""INSERT INTO images 
            (filename, upload_date, width, height, filesize, avg_color, contrast, edges, histogram, histogram_luminance, annotation, bin_edges, bin_area, file_hash) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, auto_classification, bin_edge_count, bin_area, file_hash))
        conn.commit()
        conn.close()

        return {
            'success': True,
            'filename': filename,
            'message': f"Image {filename} analysée avec succès",
            'type': 'success'
        }
        
    except Exception as e:
        # Nettoyer en cas d'erreur
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {
            'success': False,
            'message': f"Erreur lors du traitement de {filename}: {str(e)}",
            'type': 'error',
            'original_filename': filename
        }


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
            classification, debug_info = classify_bin_automatic(
                eval(request.form['avg_color']),  # Convertir la string en tuple
                int(request.form['edge_count']),
                float(request.form['contrast']),
                int(request.form['width']),
                int(request.form['height']),
                request.form['hist_luminance'],
                int(request.form.get('bin_edge_count', 0)),  # Nouveau paramètre
                int(request.form.get('bin_area', 1))  # Nouveau paramètre avec valeur par défaut pour éviter division par zéro
            )
            # Mettre à jour l'annotation dans la base de données
            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("UPDATE images SET annotation = ? WHERE filename = ?", (classification, filename))
            conn.commit()
            conn.close()
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

    # Détection des contours sur l'image entière
    edges = cv2.Canny(gray, 100, 200)
    edge_count = int(np.sum(edges > 0))
    
    # Détection de la région de la benne et calcul des contours dans cette région
    bin_region_edges, bin_edge_count, bin_region_area = detect_bin_region_and_edges(img_array, gray)

    # Histogrammes des couleurs RGB
    hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256]).flatten()
    hist_rgb_str = ','.join([f'{int(v)}' for v in np.concatenate([hist_r, hist_g, hist_b])])
    
    # Histogramme de luminance
    hist_luminance = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_luminance_str = ','.join([f'{int(v)}' for v in hist_luminance])

    return width, height, filesize_kb, str(avg_rgb), contrast, edge_count, hist_rgb_str, hist_luminance_str, bin_edge_count, bin_region_area

def detect_bin_region_and_edges(img_array, gray):
    """
    Détecte la région de la benne dans l'image et calcule les contours dans cette région uniquement
    
    Stratégies de détection :
    1. Détection de formes rectangulaires/cylindriques (bennes typiques)
    2. Segmentation par couleur (bennes souvent sombres/métalliques)
    3. Détection de contours forts (bords de la benne)
    4. Zone centrale de l'image (bennes souvent centrées)
    """
    
    height, width = gray.shape
    
    # Stratégie 1: Détection de contours pour trouver les formes principales
    edges_strong = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges_strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trouver le plus grand contour (probablement la benne)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Si le contour est assez grand (au moins 10% de l'image), l'utiliser
        if area > (width * height * 0.1):
            # Créer un masque pour la région de la benne
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Calculer les contours fins uniquement dans cette région
            edges_fine = cv2.Canny(gray, 100, 200)
            bin_region_edges = cv2.bitwise_and(edges_fine, mask)
            bin_edge_count = int(np.sum(bin_region_edges > 0))
            bin_region_area = int(area)
            
            return bin_region_edges, bin_edge_count, bin_region_area
    
    # Stratégie 2: Si pas de contour détecté, utiliser la région centrale
    # (souvent les bennes sont au centre de l'image)
    center_margin = 0.15  # 15% de marge de chaque côté
    x_start = int(width * center_margin)
    x_end = int(width * (1 - center_margin))
    y_start = int(height * center_margin)
    y_end = int(height * (1 - center_margin))
    
    # Créer un masque pour la région centrale
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y_start:y_end, x_start:x_end] = 255
    
    # Calculer les contours dans la région centrale
    edges_fine = cv2.Canny(gray, 100, 200)
    bin_region_edges = cv2.bitwise_and(edges_fine, mask)
    bin_edge_count = int(np.sum(bin_region_edges > 0))
    bin_region_area = int((x_end - x_start) * (y_end - y_start))
    
    return bin_region_edges, bin_edge_count, bin_region_area

def classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, hist_luminance_str=None, bin_edge_count=None, bin_area=None):
    """
    Algorithme de classification automatique amélioré pour déterminer si une poubelle est vide ou pleine
    
    Améliorations:
    - Correction de la logique de luminosité
    - Ajout de l'analyse de la distribution de luminance
    - Utilisation des contours spécifiques à la région de la benne
    - Pondération des critères
    - Seuils adaptatifs selon la taille de l'image
    """
    
    # Paramètres de classification ajustables
    BRIGHTNESS_THRESHOLD = 135  # Seuil de luminosité moyenne (0-255)
    EDGE_DENSITY_BASE_THRESHOLD = 0.11  # Seuil de base pour la densité des contours
    BIN_EDGE_DENSITY_THRESHOLD = 0.15   # Seuil pour les contours dans la benne (plus élevé)
    CONTRAST_THRESHOLD = 245  # Seuil de contraste (abaissé pour plus de sensibilité)
    
    # Calculer la luminosité moyenne (moyenne pondérée RGB)
    r, g, b = avg_rgb
    # Utilisation de la formule de luminance perceptuelle
    avg_brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Calculer la densité des contours
    total_pixels = width * height
    edge_density = edge_count / total_pixels if total_pixels > 0 else 0
    
    # Calculer la densité des contours dans la benne (prioritaire si disponible)
    bin_edge_density = 0
    if bin_edge_count is not None and bin_area is not None and bin_area > 0:
        bin_edge_density = bin_edge_count / bin_area
    
    # Ajustement du seuil selon la résolution (images plus grandes = plus de détails naturels)
    # Ajustement plus doux du seuil selon la résolution (moins agressif)
    resolution_factor = min(1.0, (width * height) / (640 * 480))  # Normalisation à 640x480
    adjusted_edge_threshold = EDGE_DENSITY_BASE_THRESHOLD * (1 + resolution_factor * 0.2)
    adjusted_bin_edge_threshold = BIN_EDGE_DENSITY_THRESHOLD * (1 + resolution_factor * 0.1)
    
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
    
    # Critère 2: Densité des contours dans la benne (prioritaire) ou globale (poids: 4.0)
    edge_weight = 4.0
    if bin_edge_count is not None and bin_area is not None:
        # Utiliser la densité de contours dans la benne (plus précis)
        if bin_edge_density > adjusted_bin_edge_threshold:
            criteria_scores['bin_edges'] = 1.0
            weighted_score += edge_weight
        else:
            criteria_scores['bin_edges'] = 0.0
        criteria_scores['edges'] = 'N/A (utilise bin_edges)'
    else:
        # Fallback sur la densité globale
        if edge_density > adjusted_edge_threshold:
            criteria_scores['edges'] = 1.0
            weighted_score += edge_weight
        else:
            criteria_scores['edges'] = 0.0
        criteria_scores['bin_edges'] = 'N/A (utilise edges)'
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
        'bin_edge_density': round(bin_edge_density, 4) if bin_edge_count is not None else 'N/A',
        'bin_edge_threshold': round(adjusted_bin_edge_threshold, 4) if bin_edge_count is not None else 'N/A',
        'bin_edge_count': bin_edge_count if bin_edge_count is not None else 'N/A',
        'bin_area': bin_area if bin_area is not None else 'N/A',
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
    message = request.args.get('message')
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, annotation FROM images ORDER BY upload_date DESC')
    images = cursor.fetchall()
    conn.close()
    return render_template('gallery.html', images=images, message=message)

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

@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    """Route AJAX pour le traitement des uploads multiples avec progression"""
    upload_type = request.form.get('upload_type', 'single')
    reanalyze = request.form.get('reanalyze') == 'true'
    processed_files = []
    duplicates = []
    reanalyzed = []
    errors = []
    total_files = 0
    
    try:
        if upload_type == 'single':
            files = [request.files.get('image')]
        elif upload_type == 'multiple':
            files = request.files.getlist('images')
        elif upload_type == 'folder':
            files = request.files.getlist('folder')
        else:
            return jsonify({'error': 'Type d\'upload invalide'}), 400
        
        # Filtrer les fichiers valides
        valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]
        total_files = len(valid_files)
        
        if total_files == 0:
            return jsonify({'error': 'Aucun fichier valide trouvé'}), 400
        
        for i, file in enumerate(valid_files):
            try:
                result = process_single_file(file, reanalyze=reanalyze)
                
                if result['success']:
                    if result['type'] == 'reanalyzed':
                        reanalyzed.append(result)
                    else:
                        processed_files.append(result['filename'])
                else:
                    if result['type'] == 'duplicate':
                        duplicates.append(result)
                    else:
                        errors.append(result)
                
                # Progression
                progress = ((i + 1) / total_files) * 100
                
            except Exception as e:
                error_result = {
                    'success': False,
                    'message': f"Erreur lors du traitement de {file.filename}: {str(e)}",
                    'type': 'error',
                    'original_filename': file.filename
                }
                errors.append(error_result)
                continue
        
        return jsonify({
            'success': True,
            'processed_count': len(processed_files),
            'reanalyzed_count': len(reanalyzed),
            'duplicate_count': len(duplicates),
            'error_count': len(errors),
            'total_count': total_files,
            'files': processed_files,
            'reanalyzed': [r['message'] for r in reanalyzed],
            'duplicates': [d['message'] for d in duplicates],
            'errors': [e['message'] for e in errors]
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement: {str(e)}'}), 500
        

def calculate_file_hash(file_path):
    """Calcule le hash SHA-256 d'un fichier"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def is_duplicate_file(file_path, filename):
    """
    Vérifie si un fichier est un doublon basé sur le hash et/ou le nom
    
    Returns:
        tuple: (is_duplicate, duplicate_info)
        - is_duplicate: bool, True si c'est un doublon
        - duplicate_info: dict avec les infos du doublon ou None
    """
    file_hash = calculate_file_hash(file_path)
    
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    
    # Vérifier d'abord par hash (plus fiable)
    c.execute("SELECT id, filename, upload_date, annotation FROM images WHERE file_hash = ?", (file_hash,))
    hash_duplicate = c.fetchone()
    
    if hash_duplicate:
        conn.close()
        return True, {
            'type': 'hash',
            'id': hash_duplicate[0],
            'filename': hash_duplicate[1],
            'upload_date': hash_duplicate[2],
            'annotation': hash_duplicate[3],
            'message': f"Fichier identique déjà analysé (même contenu): {hash_duplicate[1]}"
        }
    
    # Vérifier ensuite par nom de fichier
    c.execute("SELECT id, filename, upload_date, annotation FROM images WHERE filename = ?", (filename,))
    name_duplicate = c.fetchone()
    
    conn.close()
    
    if name_duplicate:
        return True, {
            'type': 'filename',
            'id': name_duplicate[0],
            'filename': name_duplicate[1],
            'upload_date': name_duplicate[2],
            'annotation': name_duplicate[3],
            'message': f"Fichier avec le même nom déjà analysé: {name_duplicate[1]}"
        }
    
    return False, None

@app.route('/reanalyze/<int:image_id>', methods=['POST'])
def reanalyze_image(image_id):
    """Route pour ré-analyser une image spécifique depuis la galerie"""
    try:
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("SELECT filename FROM images WHERE id = ?", (image_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return redirect(url_for('images', message="Image non trouvée"))
        
        filename = row[0]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(image_path):
            return redirect(url_for('images', message=f"Fichier {filename} non trouvé sur le disque"))
        
        # Ré-analyser l'image
        width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, bin_edge_count, bin_area = extract_features(image_path)
        avg_rgb = eval(avg_color)
        auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance, bin_edge_count, bin_area)
        file_hash = calculate_file_hash(image_path)
        
        # Mettre à jour la base de données
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("""UPDATE images SET 
            upload_date = ?, width = ?, height = ?, filesize = ?, avg_color = ?, 
            contrast = ?, edges = ?, histogram = ?, histogram_luminance = ?, 
            annotation = ?, bin_edges = ?, bin_area = ?, file_hash = ?
            WHERE id = ?""",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             width, height, filesize, avg_color, contrast, edge_count, histogram, 
             histogram_luminance, auto_classification, bin_edge_count, bin_area, 
             file_hash, image_id))
        conn.commit()
        conn.close()
        
        return redirect(url_for('images', message=f"Image {filename} ré-analysée avec succès"))
        
    except Exception as e:
        return redirect(url_for('images', message=f"Erreur lors de la ré-analyse: {str(e)}"))

if __name__ == '__main__':
    app.run(debug=True)
