# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask import g
import os
import sqlite3
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif AVANT d'importer pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__, template_folder="../FrontEnd", static_folder="../static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # Ajoute une clé secrète pour la session

@app.before_request
def set_language():
    """
    Fonction exécutée avant chaque requête pour définir la langue de session.
    """
    g.lang = session.get('lang', 'fr')

@app.route('/set_language/<lang>')
def set_language_route(lang):
    """
    Route pour changer la langue de l'interface utilisateur.
    """
    if lang in ['fr', 'en']:
        session['lang'] = lang
    return redirect(request.referrer or url_for('upload_image'))

# Init base SQLite
def init_db():
    """
    Initialise la base de données SQLite pour stocker les informations des images.

    Args : 
        - Aucun

    Return : 
        - Aucun Mais crée ou réinitialise la base de données 'db.sqlite' avec une table 'images'."""
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
            histogram_luminance TEXT,
            bin_edges INTEGER,
            bin_area INTEGER,
            patch_diversity REAL,
            file_hash TEXT UNIQUE,
            latitude REAL,
            longitude REAL
        )
    ''')
    # Ajout d'index pour accélérer les requêtes de filtrage/recherche
    c.execute('CREATE INDEX IF NOT EXISTS idx_annotation ON images(annotation)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_filename ON images(filename)')
    conn.commit()
    conn.close()


init_db()

def allowed_file(filename):
    """
    Vérifie si un fichier a une extension autorisée pour l'upload.
    
    Args:
        filename (str): Le nom du fichier à vérifier
    
    Returns:
        bool: True si l'extension est autorisée, False sinon
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    """
    Route principale pour l'upload d'images. Gère les uploads simples, multiples et par dossier.
    
    Args:
        Aucun (utilise request.method, request.form, request.files)
    
    Returns:
        flask.Response: Redirection vers validate_location, images, ou render du template upload.html
    """
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
        
        # Construire le message de résultat multilingue
        message_parts = []
        if processed_files:
            message_parts.append(tr(
                f"{len(processed_files)} nouvelle(s) image(s) traitée(s)",
                f"{len(processed_files)} new image(s) processed"
            ))
        if reanalyzed:
            message_parts.append(tr(
                f"{len(reanalyzed)} image(s) ré-analysée(s)",
                f"{len(reanalyzed)} image(s) re-analyzed"
            ))
        if duplicates:
            message_parts.append(tr(
                f"{len(duplicates)} doublon(s) ignoré(s)",
                f"{len(duplicates)} duplicate(s) ignored"
            ))
        if errors:
            message_parts.append(tr(
                f"{len(errors)} erreur(s)",
                f"{len(errors)} error(s)"
            ))
        
        message = " • ".join(message_parts) if message_parts else tr("Aucun fichier traité", "No file processed")
        
        if processed_files or reanalyzed:
            if len(processed_files) == 1 and not reanalyzed and not duplicates and not errors:
                # Une seule nouvelle image traitée sans complications, rediriger vers la page de validation de localisation
                return redirect(url_for('validate_location', filename=processed_files[0]))
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
                'message': tr(
                    duplicate_info['message'],
                    duplicate_info['message'].replace("Fichier", "File").replace("déjà analysé", "already analyzed").replace("identique", "identical").replace("même contenu", "same content").replace("avec le même nom", "with the same name")
                ),
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
            width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, bin_edge_count, bin_area, patch_diversity = extract_features(final_path)
            avg_rgb = eval(avg_color)
            auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance, bin_edge_count, bin_area, patch_diversity)
            file_hash = calculate_file_hash(final_path)
            # Mettre à jour la base de données
            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("""UPDATE images SET \
                upload_date = ?, width = ?, height = ?, filesize = ?, avg_color = ?, \
                contrast = ?, edges = ?, histogram = ?, histogram_luminance = ?, \
                annotation = ?, bin_edges = ?, bin_area = ?, patch_diversity = ?, file_hash = ?\
                WHERE id = ?""",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 width, height, filesize, avg_color, contrast, edge_count, histogram, 
                 histogram_luminance, auto_classification, bin_edge_count, bin_area, 
                 patch_diversity, file_hash, duplicate_info['id']))
            conn.commit()
            conn.close()
            return {
                'success': True,
                'filename': existing_filename,
                'message': tr(f"Image {existing_filename} ré-analysée avec succès", f"Image {existing_filename} successfully re-analyzed"),
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
        width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, bin_edge_count, bin_area, patch_diversity = extract_features(final_path)
        # Classification automatique
        avg_rgb = eval(avg_color)  # Convertir la string en tuple
        auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance, bin_edge_count, bin_area, patch_diversity)
        # Calculer le hash pour la base de données
        file_hash = calculate_file_hash(final_path)
        # Insertion en base de données
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("""INSERT INTO images \
            (filename, upload_date, width, height, filesize, avg_color, contrast, edges, histogram, histogram_luminance, annotation, bin_edges, bin_area, patch_diversity, file_hash, latitude, longitude) \
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
            (filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, auto_classification, bin_edge_count, bin_area, patch_diversity, file_hash))
        conn.commit()
        conn.close()
        return {
            'success': True,
            'filename': filename,
            'message': tr(f"Image {filename} analysée avec succès", f"Image {filename} successfully analyzed"),
            'type': 'success'
        }
    except Exception as e:
        # Nettoyer en cas d'erreur
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {
            'success': False,
            'message': tr(f"Erreur lors du traitement de {filename}: {str(e)}", f"Error processing {filename}: {str(e)}"),
            'type': 'error',
            'original_filename': filename
        }

@app.route('/validate_location/<filename>', methods=['GET', 'POST'])
def validate_location(filename):
    """
    Route pour valider et enregistrer la localisation GPS d'une image.
    
    Args:
        filename (str): Le nom du fichier image
    
    Returns:
        flask.Response: Redirection vers annotate ou render du template validate_location.html
    """
    if request.method == 'POST':
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("UPDATE images SET latitude = ?, longitude = ? WHERE filename = ?", (latitude, longitude, filename))
        conn.commit()
        conn.close()
        return redirect(url_for('annotate', filename=filename))
    return render_template('validate_location.html', filename=filename)

@app.route('/annotate/<filename>', methods=['GET', 'POST'])
def annotate(filename):
    """
    Route pour annoter une image (pleine, vide, ou classification automatique).
    
    Args:
        filename (str): Le nom du fichier image à annoter
    
    Returns:
        flask.Response: Redirection vers upload_image ou render du template annotate.html
    """
    if request.method == 'POST':
        annotation = request.form['annotation']
        if annotation in ['pleine', 'vide']:
            conn = sqlite3.connect('db.sqlite')
            c = conn.cursor()
            c.execute("UPDATE images SET annotation = ? WHERE filename = ?", (annotation, filename))
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
    """
    Extrait les caractéristiques visuelles d'une image pour la classification.
    
    Args:
        image_path (str): Chemin vers le fichier image
    
    Returns:
        tuple: (width, height, filesize_kb, avg_color_str, contrast, edge_count, 
                hist_rgb_str, hist_luminance_str, bin_edge_count, bin_region_area, patch_diversity)
    """
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

    # Analyse de patchs et diversité de teintes
    patch_diversity = analyze_patch_color_diversity(img_array)

    # Histogrammes des couleurs RGB
    hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256]).flatten()
    hist_rgb_str = ','.join([f'{int(v)}' for v in np.concatenate([hist_r, hist_g, hist_b])])
    
    # Histogramme de luminance
    hist_luminance = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_luminance_str = ','.join([f'{int(v)}' for v in hist_luminance])

    return width, height, filesize_kb, str(avg_rgb), contrast, edge_count, hist_rgb_str, hist_luminance_str, bin_edge_count, bin_region_area, patch_diversity

def analyze_patch_color_diversity(img_array):
    """
    Analyse la diversité de couleurs dans différents patchs de l'image
    UNIQUEMENT dans les mêmes zones que celles utilisées par detect_bin_region_and_edges
    
    Une poubelle pleine aura généralement plus de diversité de couleurs (déchets variés)
    qu'une poubelle vide (couleur uniforme du fond/intérieur de la poubelle)
    
    Args:
        img_array: Image numpy array en RGB
    
    Returns:
        float: Score de diversité de couleurs (0-1, plus élevé = plus diverse)
    """
    height, width, _ = img_array.shape
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Utiliser la même logique de détection de région que detect_bin_region_and_edges
    # Définir la région d'intérêt : partie basse de l'image (éviter les arbres/ciel)
    roi_top = int(height * 0.55)
    roi_bottom = height
    roi_left = 0
    roi_right = width
    
    # Extraire la région d'intérêt pour l'analyse
    gray_roi = gray[roi_top:roi_bottom, roi_left:roi_right]
    img_roi = img_array[roi_top:roi_bottom, roi_left:roi_right]
    
    # Stratégie 1: Essayer de détecter la région de la benne comme dans detect_bin_region_and_edges
    edges_strong = cv2.Canny(gray_roi, 50, 150)
    contours, _ = cv2.findContours(edges_strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    target_region = None
    
    # Ajuster les coordonnées des contours pour correspondre à l'image complète
    if contours:
        adjusted_contours = []
        for contour in contours:
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 1] += roi_top  # Ajouter l'offset vertical
            adjusted_contours.append(adjusted_contour)
        
        if adjusted_contours:
            largest_contour = max(adjusted_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Si le contour est assez grand (au moins 8% de la région d'intérêt), l'utiliser
            roi_area = (roi_bottom - roi_top) * (roi_right - roi_left)
            if area > (roi_area * 0.08):
                # Créer un masque pour la région de la benne
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                # Extraire la région masquée pour l'analyse de diversité
                target_region = extract_masked_region(img_array, mask)
    
    # Stratégie 2: Si pas de contour détecté, utiliser la région centrale-basse
    if target_region is None:
        center_margin_x = 0.12
        bottom_margin_y = 0.07
        top_margin_y = 0.55
        
        x_start = int(width * center_margin_x)
        x_end = int(width * (1 - center_margin_x))
        y_start = int(height * top_margin_y)
        y_end = int(height * (1 - bottom_margin_y))
        
        target_region = img_array[y_start:y_end, x_start:x_end]
    
    # Analyser la diversité de couleurs dans la région ciblée
    return analyze_region_patch_diversity(target_region)

def extract_masked_region(img_array, mask):
    """
    Extrait la région de l'image correspondant au masque et la convertit en rectangle
    pour l'analyse de patchs.
    
    Args:
        img_array (numpy.ndarray): Image en format numpy array RGB
        mask (numpy.ndarray): Masque binaire de la région à extraire
    
    Returns:
        numpy.ndarray or None: Région rectangulaire extraite ou None si le masque est vide
    """
    # Trouver les coordonnées de la bounding box du masque
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extraire la région rectangulaire englobante
    return img_array[y_min:y_max+1, x_min:x_max+1]

def analyze_region_patch_diversity(region_array):
    """
    Analyse la diversité de couleurs par patchs dans une région spécifique
    
    Args:
        region_array: Région d'image numpy array en RGB
    
    Returns:
        float: Score de diversité de couleurs (0-1, plus élevé = plus diverse)
    """
    if region_array is None or region_array.size == 0:
        return 0.0
    
    height, width = region_array.shape[:2]
    
    # Adapter la taille des patchs à la région
    patch_size = min(32, width // 3, height // 3)  # Patchs plus petits pour les régions focalisées
    if patch_size < 8:  # Région trop petite
        # Analyser la région entière comme un seul "patch"
        return calculate_patch_color_diversity(region_array)
    
    diversities = []
    
    # Parcourir la région par patchs avec chevauchement
    step_size = max(patch_size // 2, 4)  # Au moins 4 pixels de step
    
    for y in range(0, height - patch_size + 1, step_size):
        for x in range(0, width - patch_size + 1, step_size):
            # Extraire le patch
            patch = region_array[y:y+patch_size, x:x+patch_size]
            
            # Calculer la diversité de couleurs dans ce patch
            patch_diversity = calculate_patch_color_diversity(patch)
            diversities.append(patch_diversity)
    
    # Retourner la diversité moyenne de tous les patchs
    return np.mean(diversities) if diversities else 0.0

def calculate_patch_color_diversity(patch):
    """
    Calcule la diversité de couleurs dans un patch donné
    
    Args:
        patch: Patch d'image numpy array en RGB
    
    Returns:
        float: Score de diversité (0-1)
    """
    # Méthode 1: Variance des couleurs HSV (teinte, saturation, valeur)
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    
    # Calculer la variance de la teinte (Hue) - indicateur clé de diversité
    hue_variance = np.var(patch_hsv[:, :, 0])
    
    # Calculer la variance de la saturation
    saturation_variance = np.var(patch_hsv[:, :, 1])
    
    # Calculer la variance de la valeur (luminosité)
    value_variance = np.var(patch_hsv[:, :, 2])
    
    # Normaliser les variances (les valeurs HSV vont de 0 à 179 pour H, 0 à 255 pour S et V)
    hue_diversity = min(hue_variance / (90.0 ** 2), 1.0)  # Normalisation pour teinte
    saturation_diversity = min(saturation_variance / (128.0 ** 2), 1.0)  # Normalisation pour saturation
    value_diversity = min(value_variance / (128.0 ** 2), 1.0)  # Normalisation pour valeur
    
    # Méthode 2: Nombre de couleurs distinctes (quantifiées)
    # Quantifier les couleurs pour réduire le bruit
    patch_quantized = patch // 32 * 32  # Réduire à 8 niveaux par canal
    unique_colors = len(np.unique(patch_quantized.reshape(-1, 3), axis=0))
    max_possible_colors = min(64, patch.shape[0] * patch.shape[1])  # Maximum théorique raisonnable
    color_count_diversity = unique_colors / max_possible_colors
    
    # Combiner les métriques avec pondération
    # La teinte est le plus important pour la diversité visuelle
    diversity_score = (
        0.4 * hue_diversity +           # Teinte (plus important)
        0.25 * saturation_diversity +   # Saturation
        0.15 * value_diversity +        # Luminosité
        0.2 * color_count_diversity     # Nombre de couleurs distinctes
    )
    
    return min(diversity_score, 1.0)

def detect_bin_region_and_edges(img_array, gray):
    """
    Détecte la région de la benne dans l'image et calcule les contours dans cette région uniquement
    
    Stratégies de détection :
    1. Détection de formes rectangulaires/cylindriques (bennes typiques) dans la partie basse
    2. Segmentation par couleur (bennes souvent sombres/métalliques) dans la partie basse
    3. Détection de contours forts (bords de la benne) dans la partie basse
    4. Zone centrale-basse de l'image (bennes souvent centrées et au sol)
    """
    
    height, width = gray.shape
    
    # Définir la région d'intérêt : partie basse de l'image (éviter les arbres/ciel)
    # On se concentre sur les 60% inférieurs de l'image où se trouvent généralement les bennes
    roi_top = int(height * 0.55)  # Commencer à 25% de la hauteur
    roi_bottom = height
    roi_left = 0
    roi_right = width
    
    # Extraire la région d'intérêt pour l'analyse
    gray_roi = gray[roi_top:roi_bottom, roi_left:roi_right]
    
    # Stratégie 1: Détection de contours dans la région d'intérêt uniquement
    edges_strong = cv2.Canny(gray_roi, 50, 150)
    contours, _ = cv2.findContours(edges_strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ajuster les coordonnées des contours pour correspondre à l'image complète
    if contours:
        # Ajuster les coordonnées des contours
        adjusted_contours = []
        for contour in contours:
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 1] += roi_top  # Ajouter l'offset vertical
            adjusted_contours.append(adjusted_contour)
        
        if adjusted_contours:
            largest_contour = max(adjusted_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Si le contour est assez grand (au moins 8% de la région d'intérêt), l'utiliser
            roi_area = (roi_bottom - roi_top) * (roi_right - roi_left)
            if area > (roi_area * 0.08):
                # Créer un masque pour la région de la benne
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                # Calculer les contours fins uniquement dans cette région
                edges_fine = cv2.Canny(gray, 100, 200)
                bin_region_edges = cv2.bitwise_and(edges_fine, mask)
                bin_edge_count = int(np.sum(bin_region_edges > 0))
                bin_region_area = int(area)
                
                return bin_region_edges, bin_edge_count, bin_region_area
    
    # Stratégie 2: Si pas de contour détecté, utiliser la région centrale-basse
    # (souvent les bennes sont au centre horizontalement et dans la partie basse verticalement)
    center_margin_x = 0.12  # 10% de marge de chaque côté horizontalement
    bottom_margin_y = 0.07  # 5% de marge depuis le bas
    top_margin_y = 0.55     # Commencer à 25% de la hauteur (partie basse)
    
    x_start = int(width * center_margin_x)
    x_end = int(width * (1 - center_margin_x))
    y_start = int(height * top_margin_y)    # Partie basse seulement
    y_end = int(height * (1 - bottom_margin_y))  # Laisser une marge en bas
    
    # Créer un masque pour la région centrale-basse
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y_start:y_end, x_start:x_end] = 255
    
    # Calculer les contours dans la région centrale-basse
    edges_fine = cv2.Canny(gray, 100, 200)
    bin_region_edges = cv2.bitwise_and(edges_fine, mask)
    bin_edge_count = int(np.sum(bin_region_edges > 0))
    bin_region_area = int((x_end - x_start) * (y_end - y_start))
    
    return bin_region_edges, bin_edge_count, bin_region_area

def classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, hist_luminance_str=None, bin_edge_count=None, bin_area=None, patch_diversity=None):
    """
    Algorithme de classification automatique amélioré pour déterminer si une poubelle est vide ou pleine
    
    Améliorations:
    - Correction de la logique de luminosité
    - Ajout de l'analyse de la distribution de luminance
    - Utilisation des contours spécifiques à la région de la benne
    - Ajout de l'analyse de diversité de couleurs par patchs
    - Pondération des critères
    - Seuils adaptatifs selon la taille de l'image
    """
    
    # Paramètres de classification ajustables - OPTIMISÉS
    BRIGHTNESS_THRESHOLD = 135  # Seuil de luminosité moyenne (0-255) - OPTIMISÉ
    EDGE_DENSITY_BASE_THRESHOLD = 0.9   # Seuil de base pour la densité des contours - OPTIMISÉ
    BIN_EDGE_DENSITY_THRESHOLD = 0.12    # Seuil pour les contours dans la benne - OPTIMISÉ
    CONTRAST_THRESHOLD = 245  # Seuil de contraste (abaissé pour plus de sensibilité)
    PATCH_DIVERSITY_THRESHOLD = 0.1  # Seuil pour la diversité de couleurs par patchs - NOUVEAU
    PATCH_DIVERSITY_WEIGHT = 2  # Poids pour le critère de diversité de patchs - NOUVEAU
    
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
    
    # Critère 1: Luminosité (poids: 2.0) - CORRIGÉ: poubelle pleine = plus claire - OPTIMISÉ
    brightness_weight = 2.0  # OPTIMISÉ
    if avg_brightness > BRIGHTNESS_THRESHOLD:  # Correction: < au lieu de >
        criteria_scores['brightness'] = 1.0
        weighted_score += brightness_weight
    else:
        criteria_scores['brightness'] = 0.0
    total_weight += brightness_weight
    
    # Critère 2: Densité des contours dans la benne (prioritaire) ou globale (poids: 4.0) - OPTIMISÉ
    edge_weight = 4.0  # OPTIMISÉ
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

    # Critère 3: Diversité de couleurs par patchs (poids: 2.5) - NOUVEAU
    diversity_weight = PATCH_DIVERSITY_WEIGHT
    if patch_diversity is not None:
        if patch_diversity > PATCH_DIVERSITY_THRESHOLD:
            criteria_scores['patch_diversity'] = 1.0
            weighted_score += diversity_weight
        else:
            criteria_scores['patch_diversity'] = 0.0
        total_weight += diversity_weight
    else:
        criteria_scores['patch_diversity'] = 'N/A (non calculé)'

    # Critère 4: Non-uniformité de la luminance (poids: 1.0)
    uniformity_weight = 1.0
    uniformity_threshold = 0.5
    if luminance_uniformity > uniformity_threshold:  # Plus de variation = plus plein
        criteria_scores['uniformity'] = 1.0
        weighted_score += uniformity_weight
    else:
        criteria_scores['uniformity'] = 0.0
    total_weight += uniformity_weight
    
    # Score final normalisé
    final_score = weighted_score / total_weight
    confidence_score = abs(final_score - 0.5) * 2  # Distance de 0.5, normalisée
    
    # Classification avec seuil ajustable - OPTIMISÉ
    classification_threshold = 0.43  # Seuil optimisé
    
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
        'patch_diversity': round(patch_diversity, 4) if patch_diversity is not None else 'N/A',
        'patch_diversity_threshold': PATCH_DIVERSITY_THRESHOLD,
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
    filtre = request.args.get('filtre')
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    if filtre == 'pleine':
        c.execute("SELECT * FROM images WHERE annotation = 'pleine'")
    elif filtre == 'vide':
        c.execute("SELECT * FROM images WHERE annotation = 'vide'")
    else:
        c.execute("SELECT * FROM images")
    images = c.fetchall()
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
        return tr("Image non trouvée", "Image not found"), 404

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

    # Récupérer les données des bennes pour la carte
    c.execute("SELECT latitude, longitude, annotation, filename FROM images WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    bins = c.fetchall()

    conn.close()

    # Pie chart annotation
    labels = ['Pleine', 'Vide']
    values = [full_count, empty_count]
    fig, ax = plt.subplots()
    
    # Vérifier si on a des données pour le pie chart
    if total_images > 0 and (full_count > 0 or empty_count > 0):
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    else:
        # Si aucune donnée, afficher un message
        ax.text(0.5, 0.5, 'Aucune donnée\ndisponible', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    
    ax.axis('equal')
    pie_buf = BytesIO()
    plt.savefig(pie_buf, format='png')
    pie_buf.seek(0)
    pie_png = base64.b64encode(pie_buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # Histogram taille des fichiers
    fig, ax = plt.subplots()
    
    # Vérifier si on a des données pour l'histogramme
    if sizes and len(sizes) > 0:
        ax.hist(sizes, bins=10, color='skyblue')
        ax.set_title('Distribution des tailles de fichiers (Ko)')
        ax.set_xlabel('Taille (Ko)')
        ax.set_ylabel('Fréquence')
    else:
        # Si aucune donnée, afficher un message
        ax.text(0.5, 0.5, 'Aucune donnée\ndisponible', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
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
                           dates=dates,
                           bins=[{'lat': row[0], 'lng': row[1], 'annotation': row[2], 'filename': row[3]} for row in bins])


@app.route('/bin_map')
def bin_map():
    conn = sqlite3.connect('db.sqlite')
    c = conn.cursor()
    c.execute("SELECT latitude, longitude, annotation, filename FROM images WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    bins = c.fetchall()
    conn.close()
    return render_template('bin_map.html', bins=bins)

@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """
    Route pour supprimer une image de la base de données et du système de fichiers.
    
    Args:
        image_id (int): L'ID de l'image à supprimer
    
    Returns:
        flask.Response: Redirection vers la galerie d'images
    """
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
    """
    Route AJAX pour le traitement des uploads multiples avec progression.
    Gère les uploads asynchrones avec retour JSON du statut.
    
    Args:
        Aucun (utilise request.form et request.files)
    
    Returns:
        flask.Response: Réponse JSON avec les résultats du traitement
    """
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
            return jsonify({'error': tr('Aucun fichier valide trouvé', 'No valid file found')}), 400
        
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
        return jsonify({'error': tr(f'Erreur lors du traitement: {str(e)}', f'Error during processing: {str(e)}')}), 500
        

def calculate_file_hash(file_path):
    """
    Calcule le hash SHA-256 d'un fichier pour détecter les doublons.
    
    Args:
        file_path (str): Chemin vers le fichier
    
    Returns:
        str: Hash SHA-256 en hexadécimal
    """
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
            'message': tr(
                f"Fichier identique déjà analysé (même contenu): {hash_duplicate[1]}",
                f"Identical file already analyzed (same content): {hash_duplicate[1]}"
            )
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
            'message': tr(
                f"Fichier avec le même nom déjà analysé: {name_duplicate[1]}",
                f"File with the same name already analyzed: {name_duplicate[1]}"
            )
        }
    
    return False, None

@app.route('/reanalyze/<int:image_id>', methods=['POST'])
def reanalyze_image(image_id):
    """
    Route pour ré-analyser une image spécifique depuis la galerie.
    Recalcule toutes les caractéristiques et met à jour la base de données.
    
    Args:
        image_id (int): L'ID de l'image à ré-analyser
    
    Returns:
        flask.Response: Redirection vers la galerie avec message de statut
    """
    try:
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("SELECT filename FROM images WHERE id = ?", (image_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return redirect(url_for('images', message=tr("Image non trouvée", "Image not found")))
        
        filename = row[0]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(image_path):
            return redirect(url_for('images', message=tr(f"Fichier {filename} non trouvé sur le disque", f"File {filename} not found on disk")))
        
        # Ré-analyser l'image
        width, height, filesize, avg_color, contrast, edge_count, histogram, histogram_luminance, bin_edge_count, bin_area, patch_diversity = extract_features(image_path)
        avg_rgb = eval(avg_color)
        auto_classification, debug_info = classify_bin_automatic(avg_rgb, edge_count, contrast, width, height, histogram_luminance, bin_edge_count, bin_area, patch_diversity)
        file_hash = calculate_file_hash(image_path)
        
        # Mettre à jour la base de données
        conn = sqlite3.connect('db.sqlite')
        c = conn.cursor()
        c.execute("""UPDATE images SET 
            upload_date = ?, width = ?, height = ?, filesize = ?, avg_color = ?, 
            contrast = ?, edges = ?, histogram = ?, histogram_luminance = ?, 
            annotation = ?, bin_edges = ?, bin_area = ?, patch_diversity = ?, file_hash = ?
            WHERE id = ?""",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             width, height, filesize, avg_color, contrast, edge_count, histogram, 
             histogram_luminance, auto_classification, bin_edge_count, bin_area, 
             patch_diversity, file_hash, image_id))
        conn.commit()
        conn.close()
        
        return redirect(url_for('images', message=tr(f"Image {filename} ré-analysée avec succès", f"Image {filename} successfully re-analyzed")))
    except Exception as e:
        return redirect(url_for('images', message=tr(f"Erreur lors de la ré-analyse: {str(e)}", f"Error during re-analysis: {str(e)}")))



def tr(fr, en):
    """
    Fonction utilitaire de traduction simple basée sur la langue de session.
    
    Args:
        fr (str): Texte en français
        en (str): Texte en anglais
    
    Returns:
        str: Texte dans la langue appropriée selon session['lang']
    """
    lang = session.get('lang', 'fr')
    return fr if lang == 'fr' else en

if __name__ == '__main__':
    app.run(debug=True)
