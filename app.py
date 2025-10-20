from flask import Flask, render_template, request, redirect, make_response
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import csv
from io import StringIO
from dotenv import load_dotenv
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load environment variables
load_dotenv()

# TensorFlow GPU optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.set_soft_device_placement(True)

app = Flask(__name__)

# Initialize Firebase
def initialize_firebase():
    try:
        # Use service account key file if available, otherwise use environment variables
        if os.path.exists('serviceAccountKey.json'):
            cred = credentials.Certificate('serviceAccountKey.json')
        else:
            firebase_config = {
                "type": os.environ.get('FIREBASE_TYPE', 'service_account'),
                "project_id": os.environ.get('FIREBASE_PROJECT_ID'),
                "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID'),
                "private_key": os.environ.get('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
                "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.environ.get('FIREBASE_CLIENT_ID'),
                "auth_uri": os.environ.get('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.environ.get('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": os.environ.get('FIREBASE_AUTH_PROVIDER_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_CERT_URL'),
                "universe_domain": os.environ.get('FIREBASE_UNIVERSE_DOMAIN', 'googleapis.com')
            }
            cred = credentials.Certificate(firebase_config)
        
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully")
        return firestore.client()
    except Exception as e:
        print(f"Firebase initialization failed: {str(e)}")
        return None

db = initialize_firebase()

# Flask config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load your EfficientNetB3 model
try:
    # Try multiple possible paths for the model
    model_paths = [
        'poultry_disease_mobilenetv2.h5'
    ]
    
    model = None
    for model_path in model_paths:
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            break
        except:
            continue
    
    if model is None:
        raise Exception("Could not find model file in any known location")
    
    # Get class names from your training - MUST MATCH YOUR ACTUAL TRAINING ORDER!
    class_names = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
    print(f"Class names: {class_names}")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None
    class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    if model is None:
        return "Model not loaded", 0.0, []
    
    try:
        # Load and preprocess image for EfficientNetB3 (300x300)
        img = Image.open(image_path).convert('RGB')
        
        # Resize to match your model's input size (300x300 for EfficientNetB3)
        img = img.resize((300, 300))
        
        # Convert to array
        img_array = np.array(img)
        
        # EXPAND DIMS FIRST
        img_array = np.expand_dims(img_array, axis=0)
        
        # USE THE EXACT SAME PREPROCESSING AS YOUR TRAINING
        # For EfficientNet models, use the built-in preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # DEBUG: Print raw predictions
        print(f"Raw prediction values: {prediction}")
        print(f"Predicted class index: {np.argmax(prediction)}")
        
        # Get top prediction
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Get confidence scores for all classes
        confidence_scores = []
        for i, class_name in enumerate(class_names):
            confidence_scores.append({
                'class': class_name,
                'confidence': float(prediction[0][i]) * 100
            })
        
        # Sort by confidence descending
        confidence_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # DEBUG: Print final results
        print(f"Final prediction: {predicted_class} with {confidence*100:.2f}% confidence")
        for score in confidence_scores:
            print(f"  {score['class']}: {score['confidence']:.2f}%")
        
        return predicted_class, confidence * 100, confidence_scores
        
    except Exception as e:
        print(f"Error predicting image: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0, []

def save_to_firestore(data):
    if not db:
        print("Firestore not available")
        return False
    try:
        doc_ref = db.collection('poultry_results').document()
        doc_ref.set({
            'college': data['college'],
            'department': data['department'],
            'date': data['date'],
            'time': data['time'],
            'prediction': data['prediction'],
            'confidence': data['confidence'],
            'all_predictions': data.get('all_predictions', []),
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print("Data saved to Firestore")
        return True
    except Exception as e:
        print(f"Error saving to Firestore: {str(e)}")
        return False

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)

            prediction, confidence, all_predictions = predict_image(filepath)
            confidence_percent = round(confidence, 2)

            report_data = {
                'college': "K.L.N. COLLEGE OF ENGINEERING",
                'department': "ELECTRONICS AND COMMUNICATION ENGINEERING",
                'date': datetime.now().strftime("%d-%m-%Y"),
                'time': datetime.now().strftime("%I:%M %p"),
                'prediction': prediction,
                'confidence': confidence_percent,
                'all_predictions': all_predictions,
                'image_filename': filename
            }

            save_to_firestore(report_data)
            return render_template('result.html', report=report_data)
    
    return render_template('upload.html')

@app.route('/history')
def history():
    results = []
    if db:
        try:
            docs = db.collection('poultry_results')\
                     .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                     .stream()
            for doc in docs:
                data = doc.to_dict()
                if 'timestamp' in data:
                    timestamp = data['timestamp']
                    data['date'] = timestamp.strftime("%d-%m-%Y")
                    data['time'] = timestamp.strftime("%I:%M %p")
                results.append(data)
        except Exception as e:
            print(f"Error fetching history: {str(e)}")
    return render_template('history.html', results=results)

@app.route('/download')
def download():
    results = []
    if db:
        try:
            docs = db.collection('poultry_results').stream()
            results = [doc.to_dict() for doc in docs]
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Time', 'Prediction', 'Confidence', 'College', 'Department'])
    
    for result in results:
        writer.writerow([
            result.get('date', ''),
            result.get('time', ''),
            result.get('prediction', ''),
            f"{result.get('confidence', 0)}%",
            result.get('college', ''),
            result.get('department', '')
        ])
    
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=poultry_results.csv'
    response.headers['Content-type'] = 'text/csv'
    return response

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
