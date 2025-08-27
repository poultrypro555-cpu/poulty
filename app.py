from flask import Flask, render_template, request, redirect, url_for, make_response
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
load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Initialize Firebase with environment variables
def initialize_firebase():
    try:
        # Get Firebase configuration from environment variables
        firebase_config = {
            "type": os.environ.get('FIREBASE_TYPE', 'service_account'),
            "project_id": os.environ.get('FIREBASE_PROJECT_ID'),
            "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": os.environ.get('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
            "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.environ.get('FIREBASE_CLIENT_ID'),
            "auth_uri": os.environ.get('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
            "token_uri": os.environ.get('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
            "auth_provider_x509_cert_url": os.environ.get('FIREBASE_AUTH_PROVIDER_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
            "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_CERT_URL'),
            "universe_domain": os.environ.get('FIREBASE_UNIVERSE_DOMAIN', 'googleapis.com')
        }
        
        # Check if required variables are set
        required_vars = ['FIREBASE_PROJECT_ID', 'FIREBASE_PRIVATE_KEY_ID', 
                         'FIREBASE_PRIVATE_KEY', 'FIREBASE_CLIENT_EMAIL',
                         'FIREBASE_CLIENT_CERT_URL']
        
        for var in required_vars:
            if not os.environ.get(var):
                print(f"Missing required environment variable: {var}")
                return None
        
        # Create credentials from the config
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully")
        return firestore.client()
    except Exception as e:
        print(f"Firebase initialization failed: {str(e)}")
        return None

db = initialize_firebase()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load model
model = tf.keras.models.load_model('poultry_disease_mobilenetv2.h5')
class_names = ['COCCIDIOSIS', 'HEALTHY', 'SALMONELLA']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)], float(np.max(prediction))

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
            file.save(filepath)
            
            prediction, confidence = predict_image(filepath)
            confidence_percent = round(confidence * 100, 2)
            
            report_data = {
                'college': "K.L.N. COLLEGE OF ENGINEERING",
                'department': "ELECTRONICS AND COMMUNICATION ENGINEERING",
                'date': datetime.now().strftime("%d-%m-%Y"),
                'time': datetime.now().strftime("%I:%M %p"),
                'prediction': prediction,
                'confidence': confidence_percent
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
                # Convert Firestore timestamp to readable date
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
    
    # Generate CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Time', 'Prediction', 'Confidence'])
    for result in results:
        writer.writerow([
            result.get('date', ''),
            result.get('time', ''),
            result.get('prediction', ''),
            f"{result.get('confidence', 0)}%"
        ])
    
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=poultry_results.csv'
    response.headers['Content-type'] = 'text/csv'
    return response

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)