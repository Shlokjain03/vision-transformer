import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from vit_model import VisionTransformerClassifier
from database import init_db, save_prediction, get_all_predictions

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionTransformerClassifier(device=device)
# Initialize DB
init_db()
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if  post request has  file part
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            # Predict
            predicted_class, confidence = model.predict(save_path)
            # Save to DB
            save_prediction(save_path, predicted_class, confidence)
            predictions = get_all_predictions()
            return render_template('index.html', prediction=predicted_class, confidence=confidence, image_path=save_path, history=predictions)
    predictions = get_all_predictions()
    return render_template('index.html', history=predictions)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
