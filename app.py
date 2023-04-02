from flask import Flask, render_template, request, send_file
from transformers import AutoFeatureExtractor, CvtForImageClassification
from PIL import Image

app = Flask(__name__)
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/cvt-13')
model = CvtForImageClassification.from_pretrained('microsoft/cvt-13')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Serve the uploaded image
@app.route('/temp/<filename>')
def serve_image(filename):
    return send_file(f'temp/{filename}', mimetype='image/jpeg')

# Predict the class label of the uploaded image and display the result
@app.route('/predict', methods=['POST'])
def predict():

    # Get the uploaded image
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    # Predict the class label of the uploaded image
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]

    # Save the uploaded image to a temporary folder
    temp_image_path = f"temp/{file.filename}"
    image.save(temp_image_path)

    # Pass the predicted class label and the temporary image path to the result page
    return render_template('result.html', label=predicted_class_label, image_url=temp_image_path)

    

if __name__ == '__main__':
    app.run(debug=False)
