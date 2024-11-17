import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.preprocessing import image

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn đến mô hình đã lưu
model_path = 'models/svm_model.pkl'  # Đảm bảo đúng đường dẫn tới mô hình
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Tải mô hình SVM từ file .pkl bằng joblib
model = joblib.load(model_path)

# Hàm tiền xử lý hình ảnh
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((180, 180))  # Resize theo kích thước ảnh đã huấn luyện
    img_array = np.array(img) / 255.0  # Rescale ảnh
    img_array = img_array.flatten()  # Chuyển ảnh thành vector 1D (flatten)
    img_array = img_array.reshape(1, -1)  # Đảm bảo rằng dữ liệu có dạng (1, n_features)
    return img_array

# Định tuyến trang chính
@app.route('/')
def index():
    return render_template('index.html')

# API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Lưu ảnh đã tải lên
        img_path = os.path.join(upload_folder, file.filename)
        file.save(img_path)

        # Tiền xử lý và dự đoán
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class = prediction[0]

        # Dự đoán tên lớp (cần ánh xạ lại với tên lớp)
        data_cat = {
            0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 
            6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 
            12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 
            18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 
            24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 
            30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
        }

        # Dự đoán tên lớp (sửa lại cho phù hợp với số lớp của bạn)
        predicted_label = data_cat[predicted_class]

        return jsonify({'prediction': predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
