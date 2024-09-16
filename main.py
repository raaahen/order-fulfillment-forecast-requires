from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os

app = Flask(__name__)

# Инициализация моделей и векторизаторов
label_encoders = {}
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
model = None
model_file = "model.joblib"

# Список операций
operation_names = ['op_pallet_pick', 'op_carton_pick', 'op_unit_pick', 'op_load_settle', 'op_cont_settle', 'op_load_deliver', 'op_cont_deliver']

# Функция для кодирования категориальных данных
def encode_categorical_data(records):
    global label_encoders
    encoded_data = []
    for record in records:
        encoded_record = {}
        for key, value in record.items():
            if key != 'sku_id':  # Исключаем SKU
                if key not in label_encoders:
                    label_encoders[key] = LabelEncoder()
                    label_encoders[key].fit([r[key] for r in records])
                encoded_record[key] = label_encoders[key].transform([value])[0]
        encoded_data.append(encoded_record)
    return encoded_data

# Функция для обучения модели
def train_model(X, y):
    global model
    model = RandomForestRegressor()
    model.fit(X, y)
    joblib.dump(model, model_file)

# Загрузка модели из файла
def load_model():
    global model
    if os.path.exists(model_file):
        model = joblib.load(model_file)

# Маршрут для прогноза
@app.route('/api/v1/order_fulfillment_forecast', methods=['POST'])
def order_fulfillment_forecast():
    load_model()  # Загружаем модель, если она существует
    if model is None:
        return jsonify({"error": "Модель не обучена."}), 400
    
    # Получаем данные из запроса
    data = request.json
    if not isinstance(data, list):
        data = [data]
    
    # Кодируем категориальные данные и SKU
    encoded_data = encode_categorical_data(data)
    sku_ids = [record['sku_id'] for record in data]
    sku_vectors = vectorizer.transform(sku_ids)

    # Преобразуем категориальные данные и векторы SKU
    X_categorical = np.column_stack([list(record.values())[:-1] for record in encoded_data])
    X_new = np.hstack([X_categorical, sku_vectors.toarray()])

    # Предсказание
    predictions = model.predict(X_new)

    # Формируем JSON-ответ
    results = []
    for i, prediction in enumerate(predictions):
        result = {operation_names[j]: prediction[j] for j in range(len(operation_names))}
        results.append({"order": data[i], "predicted_operations": result})
    
    return jsonify(results)

# Маршрут для обучения или дообучения
@app.route('/api/v1/machine_learning_by_order', methods=['POST'])
def machine_learning_by_order():
    # Получаем данные для обучения
    data = request.json.get("data")
    y = request.json.get("target")
    
    # Кодируем данные
    encoded_data = encode_categorical_data(data)
    sku_ids = [record['sku_id'] for record in data]
    sku_vectors = vectorizer.fit_transform(sku_ids)

    # Преобразуем данные
    X_categorical = np.column_stack([list(record.values())[:-1] for record in encoded_data])
    X = np.hstack([X_categorical, sku_vectors.toarray()])

    # Если модель существует, то дообучаем, иначе обучаем с нуля
    if model is not None:
        model.fit(X, y)
    else:
        train_model(X, y)
    
    return jsonify({"message": "Модель обучена или дообучена успешно."})

# Запуск Flask-сервера
if __name__ == "__main__":
    if os.path.exists(model_file):
        load_model()
    app.run(debug=True)
