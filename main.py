from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os

app = Flask(__name__)

# Инициализация моделей и векторизаторов
label_encoders = {}
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
model = None
model_file = "model.joblib"
all_operations = set()  # Набор всех известных операций

# Функция для кодирования категориальных данных
def encode_categorical_data(records):
    global label_encoders
    encoded_data = []
    for record in records:
        encoded_record = {}
        for key, value in record.items():
            if key != 'sku_id':  # Исключаем SKU, он кодируется через TF-IDF
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

# Обработка данных для динамического расширения признаков
def prepare_data(data):
    global label_encoders, all_operations

    # Декодирование данных заказа
    encoded_data = encode_categorical_data(data)
    sku_ids = [record['sku_id'] for record in data]
    sku_vectors = vectorizer.fit_transform(sku_ids)

    # Преобразование категориальных данных и SKU в векторы
    X_categorical = np.column_stack([list(record.values())[:-1] for record in encoded_data])
    X = np.hstack([X_categorical, sku_vectors.toarray()])

    return X

# Маршрут для прогноза
@app.route('/api/v1/order_fulfillment_forecast', methods=['POST'])
def order_fulfillment_forecast():
    load_model()  # Загружаем модель, если она существует
    if model is None:
        return jsonify({"error": "Модель не обучена."}), 400

    # Получаем данные из запроса
    data = request.json.get("order")
    
    if not isinstance(data, list):
        data = [data]

    # Обработка данных заказа
    X_new = prepare_data(data)

    # Предсказание
    predictions = model.predict(X_new)

    # Формируем JSON-ответ
    results = []
    for i, prediction in enumerate(predictions):
        result = {operation: prediction[j] for j, operation in enumerate(all_operations)}
        results.append({"order_id": data[i]["order_id"], "predicted_operations": result})
    
    return jsonify({"forecast": results})

# Маршрут для обучения или дообучения
@app.route('/api/v1/machine_learning_by_order', methods=['POST'])
def machine_learning_by_order():
    global all_operations

    # Получаем данные для обучения
    data = request.json.get("data")
    target = request.json.get("target")

    # Обрабатываем все новые типы операций
    for target_entry in target:
        all_operations.update(target_entry.keys())

    # Кодируем данные заказа
    X = prepare_data(data)

    # Преобразуем операции в массив для обучения модели
    y = np.array([[target_entry.get(op, 0) for op in all_operations] for target_entry in target])

    # Если модель существует, дообучаем, иначе обучаем с нуля
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
