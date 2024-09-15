from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json

app = Flask(__name__)

# Инициализируем модель
model = RandomForestRegressor()
trained = False

# Пример начальных данных для обучения (упрощенно)
# Здесь используются закодированные категории признаков: type, processing_method, shipping_method, sku_id
initial_data = {
    "X": [
        [0, 0, 0, 1],  # Пример заказа 1: (Доставка, Сборка в док, Вагон, SKU123)
        [1, 1, 1, 2],  # Пример заказа 2: (Самовывоз, Сборка у двери, Самовывоз, SKU456)
    ],
    "y": [
        [0.5, 0.4, 0.0, 0.1, 0.05, 0.5, 0.02],  # Проценты операций для заказа 1
        [0.3, 0.6, 0.1, 0.0, 0.0, 0.3, 0.1],  # Проценты операций для заказа 2
    ]
}

@app.route('/api/v1/order_fulfillment_forecast', methods=['POST'])
def forecast():
    global trained
    if not trained:
        return jsonify({"error": "Model is not trained yet!"}), 400
    
    data = request.json
    # Преобразуем входные данные в формат для модели
    order_features = encode_features(data)
    
    # Прогнозируем проценты операций
    predictions = model.predict([order_features])
    
    # Формируем ответ
    response = {
        "op_pallet_pick": predictions[0][0],
        "op_carton_pick": predictions[0][1],
        "op_unit_pick": predictions[0][2],
        "op_load_settle": predictions[0][3],
        "op_cont_settle": predictions[0][4],
        "op_load_deliver": predictions[0][5],
        "op_cont_deliver": predictions[0][6]
    }
    
    return jsonify(response)

@app.route('/api/v1/machine_learning_by_order', methods=['POST'])
def train():
    global trained
    
    # Получаем данные для обучения
    data = request.json
    X_train = data["X"]
    y_train = data["y"]
    
    # Обучаем модель
    model.fit(X_train, y_train)
    trained = True
    
    return jsonify({"status": "Model trained successfully!"})

def encode_features(data):
    # Пример кодирования признаков заказа
    # Здесь ты можешь использовать one-hot encoding или более сложные трансформации
    type_mapping = {"Доставка": 0, "Самовывоз": 1}
    processing_method_mapping = {"Сборка в док": 0, "Сборка у двери": 1}
    shipping_method_mapping = {"Вагон": 0, "Самовывоз": 1, "Фура": 2}
    
    encoded = [
        type_mapping.get(data["type"], 0),
        processing_method_mapping.get(data["processing_method"], 0),
        shipping_method_mapping.get(data["shipping_method"], 0),
        int(data["sku_id"][-3:])  # Пример кодирования SKU через последние цифры
    ]
    
    return encoded

if __name__ == '__main__':
    app.run(debug=True)
