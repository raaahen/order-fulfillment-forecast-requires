from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib

app = Flask(__name__)

# Инициализация глобальных переменных
model = RandomForestRegressor()
vectorizer = CountVectorizer()  # Для обработки SKU-кодов
encoder = OneHotEncoder(sparse_output=False)  # Для категориальных признаков

# Загрузка обученной модели и других файлов
try:
    model = joblib.load('model.pkl')
    encoder = joblib.load('encoder.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    pass

def prepare_data(data):
    categorical_features = ['type', 'processing_method', 'shipping_method']

    # Преобразование категориальных данных
    cat_data = np.array([[record[feature] for feature in categorical_features] for record in data])
    X_categorical = encoder.transform(cat_data)
    
    # Код для обработки SKU
    sku_ids = [record['sku_id'] for record in data]
    sku_vectors = vectorizer.transform(sku_ids)  # Преобразование SKU-кодов в вектор
    
    # Проверка размеров
    if X_categorical.shape[0] != sku_vectors.shape[0]:
        raise ValueError(f"Размеры не совпадают: X_categorical имеет {X_categorical.shape[0]} строк, а sku_vectors {sku_vectors.shape[0]} строк.")
    
    X = np.hstack([X_categorical, sku_vectors.toarray()])  # Преобразование sparse матрицы в dense
    return X

@app.route('/api/v1/order_fulfillment_forecast', methods=['POST'])
def order_fulfillment_forecast():
    try:
        data = request.json['order']
        X = prepare_data(data)
        predictions = model.predict(X)
        
        result = []
        for i, pred in enumerate(predictions):
            result.append({
                "order_id": data[i]['order_id'],
                "predicted_operations": {
                    "op_pallet_pick": pred[0],
                    "op_carton_pick": pred[1],
                    "op_unit_pick": pred[2],
                    "op_load_settle": pred[3],
                    "op_cont_settle": pred[4],
                    "op_load_deliver": pred[5],
                    "op_cont_deliver": pred[6]
                }
            })
        
        return jsonify({"forecast": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/machine_learning_by_order', methods=['POST'])
def machine_learning_by_order():
    try:
        data = request.json['data']
        targets = request.json['target']
        
        # Обучение CountVectorizer
        sku_ids = [record['sku_id'] for record in data]
        vectorizer.fit(sku_ids)  # Обучаем CountVectorizer на данных SKU

        # Преобразование данных
        categorical_features = ['type', 'processing_method', 'shipping_method']
        cat_data = np.array([[record[feature] for feature in categorical_features] for record in data])
        encoder.fit(cat_data)  # Обучаем OneHotEncoder на категориальных данных
        
        X = prepare_data(data)
        y = np.array([list(target.values()) for target in targets])
        
        model.fit(X, y)
        
        # Сохранение модели и векторов
        joblib.dump(model, 'model.pkl')
        joblib.dump(encoder, 'encoder.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')
        
        return jsonify({"message": "Модель обучена или дообучена успешно."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/get_coefficients', methods=['GET'])
def get_coefficients():
    try:
        if model is None:
            return jsonify({"error": "Модель не обучена."}), 404
        
        coefficients = model.feature_importances_
        return jsonify({"coefficients": coefficients.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
