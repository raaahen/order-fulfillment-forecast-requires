import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoders
model_path = '/Users/sgzh1/projects/order-fulfillment-forecast-requires/v1/model/modelorder_fulfillment_model.pkl'
label_encoders_path = '/Users/sgzh1/projects/order-fulfillment-forecast-requires/v1/model/modellabel_encoders.pkl'

if os.path.exists(model_path) and os.path.exists(label_encoders_path):
    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)
else:
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
    label_encoders = {'type_encoder': None, 'processing_method_encoder': None}

def safe_transform(label_encoder, value, default_value):
    """
    Safely transform a value using a LabelEncoder.
    If the value is not in the encoder's classes_, return a default value.
    """
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        return label_encoder.transform([default_value])[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Encode the input data, handling unknown values
    type_encoded = safe_transform(label_encoders['type_encoder'], data['type'], label_encoders['type_encoder'].classes_[0])
    processing_method_encoded = safe_transform(label_encoders['processing_method_encoder'], data['processing_method'], label_encoders['processing_method_encoder'].classes_[0])
    
    # Prepare the input for the model
    input_data = np.array([[type_encoded, processing_method_encoded, data['sku_id']]])
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Return the predictions
    return jsonify({
        'op_pallet_pick': predictions[0][0],
        'op_cont_pick': predictions[0][1],
        'op_cont_deliver': predictions[0][2],
        'op_cont_aboard': predictions[0][3],
        'op_load_deliver': predictions[0][4],
    })

@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    # Get the path to the CSV file
    file_path = request.json.get('file_path')
    
    # Load new data from the CSV file
    new_data = pd.read_csv(file_path)

    # Preprocess the new data (similar to initial training), handling unknown values
    new_data['type_encoded'] = new_data['type'].apply(lambda x: safe_transform(label_encoders['type_encoder'], x, label_encoders['type_encoder'].classes_[0]))
    new_data['processing_method_encoded'] = new_data['processing_method'].apply(lambda x: safe_transform(label_encoders['processing_method_encoder'], x, label_encoders['processing_method_encoder'].classes_[0]))

    # Features and target variables
    X_new = new_data[['type_encoded', 'processing_method_encoded', 'sku_id']]
    y_new = new_data[['op_pallet_pick', 'op_cont_pick', 'op_cont_deliver', 'op_cont_aboard', 'op_load_deliver']]

    # Retrain the model with new data
    model.fit(X_new, y_new)

    # Save the updated model
    joblib.dump(model, '/Users/sgzh1/projects/order-fulfillment-forecast-requires/v1/model/modelorder_fulfillment_model.pkl')

    return jsonify({'message': 'Model fine-tuned successfully!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
