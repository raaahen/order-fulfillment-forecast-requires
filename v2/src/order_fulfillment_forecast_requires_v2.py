from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Check if the model and label encoders exist
model_path = '/Users/sgzh1/projects/order-fulfillment-forecast-requires/v2/model/modelorder_fulfillment_model.pkl'
label_encoders_path = '/Users/sgzh1/projects/order-fulfillment-forecast-requires/v2/model/modellabel_encoders.pkl'

if os.path.exists(model_path) and os.path.exists(label_encoders_path):
    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)
else:
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
    label_encoders = {}

# Ensure all necessary label encoders are initialized
required_encoders = ['type_encoder', 'processing_method_encoder', 'day_encoder', 'shift_encoder']
for encoder_name in required_encoders:
    if encoder_name not in label_encoders:
        label_encoders[encoder_name] = LabelEncoder()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Encode the input data
    type_encoded = label_encoders['type_encoder'].transform([data['type']])[0]
    processing_method_encoded = label_encoders['processing_method_encoder'].transform([data['processing_method']])[0]
    day_encoded = label_encoders['day_encoder'].transform([data['day']])[0]
    shift_encoded = label_encoders['shift_encoder'].transform([data['shift']])[0]
    
    # Prepare the input for the model
    input_data = np.array([[type_encoded, processing_method_encoded, day_encoded, shift_encoded, data['details'], data['sku_id'], data['qty']]])
    
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
    file_path = request.json.get('file_path')

    # Load new data
    new_data = pd.read_csv(file_path, dtype={'TYPE': str, 'PROCESSING_METHOD': str})

    # Encode categorical features if not already encoded
    for feature in ['type', 'processing_method', 'day', 'shift']:
        encoder = label_encoders[f'{feature}_encoder']
        if not hasattr(encoder, 'classes_'):
            encoder.fit(new_data[feature.upper()])
        new_data[f'{feature}_encoded'] = encoder.transform(new_data[feature.upper()])

    # Prepare the features and targets
    X_new = new_data[['type_encoded', 'processing_method_encoded', 'day_encoded', 'shift_encoded', 'sku_id', 'details', 'qty']]
    y_new = new_data[['op_pallet_pick', 'op_cont_pick', 'op_cont_deliver', 'op_cont_aboard', 'op_load_deliver']]

    # Fine-tune the model
    model.fit(X_new, y_new)

    # Save the updated model and encoders
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, label_encoders_path)

    return jsonify({'message': 'Model fine-tuned successfully!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
