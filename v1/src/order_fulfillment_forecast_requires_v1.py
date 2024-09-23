from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('/Users/sgzh1/projects/order-fulfillment-forecast-requires/v1/model/modelorder_fulfillment_model.pkl')
label_encoders = joblib.load('/Users/sgzh1/projects/order-fulfillment-forecast-requires/v1/model/modellabel_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Encode the input data
    type_encoded = label_encoders['type_encoder'].transform([data['type']])[0]
    pm_encoded = label_encoders['pm_encoder'].transform([data['processing_method']])[0]
    
    # Prepare the input for the model
    input_data = np.array([[type_encoded, pm_encoded, data['sku_id']]])
    
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

    # Preprocess the new data (similar to initial training)
    new_data['type_encoded'] = label_encoders['type_encoder'].transform(new_data['type'])
    new_data['processing_method_encoded'] = label_encoders['pm_encoder'].transform(new_data['processing_method'])

    # Features and target variables
    X_new = new_data[['type_encoded', 'processing_method_encoded', 'sku_id']]
    y_new = new_data[['op_pallet_pick', 'op_cont_pick', 'op_cont_deliver', 'op_cont_aboard', 'op_load_deliver']]

    # Combine old and new data for retraining
    X_old, y_old = model.estimator_.estimators_[0].tree_.feature_importances_.reshape(-1, 3), model.estimator_.estimators_[0].tree_.value
    X_combined = np.vstack((X_old, X_new))
    y_combined = np.vstack((y_old, y_new))

    # Retrain the model
    model.fit(X_combined, y_combined)

    # Save the updated model
    joblib.dump(model, '/Users/sgzh1/projects/order-fulfillment-forecast-requires/v1/model/modelorder_fulfillment_model.pkl')

    return jsonify({'message': 'Model fine-tuned successfully!'}), 200

if __name__ == '__main__':
    app.run(debug=True)