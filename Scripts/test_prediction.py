#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import DiabetesPredictionModel

def test_prediction():
    print("Testing Diabetes Prediction Model...")
    
    try:
        # Initialize the model
        model = DiabetesPredictionModel()
        
        # Load the trained model
        print("Loading trained model...")
        try:
            model.load_model()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            
            # Train the model
            X, y = model.load_and_preprocess_data('diabetes.csv')
            X_train, X_test, y_train, y_test = model.split_and_scale_data(X, y)
            model.train_model(X_train, y_train)
            model.save_model()
            print("Model trained and saved")
        
        # Test prediction with sample data
        print("\nTesting prediction...")
        test_data = {
            'Pregnancies': 0,
            'Glucose': 120,
            'BloodPressure': 80,
            'SkinThickness': 0,
            'Insulin': 0,
            'BMI': 28.5,
            'DiabetesPedigreeFunction': 0.5,
            'Age': 45
        }
        
        print(f"Input data: {test_data}")
        
        result = model.predict_diabetes_risk(test_data)
        print(f"Prediction result: {result}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
    input("\nPress Enter to exit...")
