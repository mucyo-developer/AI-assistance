import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the diabetes dataset"""
        # Load the data
        df = pd.read_csv(file_path)
        
        # Display basic information about the dataset
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Handle missing values (replace 0 with NaN for medical measurements that shouldn't be 0)
        columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in columns_with_zeros:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        self.feature_names = X.columns.tolist()
        
        print(f"\nClass distribution:")
        print(y.value_counts())
        print(f"Diabetes prevalence: {y.mean():.2%}")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Split data and apply feature scaling"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
        # Display feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return feature_importance
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return accuracy, y_pred, y_pred_proba
    
    def predict_diabetes_risk(self, patient_data):
        """Predict diabetes risk for new patient data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure data is in the right format and correct feature order
        if isinstance(patient_data, dict):
            # Create DataFrame with features in the correct order
            patient_df = pd.DataFrame([patient_data])
            # Reorder columns to match training data
            patient_df = patient_df[self.feature_names]
        else:
            patient_df = pd.DataFrame(patient_data)
            # Reorder columns to match training data
            patient_df = patient_df[self.feature_names]
        
        # Scale the data
        patient_scaled = self.scaler.transform(patient_df)
        
        # Make prediction
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0, 1]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
    
    def save_model(self, model_path='diabetes_model.pkl', scaler_path='scaler.pkl'):
        """Save the trained model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='diabetes_model.pkl', scaler_path='scaler.pkl'):
        """Load a trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Ensure feature names are set correctly
        if self.feature_names is None:
            self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        print("Model and scaler loaded successfully!")
    
    def create_visualizations(self, X, y, feature_importance):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of features by outcome
        for i, feature in enumerate(['Glucose', 'BMI', 'Age', 'Insulin']):
            row, col = i // 2, i % 2
            for outcome in [0, 1]:
                sns.histplot(X[feature][y == outcome], 
                           label=f'No Diabetes' if outcome == 0 else 'Diabetes',
                           alpha=0.6, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig('diabetes_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature')
        plt.title('Feature Importance in Diabetes Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the diabetes prediction pipeline"""
    # Initialize the model
    diabetes_model = DiabetesPredictionModel()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = diabetes_model.load_and_preprocess_data('diabetes.csv')
    
    # Split and scale data
    print("\nSplitting and scaling data...")
    X_train, X_test, y_train, y_test = diabetes_model.split_and_scale_data(X, y)
    
    # Train the model
    print("\nTraining model...")
    feature_importance = diabetes_model.train_model(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model...")
    accuracy, y_pred, y_pred_proba = diabetes_model.evaluate_model(X_test, y_test)
    
    # Create visualizations
    print("\nCreating visualizations...")
    diabetes_model.create_visualizations(X, y, feature_importance)
    
    # Save the model
    diabetes_model.save_model()
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Example 1: High risk patient
    high_risk_patient = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    result1 = diabetes_model.predict_diabetes_risk(high_risk_patient)
    print(f"High Risk Patient: {result1}")
    
    # Example 2: Low risk patient
    low_risk_patient = {
        'Pregnancies': 1,
        'Glucose': 85,
        'BloodPressure': 66,
        'SkinThickness': 29,
        'Insulin': 0,
        'BMI': 26.6,
        'DiabetesPedigreeFunction': 0.351,
        'Age': 31
    }
    
    result2 = diabetes_model.predict_diabetes_risk(low_risk_patient)
    print(f"Low Risk Patient: {result2}")
    
    return diabetes_model

if __name__ == "__main__":
    model = main()