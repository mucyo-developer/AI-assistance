from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import re

app = Flask(__name__)
app.secret_key = 'diabetes_assistant_2024'

class WorkingDiabetesAssistant:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load the diabetes prediction model
        self.load_prediction_model()
        
        print("✅ Working Diabetes Assistant initialized")
    
    def load_prediction_model(self):
        """Load the trained diabetes prediction model"""
        try:
            self.model = joblib.load('diabetes_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            print("✅ Diabetes prediction model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Training new diabetes prediction model...")
            self.train_new_model()
    
    def train_new_model(self):
        """Train a new diabetes prediction model"""
        # Load and preprocess data
        df = pd.read_csv('diabetes.csv')
        
        # Handle missing values
        columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in columns_with_zeros:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        self.feature_names = X.columns.tolist()
        
        # Split and scale data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(self.model, 'diabetes_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        print("✅ Model trained and saved successfully")
    
    def predict_diabetes_risk(self, metrics: dict) -> dict:
        """Predict diabetes risk based on health metrics"""
        try:
            # Ensure data is in the right format and correct feature order
            patient_df = pd.DataFrame([metrics])
            patient_df = patient_df[self.feature_names]
            
            # Scale the data
            patient_scaled = self.scaler.transform(patient_df)
            
            # Make prediction
            prediction = self.model.predict(patient_scaled)[0]
            probability = self.model.predict_proba(patient_scaled)[0, 1]
            
            risk_level = 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_diabetes_info(self, topic: str) -> str:
        """Get diabetes information based on topic"""
        topic_lower = topic.lower()
        
        if 'diabetes' in topic_lower or any(word in topic_lower for word in ['what is', 'tell me about', 'explain']):
            return """🩺 About Diabetes:

What is Diabetes?
Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).

Types of Diabetes:
• Type 1: Autoimmune condition, body doesn't produce insulin
• Type 2: Body doesn't use insulin properly (most common type)
• Gestational: Develops during pregnancy

Common Symptoms:
• Frequent urination
• Excessive thirst
• Unexplained weight loss
• Fatigue
• Blurred vision

Risk Factors:
• Family history
• Age over 45
• Overweight/obesity
• Physical inactivity
• High blood pressure

Management:
• Regular blood sugar monitoring
• Healthy diet
• Regular exercise
• Medications as prescribed
• Regular medical check-ups"""
        
        elif any(word in topic_lower for word in ['symptom', 'sign']):
            return """🩺 Diabetes Symptoms:

Common Symptoms:
• Frequent urination - especially at night
• Excessive thirst - feeling very thirsty
• Extreme hunger - constant hunger
• Unexplained weight loss - despite eating more
• Fatigue - feeling tired and weak
• Blurred vision - difficulty focusing

Additional Symptoms:
• Slow-healing sores or cuts
• Frequent infections (skin, gum, bladder)
• Numbness or tingling in hands/feet
• Dry, itchy skin
• Irritability

Important: Some people with Type 2 diabetes may have no symptoms for years. Regular check-ups are important for early detection."""
        
        elif any(word in topic_lower for word in ['prevent', 'avoid', 'reduce risk']):
            return """🩺 Diabetes Prevention:

Lifestyle Changes:
• Maintain healthy weight - even 5-7% weight loss helps
• Regular physical activity - 150 minutes per week
• Healthy eating - balanced diet, limit processed foods
• Quit smoking - increases diabetes risk

Diet Recommendations:
• Eat more fruits, vegetables, whole grains
• Choose lean proteins (fish, chicken, beans)
• Limit sugary drinks and refined carbs
• Control portion sizes
• Drink plenty of water

Exercise Tips:
• Start with 10-15 minutes daily
• Gradually increase to 30 minutes most days
• Include both cardio and strength training
• Find activities you enjoy

Regular Check-ups:
• Annual physical exams
• Blood sugar testing if at risk
• Blood pressure monitoring
• Cholesterol checks"""
        
        elif any(word in topic_lower for word in ['diet', 'food', 'eat', 'nutrition']):
            return """🩺 Diabetes Diet Guide:

Foods to Emphasize:
• Non-starchy vegetables - broccoli, spinach, cauliflower
• Lean proteins - chicken, fish, tofu, beans
• Whole grains - quinoa, brown rice, whole wheat bread
• Healthy fats - avocado, nuts, olive oil
• Low-glycemic fruits - berries, apples, citrus

Foods to Limit:
• Sugary beverages (soda, sweetened juices)
• White bread, rice, pasta
• Fried foods
• Processed snacks
• High-fat dairy products
• Sweets and desserts

Meal Planning:
• Eat consistent meals at regular times
• Don't skip meals
• Control portion sizes
• Balance carbs with protein and healthy fats
• Read food labels carefully

Carb Counting:
• Aim for 45-60g carbs per meal
• Work with a dietitian for personalized plan
• Consider carb counting for better control"""
        
        elif any(word in topic_lower for word in ['exercise', 'activity', 'fitness']):
            return """🩺 Exercise for Diabetes:

Recommended Activity:
• 150 minutes per week of moderate aerobic activity
• 2-3 sessions of strength training weekly
• Daily movement - walking, stretching

Best Exercises:
• Walking - easy, accessible, effective
• Swimming - low impact, full body workout
• Cycling - good for cardiovascular health
• Strength training - builds muscle, improves insulin sensitivity
• Yoga - reduces stress, improves flexibility

Safety Guidelines:
• Check blood sugar before and after exercise
• Carry fast-acting carbs during exercise
• Stay hydrated
• Wear proper footwear
• Avoid exercise if blood sugar >300 mg/dL

Benefits:
• Lowers blood sugar levels
• Improves insulin sensitivity
• Helps with weight management
• Reduces cardiovascular risk
• Improves mood and energy"""
        
        else:
            return """🩺 How can I help you with diabetes?

I can provide information about:
• What is diabetes - overview and types
• Symptoms - signs to watch for
• Prevention - how to reduce your risk
• Diet - eating guidelines for diabetes
• Exercise - physical activity recommendations
• Risk assessment - evaluate your personal risk factors

Ask me about:
- "Tell me about diabetes"
- "What are the symptoms?"
- "How can I prevent diabetes?"
- "What should I eat?"
- "What exercise should I do?"

You can also use the risk assessment form to evaluate your diabetes risk based on your health metrics."""
    
    def get_blood_sugar_advice(self, glucose: int) -> str:
        """Provide advice based on blood sugar reading"""
        if glucose < 70:
            return """🔴 Low Blood Sugar (Hypoglycemia):

Immediate Action:
• Eat/drink 15-20g fast-acting carbs
• Examples: 4 oz juice, regular soda, glucose tablets
• Wait 15 minutes and recheck
• Repeat if still low

Prevention:
• Eat regular meals
• Don't skip meals
• Monitor blood sugar regularly
• Adjust medications as needed"""
        
        elif glucose > 250:
            return """🔴 High Blood Sugar (Hyperglycemia):

Immediate Action:
• Drink plenty of water
• Take prescribed medications
• Monitor blood sugar frequently
• Avoid strenuous exercise if >300 mg/dL

Contact doctor if:
• Blood sugar remains high for several hours
• You have ketones
• You feel very unwell"""
        
        elif 70 <= glucose <= 140:
            return """🟢 Normal Blood Sugar:

Great job! Your blood sugar is in the target range.

Keep up the good work:
• Continue current management plan
• Regular monitoring
• Healthy lifestyle habits
• Follow your healthcare provider's advice"""
        
        else:
            return """🟡 Elevated Blood Sugar:

Monitor closely:
• Check blood sugar more frequently
• Review recent meals and activity
• Consider light exercise (if approved)
• Contact doctor if consistently high

Prevention:
• Follow meal plan
• Take medications as prescribed
• Regular exercise
• Stress management"""

# Initialize the assistant
assistant = WorkingDiabetesAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({
                'response': 'Error: No message provided',
                'success': False
            })
        
        user_input = data.get('message', '').strip()
        if not user_input:
            return jsonify({
                'response': 'Please enter a message.',
                'success': False
            })
        
        # Check for blood sugar reading
        glucose_match = re.search(r'blood sugar.*?(\d+)|glucose.*?(\d+)', user_input.lower())
        if glucose_match:
            glucose = int(glucose_match.group(1) or glucose_match.group(2))
            response = assistant.get_blood_sugar_advice(glucose)
        else:
            # Get diabetes information
            response = assistant.get_diabetes_info(user_input)
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'response': f'I apologize, but I encountered an error: {str(e)}. Please try again.',
            'success': False
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'metrics' not in data:
            return jsonify({
                'success': False,
                'error': 'No metrics data provided'
            })
        
        metrics = data.get('metrics', {})
        
        # Validate required metrics
        required_fields = ['Age', 'BMI', 'Glucose', 'BloodPressure']
        for field in required_fields:
            if field not in metrics or metrics[field] is None:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                })
        
        # Ensure features are in the correct order as expected by the model
        ordered_metrics = {
            'Pregnancies': metrics.get('Pregnancies', 0),
            'Glucose': metrics.get('Glucose', 0),
            'BloodPressure': metrics.get('BloodPressure', 0),
            'SkinThickness': metrics.get('SkinThickness', 0),
            'Insulin': metrics.get('Insulin', 0),
            'BMI': metrics.get('BMI', 0),
            'DiabetesPedigreeFunction': metrics.get('DiabetesPedigreeFunction', 0.5),
            'Age': metrics.get('Age', 0)
        }
        
        result = assistant.predict_diabetes_risk(ordered_metrics)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
