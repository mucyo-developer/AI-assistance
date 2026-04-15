from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import json
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
import os
import uuid
from app import DiabetesPredictionModel

app = Flask(__name__)
app.secret_key = 'diabetes_assistant_secret_key_2024'
CORS(app)

class WebDiabetesAssistant:
    def __init__(self, model_name="llama3.2:latest"):
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Try to load sentence-transformers with offline fallback
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
            print("SentenceTransformer loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer (offline mode): {e}")
            self.embedding_model = None
            self.use_embeddings = False
        
        # Initialize ChromaDB for conversation memory
        self.chroma_client = chromadb.PersistentClient(path="./diabetes_memory")
        self.conversation_collection = self.chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.knowledge_collection = self.chroma_client.get_or_create_collection(
            name="diabetes_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize diabetes prediction model
        self.prediction_model = DiabetesPredictionModel()
        self.load_prediction_model()
        
        # Initialize knowledge base
        self.initialize_knowledge_base()
        
        # Available tools
        self.tools = {
            "predict_diabetes_risk": self.predict_diabetes_risk,
            "get_blood_sugar_advice": self.get_blood_sugar_advice,
            "get_diet_recommendations": self.get_diet_recommendations,
            "get_exercise_advice": self.get_exercise_advice,
            "get_medication_reminder": self.get_medication_reminder,
            "emergency_guidance": self.emergency_guidance,
            "log_patient_data": self.log_patient_data,
            "get_conversation_history": self.get_conversation_history
        }
        
        print("Web Diabetes Assistant initialized successfully")
    
    def load_prediction_model(self):
        """Load the trained diabetes prediction model"""
        try:
            self.prediction_model.load_model()
            print("Diabetes prediction model loaded successfully")
        except FileNotFoundError:
            print("Training new diabetes prediction model...")
            # Train the model if not available
            X, y = self.prediction_model.load_and_preprocess_data('diabetes.csv')
            X_train, X_test, y_train, y_test = self.prediction_model.split_and_scale_data(X, y)
            self.prediction_model.train_model(X_train, y_train)
            self.prediction_model.save_model()
            print("Model trained and saved successfully")
    
    def initialize_knowledge_base(self):
        """Initialize the diabetes knowledge base"""
        knowledge_base = [
            {
                "content": "Normal blood sugar levels: Fasting 70-100 mg/dL, After eating 70-140 mg/dL. Diabetes diagnosis: Fasting >126 mg/dL or After eating >200 mg/dL.",
                "category": "blood_sugar"
            },
            {
                "content": "Type 1 diabetes is autoimmune, requires insulin. Type 2 is insulin resistance, managed with diet, exercise, and sometimes medication.",
                "category": "diabetes_types"
            },
            {
                "content": "Low blood sugar (hypoglycemia) symptoms: shakiness, sweating, confusion, dizziness. Treatment: consume 15g fast-acting carbs like glucose tablets or juice.",
                "category": "emergency"
            },
            {
                "content": "High blood sugar (hyperglycemia) symptoms: increased thirst, frequent urination, fatigue, blurred vision. Contact doctor if >250 mg/dL for extended time.",
                "category": "emergency"
            },
            {
                "content": "Diabetic diet: Focus on whole grains, lean proteins, vegetables, fruits in moderation. Limit processed foods, sugar, and refined carbs.",
                "category": "diet"
            },
            {
                "content": "Exercise benefits: 30 minutes moderate activity most days improves insulin sensitivity. Check blood sugar before and after exercise.",
                "category": "exercise"
            },
            {
                "content": "Foot care: Check feet daily for cuts, blisters, or sores. Wear proper footwear. See doctor for any foot problems immediately.",
                "category": "complications"
            },
            {
                "content": "Medication timing: Take medications as prescribed. Never skip doses. Keep a medication schedule and set reminders.",
                "category": "medication"
            },
            {
                "content": "Stress management: Stress can raise blood sugar. Practice relaxation techniques, deep breathing, or meditation.",
                "category": "lifestyle"
            },
            {
                "content": "Regular monitoring: Check blood sugar as directed by doctor. Keep a log of readings, meals, exercise, and medications.",
                "category": "monitoring"
            }
        ]
        
        # Clear existing knowledge base
        try:
            self.knowledge_collection.delete()
            self.knowledge_collection = self.chroma_client.get_or_create_collection(
                name="diabetes_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            pass
        
        # Add knowledge to vector database
        for i, knowledge in enumerate(knowledge_base):
            if self.use_embeddings:
                embedding = self.embedding_model.encode(knowledge["content"]).tolist()
                self.knowledge_collection.add(
                    ids=[f"knowledge_{i}"],
                    embeddings=[embedding],
                    documents=[knowledge["content"]],
                    metadatas=[{"category": knowledge["category"]}]
                )
            else:
                # Offline mode: store without embeddings
                self.knowledge_collection.add(
                    ids=[f"knowledge_{i}"],
                    documents=[knowledge["content"]],
                    metadatas=[{"category": knowledge["category"]}]
                )
        
        print("Knowledge base initialized")
    
    def store_conversation(self, user_input: str, assistant_response: str, user_id: str):
        """Store conversation in vector database"""
        timestamp = datetime.datetime.now().isoformat()
        conversation_text = f"User: {user_input}\nAssistant: {assistant_response}"
        
        if self.use_embeddings:
            embedding = self.embedding_model.encode(conversation_text).tolist()
            self.conversation_collection.add(
                ids=[f"conv_{user_id}_{timestamp}"],
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[{"user_id": user_id, "timestamp": timestamp}]
            )
        else:
            # Offline mode: store without embeddings
            self.conversation_collection.add(
                ids=[f"conv_{user_id}_{timestamp}"],
                documents=[conversation_text],
                metadatas=[{"user_id": user_id, "timestamp": timestamp}]
            )
    
    def get_relevant_context(self, query: str, n_results: int = 3) -> List[str]:
        """Get relevant context from knowledge base and conversation history"""
        if self.use_embeddings:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Get relevant knowledge
            knowledge_results = self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Get relevant conversation history
            conversation_results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
            
            context = []
            if knowledge_results['documents']:
                context.extend(knowledge_results['documents'][0])
            if conversation_results['documents']:
                context.extend(conversation_results['documents'][0])
            
            return context
        else:
            # Offline mode: get recent documents without embedding search
            knowledge_results = self.knowledge_collection.get(limit=n_results)
            conversation_results = self.conversation_collection.get(limit=2)
            
            context = []
            if knowledge_results['documents']:
                context.extend(knowledge_results['documents'])
            if conversation_results['documents']:
                context.extend(conversation_results['documents'])
            
            return context
    
    def detect_tool_call(self, user_input: str) -> Dict[str, Any]:
        """Detect if user input requires a tool call"""
        tool_patterns = {
            "predict_diabetes_risk": r"(predict|risk|assess|check).*(diabetes|blood sugar)",
            "get_blood_sugar_advice": r"(blood sugar|glucose|reading).*\d+",
            "get_diet_recommendations": r"(diet|food|eat|nutrition)",
            "get_exercise_advice": r"(exercise|workout|activity|fitness)",
            "get_medication_reminder": r"(medication|medicine|pill|drug)",
            "emergency_guidance": r"(emergency|urgent|help|severe)",
            "log_patient_data": r"(log|record|save|track).*(reading|data|measurement)",
            "get_conversation_history": r"(history|past|previous|remember)"
        }
        
        for tool, pattern in tool_patterns.items():
            if re.search(pattern, user_input.lower()):
                return {"tool": tool, "confidence": "high"}
        
        return {"tool": None, "confidence": "low"}
    
    def call_tool(self, tool_name: str, user_input: str, user_id: str) -> str:
        """Execute the appropriate tool"""
        if tool_name in self.tools:
            return self.tools[tool_name](user_input, user_id)
        else:
            return "I don't have that tool available."
    
    def predict_diabetes_risk(self, user_input: str, user_id: str) -> str:
        """Predict diabetes risk based on user input"""
        try:
            # Extract health metrics from user input
            metrics = self.extract_health_metrics(user_input)
            
            if not metrics:
                return "I need your health metrics to predict diabetes risk. Please provide: age, BMI, glucose level, blood pressure, and family history."
            
            result = self.prediction_model.predict_diabetes_risk(metrics)
            
            response = f"**Diabetes Risk Assessment:**\n"
            response += f"Risk Level: {result['risk_level']}\n"
            response += f"Probability: {result['probability']:.1%}\n\n"
            
            if result['risk_level'] == 'High':
                response += "**High Risk Detected:**\n"
                response += "• Consult a healthcare provider immediately\n"
                response += "• Monitor blood sugar regularly\n"
                response += "• Implement lifestyle changes immediately\n"
                response += "• Consider medication as prescribed by doctor"
            elif result['risk_level'] == 'Medium':
                response += "**Medium Risk:**\n"
                response += "• Schedule a doctor appointment soon\n"
                response += "• Improve diet and exercise habits\n"
                response += "• Monitor blood sugar weekly\n"
                response += "• Maintain healthy weight"
            else:
                response += "**Low Risk:**\n"
                response += "• Continue healthy lifestyle\n"
                response += "• Annual check-ups recommended\n"
                response += "• Maintain balanced diet and regular exercise"
            
            return response
        except Exception as e:
            return f"Error predicting diabetes risk: {str(e)}"
    
    def extract_health_metrics(self, text: str) -> Dict[str, float]:
        """Extract health metrics from user input"""
        metrics = {}
        
        # Extract age
        age_match = re.search(r'age[:\s]+(\d+)', text.lower())
        if age_match:
            metrics['Age'] = float(age_match.group(1))
        
        # Extract BMI
        bmi_match = re.search(r'bmi[:\s]+(\d+\.?\d*)', text.lower())
        if bmi_match:
            metrics['BMI'] = float(bmi_match.group(1))
        
        # Extract glucose
        glucose_match = re.search(r'(glucose|blood sugar)[:\s]+(\d+)', text.lower())
        if glucose_match:
            metrics['Glucose'] = float(glucose_match.group(2))
        
        # Extract blood pressure
        bp_match = re.search(r'(blood pressure|bp)[:\s]+(\d+)', text.lower())
        if bp_match:
            metrics['BloodPressure'] = float(bp_match.group(2))
        
        # Set default values for missing metrics
        defaults = {
            'Pregnancies': 0,
            'SkinThickness': 0,
            'Insulin': 0,
            'DiabetesPedigreeFunction': 0.5
        }
        
        for key, value in defaults.items():
            if key not in metrics:
                metrics[key] = value
        
        # Only return if we have essential metrics
        if 'Age' in metrics and 'BMI' in metrics and 'Glucose' in metrics:
            return metrics
        else:
            return {}
    
    def get_blood_sugar_advice(self, user_input: str, user_id: str) -> str:
        """Provide advice based on blood sugar reading"""
        glucose_match = re.search(r'(\d+)', user_input)
        if not glucose_match:
            return "Please provide your blood sugar reading in mg/dL."
        
        glucose = int(glucose_match.group(1))
        
        if glucose < 70:
            return "**Low Blood Sugar (Hypoglycemia):**\n"
            "• Immediately consume 15-20g fast-acting carbs\n"
            "• Examples: 4 oz juice, regular soda, or glucose tablets\n"
            "• Wait 15 minutes and recheck\n"
            "• If still low, repeat treatment\n"
            "• Seek emergency help if unconscious or seizure occurs"
        elif glucose > 250:
            return "**High Blood Sugar (Hyperglycemia):**\n"
            "• Drink plenty of water\n"
            "• Check for ketones if instructed by doctor\n"
            "• Take prescribed medication\n"
            "• Avoid exercise if >300 mg/dL\n"
            "• Contact doctor if remains high for several hours"
        elif 70 <= glucose <= 140:
            return "**Normal Blood Sugar:**\n"
            "• Great job maintaining healthy levels!\n"
            "• Continue current treatment plan\n"
            "• Monitor at regular intervals as prescribed"
        else:
            return "**Elevated Blood Sugar:**\n"
            "• Monitor closely\n"
            "• Review recent meals and activity\n"
            "• Consider light exercise if approved by doctor\n"
            "• Follow up with healthcare provider"
    
    def get_diet_recommendations(self, user_input: str, user_id: str) -> str:
        """Provide diet recommendations for diabetes management"""
        return "**Diabetes Diet Recommendations:**\n\n"
        "**Foods to Emphasize:**\n"
        "• Non-starchy vegetables (broccoli, spinach, cauliflower)\n"
        "• Lean proteins (chicken, fish, tofu, beans)\n"
        "• Whole grains (quinoa, brown rice, whole wheat bread)\n"
        "• Healthy fats (avocado, nuts, olive oil)\n"
        "• Low-glycemic fruits (berries, apples, citrus)\n\n"
        "**Foods to Limit:**\n"
        "• Sugary beverages and desserts\n"
        "• White bread, rice, and pasta\n"
        "• Fried foods and processed snacks\n"
        "• High-fat dairy products\n\n"
        "**Meal Timing:**\n"
        "• Eat consistent meals at regular times\n"
        "• Don't skip meals\n"
        "• Space meals 4-5 hours apart\n"
        "• Consider smaller, frequent meals"
    
    def get_exercise_advice(self, user_input: str, user_id: str) -> str:
        """Provide exercise recommendations for diabetes management"""
        return "**Exercise Recommendations for Diabetes:**\n\n"
        "**Aerobic Exercise (150 minutes/week):**\n"
        "• Brisk walking, swimming, cycling\n"
        "• 30 minutes, 5 days per week\n"
        "• Start slow and gradually increase intensity\n\n"
        "**Strength Training (2-3 times/week):**\n"
        "• Light weights or resistance bands\n"
        "• Major muscle groups, 2-3 sets of 10-15 reps\n\n"
        "**Safety Guidelines:**\n"
        "• Check blood sugar before and after exercise\n"
        "• Avoid exercise if blood sugar >300 mg/dL\n"
        "• Carry fast-acting carbs during exercise\n"
        "• Stay hydrated\n"
        "• Wear proper footwear\n"
        "• Consult doctor before starting new exercise program"
    
    def get_medication_reminder(self, user_input: str, user_id: str) -> str:
        """Provide medication reminder advice"""
        return "**Medication Management:**\n\n"
        "**Best Practices:**\n"
        "• Take medications at the same time daily\n"
        "• Use pill organizers or reminder apps\n"
        "• Keep a medication schedule\n"
        "• Never skip doses without doctor approval\n\n"
        "**Storage:**\n"
        "• Store at room temperature unless specified\n"
        "• Keep away from moisture and direct sunlight\n"
        "• Check expiration dates regularly\n\n"
        "**Important:**\n"
        "• Inform all healthcare providers about all medications\n"
        "• Report side effects to your doctor\n"
        "• Refill prescriptions before running out\n"
        "• Keep emergency contact information available"
    
    def emergency_guidance(self, user_input: str, user_id: str) -> str:
        """Provide emergency guidance"""
        return "**DIABETES EMERGENCY GUIDANCE:**\n\n"
        "**Call 911 Immediately If:**\n"
        "• Person is unconscious or having seizures\n"
        "• Difficulty breathing\n"
        "• Confusion or disorientation\n"
        "• Blood sugar <50 mg/dL and not responding to treatment\n\n"
        "**While Waiting for Help:**\n"
        "• Check if person is conscious and can swallow\n"
        "• If unconscious, place on side and check breathing\n"
        "• Do not give food or drink to unconscious person\n"
        "• Have glucagon injection available if prescribed\n\n"
        "**Non-Emergency Contact Doctor For:**\n"
        "• Blood sugar >250 mg/dL for several hours\n"
        "• Persistent nausea/vomiting\n"
        "• Signs of infection (fever, urinary issues)\n"
        "• Unexplained weight loss\n\n"
        "**Emergency Kit Should Include:**\n"
        "• Glucose tablets/gel\n"
        "• Glucagon kit\n"
        "• Emergency contact numbers\n"
        "• Medical identification"
    
    def log_patient_data(self, user_input: str, user_id: str) -> str:
        """Log patient data"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Create a simple log entry
        log_entry = {
            "user_id": user_id,
            "timestamp": timestamp,
            "data": user_input
        }
        
        # Store in conversation collection with special metadata
        embedding = self.embedding_model.encode(f"LOG: {user_input}").tolist()
        self.conversation_collection.add(
            ids=[f"log_{user_id}_{timestamp}"],
            embeddings=[embedding],
            documents=[f"Patient Log: {user_input}"],
            metadatas=[{"type": "log", "user_id": user_id, "timestamp": timestamp}]
        )
        
        return "**Data Logged Successfully:**\n"
        f"Your health data has been recorded at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "This information will help track your progress and provide better personalized advice."
    
    def get_conversation_history(self, user_input: str, user_id: str) -> str:
        """Retrieve conversation history"""
        results = self.conversation_collection.query(
            query_embeddings=[self.embedding_model.encode("conversation history").tolist()],
            where={"user_id": user_id},
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No previous conversation history found."
        
        history = "**Recent Conversation History:**\n\n"
        for doc in results['documents'][0]:
            if not doc.startswith("Patient Log:"):
                history += f"{doc}\n\n"
        
        return history
    
    def generate_response(self, user_input: str, user_id: str) -> str:
        """Generate response using Ollama with tool calling"""
        try:
            # Check if tool call is needed
            tool_detection = self.detect_tool_call(user_input)
            
            if tool_detection["tool"]:
                tool_result = self.call_tool(tool_detection["tool"], user_input, user_id)
                
                # Use tool result as context for the model
                context = self.get_relevant_context(user_input)
                context_str = "\n".join(context) if context else ""
                
                prompt = f"""You are a helpful diabetes assistant. A user asked: "{user_input}"

Relevant context:
{context_str}

Tool result:
{tool_result}

Provide a caring, informative response that incorporates the tool result. Be empathetic and encouraging. Always remind them to consult healthcare professionals for medical advice."""
            else:
                # Get relevant context
                context = self.get_relevant_context(user_input)
                context_str = "\n".join(context) if context else ""
                
                prompt = f"""You are a helpful diabetes assistant. A user asked: "{user_input}"

Relevant context from previous conversations and knowledge base:
{context_str}

Provide a caring, informative response. Be empathetic and encouraging. Always remind them to consult healthcare professionals for medical advice."""
            
            print(f"Sending prompt to Ollama: {prompt[:200]}...")
            
            # Try to get response from Ollama
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                assistant_response = response['message']['content']
                print(f"Ollama response received: {assistant_response[:100]}...")
                
            except Exception as ollama_error:
                print(f"Ollama error: {str(ollama_error)}")
                # Fallback response if Ollama fails
                if tool_detection["tool"]:
                    assistant_response = tool_result
                else:
                    assistant_response = "I'm here to help with diabetes management. I can provide information about diet, exercise, blood sugar monitoring, and medication reminders. For specific medical advice, please consult with your healthcare provider. Is there a specific aspect of diabetes care you'd like to know about?"
            
            # Store conversation
            self.store_conversation(user_input, assistant_response, user_id)
            
            return assistant_response
            
        except Exception as e:
            print(f"General error in generate_response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again or consult with a healthcare professional for immediate assistance."

# Initialize the assistant
assistant = WebDiabetesAssistant()

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
                'user_id': session.get('user_id', 'unknown')
            })
        
        user_input = data.get('message', '').strip()
        if not user_input:
            return jsonify({
                'response': 'Please enter a message.',
                'user_id': session.get('user_id', 'unknown')
            })
        
        # Get or create user ID from session
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        user_id = session['user_id']
        
        print(f"Chat request from user {user_id}: {user_input}")
        
        # Generate response
        response = assistant.generate_response(user_input, user_id)
        
        print(f"Generated response: {response[:100]}...")
        
        return jsonify({
            'response': response,
            'user_id': user_id
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Chat error: {error_details}")
        
        # Return a fallback response if there's an error
        fallback_response = "I apologize, but I'm experiencing technical difficulties. Please try again or consult with a healthcare professional for immediate assistance."
        
        return jsonify({
            'response': fallback_response,
            'user_id': session.get('user_id', 'unknown')
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
        
        print(f"Received metrics: {metrics}")
        print(f"Ordered metrics: {ordered_metrics}")
        
        result = assistant.prediction_model.predict_diabetes_risk(ordered_metrics)
        print(f"Prediction result: {result}")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
