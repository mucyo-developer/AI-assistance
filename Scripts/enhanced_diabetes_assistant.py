from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import re
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'enhanced_diabetes_assistant_2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///diabetes_assistant.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    emergency_contacts = db.relationship('EmergencyContact', backref='user', lazy=True)
    emergencies = db.relationship('EmergencyAlert', backref='user', lazy=True)

# Admin model (extends User with admin-specific functionality)
class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    department = db.Column(db.String(100))
    license_number = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship back to User
    user = db.relationship('User', backref=db.backref('admin_profile', uselist=False))

# Conversation model
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Emergency Contact model
class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    relationship = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120))
    is_primary = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Emergency Alert model
class EmergencyAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # 'low_blood_sugar', 'high_blood_sugar', 'symptoms', 'distress'
    severity = db.Column(db.String(20), nullable=False)  # 'low', 'medium', 'high', 'critical'
    description = db.Column(db.Text, nullable=False)
    patient_message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'notified', 'resolved'
    doctor_notified = db.Column(db.Boolean, default=False)
    contacts_notified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class EnhancedDiabetesAssistant:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load the diabetes prediction model
        self.load_prediction_model()
        
        print("✅ Enhanced Diabetes Assistant initialized")
    
    def detect_emergency(self, user_input: str, user_id: int = None) -> dict:
        """Detect emergency situations from user input"""
        user_input_lower = user_input.lower()
        
        # Emergency keywords and patterns
        emergency_patterns = {
            'critical': [
                'call 911', 'emergency', 'ambulance', 'hospital now', 'dying', 'cant breathe',
                'chest pain', 'severe pain', 'unconscious', 'fainted', 'passed out'
            ],
            'high': [
                'very high blood sugar', 'blood sugar over 400', 'ketones', 'diabetic ketoacidosis',
                'dka', 'vomiting', 'nausea', 'confusion', 'blurred vision severe'
            ],
            'medium': [
                'blood sugar over 300', 'very thirsty', 'frequent urination', 'feeling very sick',
                'worried', 'scared', 'anxious', 'panic', 'help me'
            ],
            'low': [
                'blood sugar under 50', 'hypoglycemia severe', 'shaking badly', 'confused',
                'sweating', 'dizzy', 'weak', 'headache severe'
            ]
        }
        
        # Check for emergency patterns
        detected_severity = None
        alert_type = None
        
        for severity, patterns in emergency_patterns.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    detected_severity = severity
                    if 'blood sugar' in pattern or 'glucose' in pattern:
                        alert_type = 'blood_sugar'
                    elif any(word in pattern for word in ['pain', 'breathe', 'unconscious']):
                        alert_type = 'medical_emergency'
                    else:
                        alert_type = 'symptoms'
                    break
            if detected_severity:
                break
        
        # Check blood sugar values if mentioned
        glucose_match = re.search(r'blood sugar.*?(\d+)|glucose.*?(\d+)', user_input_lower)
        if glucose_match:
            glucose = int(glucose_match.group(1) or glucose_match.group(2))
            if glucose < 40:
                detected_severity = 'critical'
                alert_type = 'low_blood_sugar'
            elif glucose < 60:
                detected_severity = 'high'
                alert_type = 'low_blood_sugar'
            elif glucose > 500:
                detected_severity = 'critical'
                alert_type = 'high_blood_sugar'
            elif glucose > 350:
                detected_severity = 'high'
                alert_type = 'high_blood_sugar'
            elif glucose > 250:
                detected_severity = 'medium'
                alert_type = 'high_blood_sugar'
        
        if detected_severity:
            return {
                'emergency_detected': True,
                'severity': detected_severity,
                'alert_type': alert_type,
                'user_input': user_input
            }
        
        return {'emergency_detected': False}
    
    def create_emergency_alert(self, user_id: int, emergency_data: dict) -> int:
        """Create emergency alert in database"""
        try:
            alert = EmergencyAlert(
                user_id=user_id,
                alert_type=emergency_data['alert_type'],
                severity=emergency_data['severity'],
                description=f"Emergency detected: {emergency_data['alert_type']} - {emergency_data['severity']} severity",
                patient_message=emergency_data['user_input']
            )
            
            db.session.add(alert)
            db.session.commit()
            
            # Trigger notifications
            self.send_emergency_notifications(alert.id)
            
            return alert.id
            
        except Exception as e:
            print(f"Error creating emergency alert: {e}")
            return None
    
    def send_emergency_notifications(self, alert_id: int):
        """Send emergency notifications to doctor and contacts"""
        try:
            alert = EmergencyAlert.query.get(alert_id)
            if not alert:
                return
            
            user = User.query.get(alert.user_id)
            contacts = EmergencyContact.query.filter_by(user_id=alert.user_id).all()
            
            # In a real implementation, this would send actual notifications
            # For demo purposes, we'll just log the notifications
            
            print(f"🚨 EMERGENCY ALERT - Patient: {user.username}")
            print(f"📞 Alert Type: {alert.alert_type}")
            print(f"⚠️ Severity: {alert.severity}")
            print(f"💬 Patient Message: {alert.patient_message}")
            print(f"🕐 Time: {alert.created_at}")
            
            # Notify doctor (in real implementation, this would send email/SMS)
            print("👨‍⚕️ DOCTOR NOTIFICATION SENT")
            alert.doctor_notified = True
            
            # Notify emergency contacts
            if contacts:
                print("👥 EMERGENCY CONTACTS NOTIFIED:")
                for contact in contacts:
                    print(f"   - {contact.name} ({contact.relationship}): {contact.phone}")
                alert.contacts_notified = True
            
            db.session.commit()
            
        except Exception as e:
            print(f"Error sending emergency notifications: {e}")
    
    def get_emergency_response(self, emergency_data: dict) -> str:
        """Generate appropriate emergency response"""
        severity = emergency_data['severity']
        
        if severity == 'critical':
            return """🚨 **MEDICAL EMERGENCY DETECTED**

I've immediately notified your healthcare provider and emergency contacts.

**PLEASE TAKE ACTION NOW:**
• Call emergency services (911) immediately
• If you have someone with you, tell them you need help
• If you're alone, try to stay on the phone with emergency services

**DO NOT WAIT** - Your symptoms require immediate medical attention.

I'm staying with you here. Can you tell me if you're able to call for help right now?"""
        
        elif severity == 'high':
            return """⚠️ **HIGH PRIORITY ALERT**

I've notified your healthcare provider about your situation.

**IMMEDIATE STEPS:**
• Contact your doctor right away
• If you can't reach your doctor, consider urgent care
• Have someone stay with you if possible
• Monitor your symptoms closely

**If symptoms worsen, call emergency services immediately.**

I'm here with you. How are you feeling right now? Are you able to contact someone for help?"""
        
        elif severity == 'medium':
            return """🔶 **ATTENTION NEEDED**

I've alerted your healthcare provider to check in on you.

**RECOMMENDED ACTIONS:**
• Contact your doctor within the next few hours
• Monitor your symptoms closely
• Have someone available if possible
• Rest and stay hydrated

**If you start to feel worse, don't wait - seek medical help.**

I'm concerned about you. Can you tell me more about how you're feeling?"""
        
        else:
            return """🔵 **HEALTH ADVISORY**

I've noted your concern and your healthcare provider will be informed.

**SUGGESTED ACTIONS:**
• Monitor your symptoms
• Contact your doctor if things don't improve
• Follow your usual diabetes management plan
• Reach out if you need support

I'm here to help. What else can I tell you about managing this situation?"""
    
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
        """Get conversational diabetes information based on topic"""
        topic_lower = topic.lower()
        
        # Check for emotional states and general conversation
        if any(word in topic_lower for word in ['feel', 'feeling', 'sad', 'happy', 'tired', 'worried', 'anxious', 'stressed', 'okay', 'fine', 'not okay']):
            return self.get_emotional_response(topic_lower)
        
        # Check for greetings
        if any(word in topic_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return self.get_greeting_response()
        
        # Check for diabetes information requests
        if 'diabetes' in topic_lower or any(word in topic_lower for word in ['what is', 'tell me about', 'explain']):
            return """I'd be happy to help you understand diabetes better. 

Diabetes is a condition that affects how your body handles sugar in your blood. Think of it like this: your body needs insulin (a hormone) to help sugar get from your blood into your cells for energy. When you have diabetes, either your body doesn't make enough insulin, or it can't use the insulin properly.

There are a few different types:
- Type 1 usually starts when people are young and their body stops making insulin
- Type 2 is more common and often develops in adults - the body either doesn't use insulin well or doesn't make enough
- Some women get gestational diabetes during pregnancy

How are you feeling about this? Are you concerned about diabetes for yourself or someone you care about? I'm here to listen and help you understand better."""
        
        elif any(word in topic_lower for word in ['symptom', 'sign']):
            return """That's a really important question. The signs of diabetes can sometimes be subtle, so it's good you're asking about them.

Common symptoms include:
- Needing to urinate more often, especially at night
- Feeling very thirsty all the time
- Being hungrier than usual but still losing weight
- Feeling tired and weak
- Having blurry vision

Some people also notice:
- Cuts or sores that take a long time to heal
- Getting more infections than usual
- Tingling or numbness in their hands or feet

The tricky thing is that some people with type 2 diabetes don't notice any symptoms for years. That's why regular check-ups are so important.

Have you been experiencing any of these symptoms? It's always best to talk with a healthcare provider if you're concerned."""
        
        elif any(word in topic_lower for word in ['prevent', 'avoid', 'reduce risk']):
            return """I'm glad you're thinking about prevention - that's really smart! There's actually a lot we can do to lower our risk of diabetes.

The most powerful things are:
- Keeping a healthy weight (even losing just 5-7% of your weight makes a big difference)
- Moving your body regularly (aim for about 30 minutes most days)
- Eating more vegetables, fruits, and whole grains
- Cutting back on sugary drinks and processed foods

Small changes really add up! Maybe start with one thing that feels manageable - like taking a 10-minute walk each day, or swapping soda for water.

What feels most doable for you right now? Everyone's journey is different, and I'm here to support you in finding what works for your lifestyle."""
        
        elif any(word in topic_lower for word in ['diet', 'food', 'eat', 'nutrition']):
            return """Food can feel overwhelming when you're thinking about diabetes, but it doesn't have to be complicated!

The good news is that the same foods that are good for diabetes prevention are good for overall health too. Focus on:
- Lots of colorful vegetables (the more colors, the better!)
- Lean proteins like chicken, fish, beans, or tofu
- Whole grains like brown rice, quinoa, or whole wheat bread
- Healthy fats from nuts, avocado, or olive oil

Try to limit:
- Sugary drinks (soda, sweetened juices, specialty coffees)
- White bread, white rice, and pasta
- Fried foods and processed snacks

The key is balance, not perfection. What's your relationship with food like? Are there specific meals or foods you're wondering about?"""
        
        elif any(word in topic_lower for word in ['exercise', 'activity', 'fitness']):
            return """Exercise is such a powerful tool for managing and preventing diabetes! The best part is that you don't have to become a marathon runner - regular movement makes a huge difference.

Aim for about 150 minutes per week (that's just 30 minutes, 5 days a week). But here's the secret: it all counts! 
- Walking is fantastic and free
- Dancing in your living room counts
- Gardening counts
- Taking the stairs instead of the elevator counts

The goal is to find something you actually enjoy, because then you'll stick with it. What kind of movement do you like? Or what sounds interesting to try?

Remember to start slow and build up - even 10 minutes a day is a great beginning!"""
        
        else:
            return self.get_conversational_response(topic_lower)
    
    def get_emotional_response(self, topic: str) -> str:
        """Generate empathetic response based on emotional state"""
        if any(word in topic for word in ['sad', 'down', 'depressed']):
            return """I hear that you're feeling sad, and I want you to know that's completely okay. It takes courage to share how you're feeling.

Sometimes dealing with health concerns can feel overwhelming, and it's normal to have moments of sadness. You're not alone in this - many people feel this way when they're worried about their health.

Would you like to talk more about what's on your mind? Sometimes just putting our thoughts into words can help. I'm here to listen without judgment."""
        
        elif any(word in topic for word in ['worried', 'anxious', 'scared', 'afraid']):
            return """It sounds like you're feeling worried, and that makes so much sense. Health concerns can be really scary, and it's completely normal to feel anxious.

Let's take a deep breath together. Whatever you're facing, you don't have to figure it all out at once. We can tackle this one small step at a time.

What's worrying you the most right now? Sometimes when we name our biggest fear, it becomes a little less overwhelming."""
        
        elif any(word in topic for word in ['tired', 'exhausted', 'fatigued']):
            return """I can hear that you're feeling tired. That feeling of exhaustion can be so overwhelming, can't it?

Sometimes when we're dealing with health stuff, the emotional weight alone can make us feel physically drained. Your body and mind might be working overtime processing everything.

Have you been able to get some rest? Sometimes even just sitting quietly for a few minutes can help recharge your batteries a little bit."""
        
        elif any(word in topic for word in ['stressed', 'overwhelmed']):
            return """Feeling stressed and overwhelmed is completely understandable. Health information can feel like drinking from a fire hose sometimes!

Let's slow down and focus on just one thing at a time. You don't need to solve everything today. What feels like the most pressing concern for you right now?

We can start there and take it step by step. You're doing the right thing by reaching out and seeking information."""
        
        elif any(word in topic for word in ['happy', 'good', 'great']):
            return """That's wonderful to hear! I'm so glad you're feeling good today.

Positive moments are worth celebrating, especially when you're navigating health concerns. What's been going well for you?

Sometimes when we're dealing with health stuff, it's easy to focus on the challenges. I'd love to hear about what's bringing you joy today."""
        
        elif any(word in topic for word in ['not okay', 'bad', 'terrible']):
            return """I'm really sorry to hear that you're not doing well. Thank you for being honest about how you're feeling.

Whatever you're going through right now, you don't have to face it alone. I'm here to listen and support you through this.

Would it help to talk about what's been difficult lately? Sometimes sharing our burdens can make them feel a little lighter."""
        
        else:
            return """Thank you for sharing how you're feeling. It takes courage to be open about our emotional state, especially when we're dealing with health concerns.

How can I best support you right now? Are you looking for information, or would it help more to just talk through what's on your mind?"""
    
    def get_greeting_response(self) -> str:
        """Generate natural greeting response"""
        greetings = [
            """Hello! I'm so glad you're here today. How are you feeling?""",
            """Hi there! It's good to connect with you. What's on your mind today?""",
            """Hello! I'm here to help with any diabetes questions or concerns you might have. How can I support you today?""",
            """Hi! Thanks for reaching out. How are you doing today?"""
        ]
        import random
        return random.choice(greetings)
    
    def get_conversational_response(self, topic: str) -> str:
        """Generate natural conversational response for general queries"""
        if any(word in topic for word in ['help', 'support', 'assist']):
            return """I'm definitely here to help! Whether you have specific questions about diabetes, or you just want to talk through some health concerns, I'm here for you.

What feels most important for you to discuss right now? There's no question too small or concern too big - we can start wherever you're at."""
        
        elif any(word in topic for word in ['thank', 'thanks']):
            return """You're so welcome! I'm really glad I could help. Is there anything else you'd like to know or discuss?"""
        
        elif any(word in topic for word in ['bye', 'goodbye', 'see you']):
            return """Take care of yourself! Remember that I'm here whenever you need support or have questions. Be well!"""
        
        elif len(topic.split()) < 3:
            return """I'm here and ready to help! Feel free to ask me anything about diabetes, share how you're feeling, or tell me what's on your mind.

What would be most helpful for you right now?"""
        
        else:
            return """That's a great question. I want to make sure I give you the most helpful information possible.

Could you tell me a bit more about what you're curious about? For example, are you asking about:
- Understanding diabetes better
- Symptoms to watch for
- How to prevent or manage it
- Diet and exercise recommendations
- Something else entirely

The more I know about what you're looking for, the better I can support you!"""
    
    def get_user_database_info(self, query: str) -> str:
        """Get information about users in the database"""
        try:
            users = User.query.filter_by(is_admin=False).all()
            
            if not users:
                return "There are currently no registered patients in the database."
            
            total_users = len(users)
            active_emergencies = EmergencyAlert.query.filter_by(status='pending').count()
            total_emergencies = EmergencyAlert.query.count()
            
            response = f"""📊 **Patient Database Overview**

**Total Registered Patients:** {total_users}

**Emergency Status:**
• Active Emergencies: {active_emergencies}
• Total Emergencies: {total_emergencies}

**Recent Patient Activity:**
"""
            
            # Get recent users
            recent_users = sorted(users, key=lambda u: u.created_at, reverse=True)[:5]
            for user in recent_users:
                user_emergencies = EmergencyAlert.query.filter_by(user_id=user.id).count()
                user_conversations = Conversation.query.filter_by(user_id=user.id).count()
                response += f"""
• {user.username}
  - Registered: {user.created_at.strftime('%Y-%m-%d')}
  - Conversations: {user_conversations}
  - Emergencies: {user_emergencies}
"""
            
            response += f"""
**Patients with Active Emergencies:**
"""
            
            users_with_emergencies = []
            for alert in EmergencyAlert.query.filter_by(status='pending').all():
                user = User.query.get(alert.user_id)
                if user and not user.is_admin:
                    users_with_emergencies.append({
                        'username': user.username,
                        'severity': alert.severity,
                        'alert_type': alert.alert_type,
                        'time': alert.created_at.strftime('%Y-%m-%d %H:%M')
                    })
            
            if users_with_emergencies:
                for user_info in users_with_emergencies:
                    response += f"""
⚠️ {user_info['username']} - {user_info['severity']} severity ({user_info['alert_type']}) at {user_info['time']}
"""
            else:
                response += "✅ No active emergencies"
            
            return response
            
        except Exception as e:
            return f"Error accessing user database: {str(e)}"
    
    def get_user_status_summary(self) -> str:
        """Get summary of how users are doing"""
        try:
            users = User.query.filter_by(is_admin=False).all()
            
            if not users:
                return "There are no registered patients to check status for."
            
            response = """🏥 **Patient Status Summary**

"""
            
            for user in users:
                user_emergencies = EmergencyAlert.query.filter_by(user_id=user.id).order_by(EmergencyAlert.created_at.desc()).limit(3).all()
                user_conversations = Conversation.query.filter_by(user_id=user.id).order_by(Conversation.timestamp.desc()).limit(3).all()
                
                response += f"""
**Patient: {user.username}**
• Email: {user.email}
• Registered: {user.created_at.strftime('%Y-%m-%d')}
• Total Conversations: {Conversation.query.filter_by(user_id=user.id).count()}
• Total Emergencies: {EmergencyAlert.query.filter_by(user_id=user.id).count()}
"""
                
                # Latest emergency status
                if user_emergencies:
                    latest_emergency = user_emergencies[0]
                    status = "⚠️ Active" if latest_emergency.status == 'pending' else "✅ Resolved"
                    response += f"• Latest Emergency: {latest_emergency.severity} severity - {status}\n"
                
                # Recent conversation topic
                if user_conversations:
                    latest_conv = user_conversations[0]
                    response += f"• Last Conversation: {latest_conv.message[:50]}...\n"
                
                response += "---\n"
            
            return response
            
        except Exception as e:
            return f"Error getting user status: {str(e)}"
    
    def get_emergency_summary(self) -> str:
        """Get summary of emergencies"""
        try:
            emergencies = EmergencyAlert.query.order_by(EmergencyAlert.created_at.desc()).limit(10).all()
            
            if not emergencies:
                return "No emergencies recorded in the system."
            
            active_count = EmergencyAlert.query.filter_by(status='pending').count()
            resolved_count = EmergencyAlert.query.filter_by(status='resolved').count()
            
            response = f"""🚨 **Emergency Summary**

**Statistics:**
• Active Emergencies: {active_count}
• Resolved Emergencies: {resolved_count}
• Total Emergencies: {len(emergencies)}

**Recent Emergencies:**
"""
            
            for emergency in emergencies[:5]:
                user = User.query.get(emergency.user_id)
                if user:
                    status_icon = "⚠️" if emergency.status == 'pending' else "✅"
                    response += f"""
{status_icon} **{user.username}** - {emergency.severity.upper()} {emergency.alert_type.replace('_', ' ')}
• Time: {emergency.created_at.strftime('%Y-%m-%d %H:%M:%S')}
• Status: {emergency.status.upper()}
• Patient Message: {emergency.patient_message[:80]}...
"""
                    response += "---\n"
            
            return response
            
        except Exception as e:
            return f"Error getting emergency summary: {str(e)}"
    
    def get_conversation_summary(self) -> str:
        """Get summary of recent conversations"""
        try:
            conversations = Conversation.query.order_by(Conversation.timestamp.desc()).limit(15).all()
            
            if not conversations:
                return "No conversations recorded in the system."
            
            response = """💬 **Recent Patient Conversations**

"""
            
            for conv in conversations:
                user = User.query.get(conv.user_id)
                if user:
                    response += f"""
**Patient: {user.username}**
• Message: {conv.message[:100]}...
• Time: {conv.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
• Response: {conv.response[:100]}...
"""
                    response += "---\n"
            
            return response
            
        except Exception as e:
            return f"Error getting conversation summary: {str(e)}"
    
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
assistant = EnhancedDiabetesAssistant()

# Routes
@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Admin routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username, is_admin=True).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    # Get statistics
    total_patients = User.query.filter_by(is_admin=False).count()
    total_emergencies = EmergencyAlert.query.count()
    active_emergencies = EmergencyAlert.query.filter_by(status='pending').count()
    recent_conversations = Conversation.query.order_by(Conversation.timestamp.desc()).limit(10).all()
    
    return render_template('admin_dashboard.html', 
                         total_patients=total_patients,
                         total_emergencies=total_emergencies,
                         active_emergencies=active_emergencies,
                         recent_conversations=recent_conversations)

@app.route('/admin/patients')
@login_required
def admin_patients():
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    patients = User.query.filter_by(is_admin=False).all()
    return render_template('admin_patients.html', patients=patients)

@app.route('/admin/patient/<int:patient_id>')
@login_required
def admin_patient_details(patient_id):
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    patient = User.query.get_or_404(patient_id)
    if patient.is_admin:
        return redirect(url_for('admin_patients'))
    
    conversations = Conversation.query.filter_by(user_id=patient_id).order_by(Conversation.timestamp.desc()).limit(50).all()
    emergency_contacts = EmergencyContact.query.filter_by(user_id=patient_id).all()
    emergencies = EmergencyAlert.query.filter_by(user_id=patient_id).order_by(EmergencyAlert.created_at.desc()).limit(20).all()
    
    return render_template('admin_patient_details.html', 
                         patient=patient,
                         conversations=conversations,
                         emergency_contacts=emergency_contacts,
                         emergencies=emergencies)

@app.route('/admin/emergencies')
@login_required
def admin_emergencies():
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    emergencies = EmergencyAlert.query.order_by(EmergencyAlert.created_at.desc()).all()
    return render_template('admin_emergencies.html', emergencies=emergencies)

@app.route('/admin/emergency/<int:emergency_id>/resolve', methods=['POST'])
@login_required
def admin_resolve_emergency(emergency_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    emergency = EmergencyAlert.query.get_or_404(emergency_id)
    emergency.status = 'resolved'
    emergency.resolved_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/admin/chat', methods=['POST'])
@login_required
def admin_chat():
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
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
        
        # Check for database queries about users
        user_input_lower = user_input.lower()
        
        # Handle user database queries
        if any(word in user_input_lower for word in ['users', 'patients', 'how many users', 'how many patients', 'list users', 'list patients', 'what users', 'who are our users']):
            response = assistant.get_user_database_info(user_input_lower)
        # Handle specific user status queries
        elif any(word in user_input_lower for word in ['how are they', 'how are users', 'user status', 'patient status', 'how are patients']):
            response = assistant.get_user_status_summary()
        # Handle emergency queries
        elif any(word in user_input_lower for word in ['emergencies', 'emergency alerts', 'active emergencies', 'recent emergencies']):
            response = assistant.get_emergency_summary()
        # Handle conversation queries
        elif any(word in user_input_lower for word in ['conversations', 'chats', 'messages', 'what are users talking about']):
            response = assistant.get_conversation_summary()
        else:
            # Admin gets the same diabetes information but with admin context
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

@app.route('/api/admin/emergency-alerts')
@login_required
def api_admin_emergency_alerts():
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    try:
        emergencies = EmergencyAlert.query.order_by(EmergencyAlert.created_at.desc()).limit(50).all()
        
        alert_list = []
        for alert in emergencies:
            patient = User.query.get(alert.user_id)
            alert_list.append({
                'id': alert.id,
                'patient_name': patient.username,
                'patient_id': patient.id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'description': alert.description,
                'patient_message': alert.patient_message,
                'status': alert.status,
                'doctor_notified': alert.doctor_notified,
                'contacts_notified': alert.contacts_notified,
                'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'resolved_at': alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if alert.resolved_at else None
            })
        
        return jsonify({
            'success': True,
            'alerts': alert_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/admin/patients')
@login_required
def api_admin_patients():
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    try:
        patients = User.query.filter_by(is_admin=False).all()
        
        patient_list = []
        for patient in patients:
            # Get latest emergency for this patient
            latest_emergency = EmergencyAlert.query.filter_by(user_id=patient.id).order_by(EmergencyAlert.created_at.desc()).first()
            
            patient_list.append({
                'id': patient.id,
                'username': patient.username,
                'email': patient.email,
                'created_at': patient.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'last_activity': latest_emergency.created_at.strftime('%Y-%m-%d %H:%M:%S') if latest_emergency else None,
                'emergency_count': EmergencyAlert.query.filter_by(user_id=patient.id).count(),
                'active_emergency': EmergencyAlert.query.filter_by(user_id=patient.id, status='pending').count() > 0
            })
        
        return jsonify({
            'success': True,
            'patients': patient_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/chat', methods=['POST'])
@login_required
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
        
        # Check for emergency situations first
        emergency_data = assistant.detect_emergency(user_input, current_user.id)
        if emergency_data['emergency_detected']:
            # Create emergency alert
            alert_id = assistant.create_emergency_alert(current_user.id, emergency_data)
            response = assistant.get_emergency_response(emergency_data)
            
            # Save conversation to database
            conversation = Conversation(
                user_id=current_user.id,
                message=user_input,
                response=response
            )
            db.session.add(conversation)
            db.session.commit()
            
            return jsonify({
                'response': response,
                'success': True,
                'emergency': True,
                'alert_id': alert_id,
                'severity': emergency_data['severity']
            })
        
        # Check for blood sugar reading (non-emergency)
        glucose_match = re.search(r'blood sugar.*?(\d+)|glucose.*?(\d+)', user_input.lower())
        if glucose_match:
            glucose = int(glucose_match.group(1) or glucose_match.group(2))
            response = assistant.get_blood_sugar_advice(glucose)
        else:
            # Get diabetes information
            response = assistant.get_diabetes_info(user_input)
        
        # Save conversation to database
        conversation = Conversation(
            user_id=current_user.id,
            message=user_input,
            response=response
        )
        db.session.add(conversation)
        db.session.commit()
        
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
@login_required
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

@app.route('/api/diabetes-stats')
def get_diabetes_stats():
    """Get real-time diabetes statistics for the graph"""
    try:
        # Simulate real-time data (in production, this would come from a real API)
        years = list(range(2000, 2024))
        
        # Simulate diabetes prevalence data (percentage of world population)
        # These are realistic estimates based on WHO data trends
        prevalence_data = [
            4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4,
            6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 8.0, 8.4, 8.8, 9.2,
            9.6, 10.0, 10.4, 10.8
        ]
        
        # Create the graph data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=prevalence_data,
            mode='lines+markers',
            name='Diabetes Prevalence (%)',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Global Diabetes Prevalence (2000-2023)',
            xaxis_title='Year',
            yaxis_title='Prevalence (%)',
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Convert to JSON
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'graph': graph_json,
            'current_prevalence': prevalence_data[-1],
            'total_adults_affected': 537,  # Millions (2023 estimate)
            'projected_2030': 643,  # Millions (WHO projection)
            'projected_2045': 783   # Millions (WHO projection)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/user-conversations')
@login_required
def get_user_conversations():
    """Get user's conversation history"""
    try:
        conversations = Conversation.query.filter_by(user_id=current_user.id)\
            .order_by(Conversation.timestamp.desc()).limit(50).all()
        
        conversation_list = []
        for conv in conversations:
            conversation_list.append({
                'id': conv.id,
                'message': conv.message,
                'response': conv.response,
                'timestamp': conv.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({
            'success': True,
            'conversations': conversation_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/emergency-contacts', methods=['GET', 'POST'])
@login_required
def manage_emergency_contacts():
    """Manage emergency contacts"""
    if request.method == 'GET':
        try:
            contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
            contact_list = []
            for contact in contacts:
                contact_list.append({
                    'id': contact.id,
                    'name': contact.name,
                    'relationship': contact.relationship,
                    'phone': contact.phone,
                    'email': contact.email,
                    'is_primary': contact.is_primary
                })
            
            return jsonify({
                'success': True,
                'contacts': contact_list
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    elif request.method == 'POST':
        try:
            data = request.json
            contact = EmergencyContact(
                user_id=current_user.id,
                name=data['name'],
                relationship=data['relationship'],
                phone=data['phone'],
                email=data.get('email', ''),
                is_primary=data.get('is_primary', False)
            )
            
            # If this is primary, unmark others
            if contact.is_primary:
                EmergencyContact.query.filter_by(user_id=current_user.id, is_primary=True).update({'is_primary': False})
            
            db.session.add(contact)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'contact_id': contact.id
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })

@app.route('/api/emergency-contacts/<int:contact_id>', methods=['DELETE'])
@login_required
def delete_emergency_contact(contact_id):
    """Delete emergency contact"""
    try:
        contact = EmergencyContact.query.get(contact_id)
        if not contact or contact.user_id != current_user.id:
            return jsonify({
                'success': False,
                'error': 'Contact not found'
            })
        
        db.session.delete(contact)
        db.session.commit()
        
        return jsonify({
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/emergency-alerts')
@login_required
def get_emergency_alerts():
    """Get user's emergency alerts"""
    try:
        alerts = EmergencyAlert.query.filter_by(user_id=current_user.id)\
            .order_by(EmergencyAlert.created_at.desc()).limit(20).all()
        
        alert_list = []
        for alert in alerts:
            alert_list.append({
                'id': alert.id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'description': alert.description,
                'patient_message': alert.patient_message,
                'status': alert.status,
                'doctor_notified': alert.doctor_notified,
                'contacts_notified': alert.contacts_notified,
                'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'resolved_at': alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if alert.resolved_at else None
            })
        
        return jsonify({
            'success': True,
            'alerts': alert_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5001)
