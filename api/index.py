import sys
import os

# Add the Scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

# Import the Flask app from enhanced_diabetes_assistant
from enhanced_diabetes_assistant import app as application

# Vercel requires this to be named 'app'
app = application

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
