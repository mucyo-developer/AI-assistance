# Enhanced Diabetes Assistant

An AI-powered diabetes management system with patient monitoring, emergency detection, and admin dashboard capabilities.

## Features

- **Patient Dashboard**: Blood sugar tracking, risk assessment, AI health assistant
- **Emergency Detection**: Automatic detection of emergency keywords with alert system
- **Admin Portal**: Healthcare provider dashboard for patient monitoring and management
- **AI Database Assistant**: Admin can query patient database using natural language
- **Real-time Monitoring**: 24/7 emergency alert system
- **Secure Authentication**: Separate login systems for patients and administrators

## Tech Stack

- **Backend**: Flask, Flask-SQLAlchemy, Flask-Login
- **Frontend**: Bootstrap 5, Font Awesome, Plotly
- **AI/ML**: Scikit-learn, Ollama (LLM integration)
- **Database**: SQLite (production: PostgreSQL recommended)

## Local Development

### Prerequisites

- Python 3.12+
- pip

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Initialize the database:
   ```bash
   cd Scripts
   python setup_admin.py
   ```

5. Run the application:
   ```bash
   python enhanced_diabetes_assistant.py
   ```

6. Access the application:
   - Main page: http://localhost:5001/
   - Patient login: http://localhost:5001/login
   - Admin login: http://localhost:5001/admin/login

### Default Admin Credentials

- **Username**: medical_admin
- **Password**: HealthCare@2024

⚠️ **Important**: Change the admin password after first login!

## Deployment

### Vercel Deployment

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect the Python app

3. **Set Environment Variables in Vercel**
   - Go to Project Settings → Environment Variables
   - Add the following:
     ```
     SECRET_KEY=your_secure_secret_key
     DATABASE_URL=postgresql://user:password@host:port/database
     FLASK_ENV=production
     DEBUG=False
     ```

4. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy your application

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| SECRET_KEY | Flask secret key | (generated) |
| DATABASE_URL | Database connection string | sqlite:///diabetes_assistant.db |
| FLASK_ENV | Environment mode | production |
| DEBUG | Debug mode | False |
| OLLAMA_HOST | Ollama server URL | http://localhost:11434 |
| OLLAMA_MODEL | Ollama model name | llama3.2:latest |

## Project Structure

```
task/
├── api/
│   └── index.py              # Vercel entry point
├── Scripts/
│   ├── enhanced_diabetes_assistant.py  # Main Flask application
│   ├── setup_admin.py        # Admin user setup script
│   └── templates/            # HTML templates
│       ├── enhanced_index.html
│       ├── admin_dashboard.html
│       ├── dashboard.html
│       └── ...
├── var/                      # Database directory
├── Procfile                  # Production process file
├── vercel.json              # Vercel configuration
├── requirements.txt         # Python dependencies
└── .env.example            # Environment variables template
```

## API Endpoints

### Patient Endpoints
- `GET /` - Main landing page
- `GET /login` - Patient login page
- `POST /login` - Patient login handler
- `GET /register` - Patient registration page
- `POST /register` - Patient registration handler
- `GET /dashboard` - Patient dashboard (requires login)
- `POST /chat` - AI chat endpoint
- `POST /predict` - Diabetes risk prediction

### Admin Endpoints
- `GET /admin/login` - Admin login page
- `POST /admin/login` - Admin login handler
- `GET /admin/dashboard` - Admin dashboard (requires admin)
- `GET /admin/patients` - Patient list
- `GET /admin/emergencies` - Emergency alerts
- `POST /admin/chat` - Admin AI chat with database queries

### API Endpoints
- `GET /api/emergency-contacts` - Get emergency contacts
- `POST /api/emergency-contacts` - Add emergency contact
- `DELETE /api/emergency-contacts/<id>` - Delete emergency contact
- `GET /api/emergency-alerts` - Get emergency alerts
- `GET /api/admin/emergency-alerts` - Admin emergency alerts

## Admin Dashboard Features

The admin dashboard provides healthcare providers with:

- **Patient Database Overview**: View all registered patients
- **Emergency Monitoring**: Real-time emergency alert tracking
- **AI Database Assistant**: Natural language queries about patient data
- **Patient Management**: View detailed patient information
- **Conversation History**: Monitor patient-AI conversations

### Admin AI Assistant Commands

The admin AI assistant can answer questions like:
- "What users do we have?" - Shows patient database overview
- "How are patients doing?" - Detailed patient status summary
- "Show me recent emergencies" - Emergency summary with details
- "What are users talking about?" - Recent conversation summary

## Security Notes

- Change default admin credentials immediately
- Use strong SECRET_KEY in production
- Use PostgreSQL instead of SQLite for production
- Enable HTTPS in production
- Regularly update dependencies
- Implement rate limiting for API endpoints

## Troubleshooting

### Database Issues
If you encounter database errors:
```bash
# Delete and recreate database
rm var/enhanced_diabetes_assistant-instance/diabetes_assistant.db
python Scripts/setup_admin.py
```

### Port Already in Use
```bash
# Find and kill process using port 5001
# Windows:
netstat -ano | findstr :5001
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5001 | xargs kill -9
```

### Ollama Connection Issues
Ensure Ollama is running:
```bash
# Start Ollama
ollama serve

# Pull the model
ollama pull llama3.2:latest
```

## License

This project is for educational purposes. Please ensure compliance with healthcare data regulations in your jurisdiction.

## Support

For issues and questions, please refer to the project documentation or contact the development team.
