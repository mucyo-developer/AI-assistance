"""
Setup script to create initial admin user for the Enhanced Diabetes Assistant
"""
from enhanced_diabetes_assistant import app, db, User
from werkzeug.security import generate_password_hash

def create_admin_user():
    """Create initial admin user"""
    with app.app_context():
        # Create all database tables
        db.create_all()
        print("Database tables created successfully")
        
        # Check if admin already exists
        admin = User.query.filter_by(username='medical_admin').first()
        if admin:
            print("Admin user already exists!")
            print(f"Username: medical_admin")
            return
        
        # Create admin user
        admin = User(
            username='medical_admin',
            email='admin@diabetes-assistant.com',
            password_hash=generate_password_hash('HealthCare@2024'),
            is_admin=True
        )
        
        db.session.add(admin)
        db.session.commit()
        
        print("✅ Admin user created successfully!")
        print("Login credentials:")
        print("  Username: medical_admin")
        print("  Password: HealthCare@2024")
        print("  Admin URL: http://localhost:5001/admin/login")
        print("\n⚠️  Please change the admin password after first login!")

if __name__ == '__main__':
    create_admin_user()
