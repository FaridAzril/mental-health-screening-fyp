"""
Supabase Authentication Module
"""
from functools import wraps
from flask import session, redirect, url_for, request, make_response
from supabase_config import supabase
import time

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated via Supabase
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(required_role):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            
            # Check user role
            user_profile = get_user_profile(session['user']['id'])
            if not user_profile or user_profile.get('role') != required_role:
                return redirect(url_for('dashboard'))  # Redirect to dashboard with error message
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_user_profile(user_id):
    """Get user profile from database"""
    try:
        response = supabase.table('user_profiles').select('*').eq('id', user_id).execute()
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]
        return None
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return None

def authenticate_user(email, password):
    """Authenticate user with Supabase"""
    try:
        # Sign in with Supabase auth
        auth_response = supabase.auth.sign_in_with_password({
            'email': email,
            'password': password
        })
        
        # Check if authentication was successful
        if auth_response.status_code == 200:
            auth_data = auth_response.json()
            if 'access_token' in auth_data:
                # Store user in session
                session['user'] = {
                    'id': auth_data.get('user', {}).get('id'),
                    'email': email,
                    'access_token': auth_data.get('access_token'),
                    'refresh_token': auth_data.get('refresh_token')
                }
                session['last_activity'] = time.time()
                
                # Get user profile
                user_profile = get_user_profile(session['user']['id'])
                if user_profile:
                    session['user']['role'] = user_profile.get('role')
                    session['user']['name'] = user_profile.get('name')
                
                return True, user_profile
        
        return False, None
            
    except Exception as e:
        print(f"Authentication error: {e}")
        return False, None

def logout_user():
    """Logout user and clear session"""
    try:
        # Clear session
        session.clear()
        
        # Sign out from Supabase
        supabase.auth.sign_out()
        
        return True
    except Exception as e:
        print(f"Logout error: {e}")
        return False

def create_user_profile(user_id, email, role, name):
    """Create user profile in database"""
    try:
        profile_data = {
            'id': user_id,
            'email': email,
            'role': role,
            'name': name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        response = supabase.table('user_profiles').insert(profile_data).execute()
        return response.data[0] if response.data else None
        
    except Exception as e:
        print(f"Error creating user profile: {e}")
        return None

def update_last_activity():
    """Update user's last activity timestamp"""
    if 'user' in session:
        session['last_activity'] = time.time()

def is_session_valid():
    """Check if session is still valid"""
    if 'user' not in session:
        return False
    
    # Check if session is older than 24 hours
    last_activity = session.get('last_activity', 0)
    current_time = time.time()
    
    # 24 hours = 86400 seconds
    if current_time - last_activity > 86400:
        return False
    
    return True
