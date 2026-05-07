#!/usr/bin/env python3
"""
Professional Flask Server for Mental Health Screening Portal
Runs on standard port 5000 with proper HTTP/HTTPS support and security features
"""

from flask import Flask, send_from_directory, render_template, request, session, redirect, url_for, make_response
import webbrowser
import threading
import time
import os
import hashlib
import secrets
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
from supabase_client_working import supabase

# Authentication functions
def get_user_profile(user_id):
    """Get user profile from database"""
    try:
        auth_token = None
        if 'user' in session and session['user'].get('access_token'):
            auth_token = session['user']['access_token']
        
        response = supabase.table('user_profiles').select('*').eq('id', user_id).execute(auth_token=auth_token)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]
        return None
    except Exception as e:
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
                session.modified = True
                
                # Get user profile
                user_profile = get_user_profile(session['user']['id'])
                if user_profile:
                    session['user']['role'] = user_profile.get('role')
                    session['user']['name'] = user_profile.get('name')
                    session.modified = True
                
                return True, user_profile
        
        return False, None
            
    except Exception as e:
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
                return False

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

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated via Supabase
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__, static_folder='static')
app.secret_key = secrets.token_hex(32)  # Generate secure secret key

# Security Configuration
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30)
)

# Security Monitoring
FAILED_LOGIN_ATTEMPTS = defaultdict(deque)
RATE_LIMIT_STORAGE = defaultdict(deque)
BLOCKED_IPS = set()
SECURITY_LOG = []

# Security Headers
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Temporarily disable CSP for testing - re-enable in production
    # response.headers['Content-Security-Policy'] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; script-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; style-src 'self' 'unsafe-inline' data: blob: https://fonts.googleapis.com; img-src 'self' data: blob:; font-src 'self' data: blob: https://fonts.gstatic.com; connect-src 'self' data: blob:; frame-src 'self'; object-src 'none'; media-src 'self' data: blob:;"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'  # HTTPS only in production
    return response

# IP-based Rate Limiting
def rate_limit(max_requests=100, window_seconds=3600):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            
            # Check if IP is blocked
            if client_ip in BLOCKED_IPS:
                return make_response('IP blocked due to suspicious activity', 403)
            
            # Clean old requests
            now = time.time()
            RATE_LIMIT_STORAGE[client_ip] = deque(
                [req_time for req_time in RATE_LIMIT_STORAGE[client_ip] 
                 if now - req_time < window_seconds],
                maxlen=max_requests
            )
            
            # Check rate limit
            if len(RATE_LIMIT_STORAGE[client_ip]) >= max_requests:
                return make_response('Rate limit exceeded', 429)
            
            # Add current request
            RATE_LIMIT_STORAGE[client_ip].append(now)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Brute Force Protection
def brute_force_protection(max_attempts=5, lockout_minutes=15):
    """Brute force protection decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method == 'POST':
                client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                username = request.form.get('username', '')
                
                # Clean old attempts
                now = time.time()
                FAILED_LOGIN_ATTEMPTS[client_ip] = deque(
                    [attempt for attempt in FAILED_LOGIN_ATTEMPTS[client_ip]
                     if now - attempt['timestamp'] < lockout_minutes * 60],
                    maxlen=max_attempts
                )
                
                # Check lockout
                if len(FAILED_LOGIN_ATTEMPTS[client_ip]) >= max_attempts:
                    BLOCKED_IPS.add(client_ip)
                    log_security_event('BRUTE_FORCE_BLOCK', client_ip, f'IP blocked after {max_attempts} failed attempts')
                    return make_response('Account temporarily locked due to too many failed attempts', 429)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Security Logging
def log_security_event(event_type, ip_address, details):
    """Log security events"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'type': event_type,
        'ip': ip_address,
        'details': details
    }
    SECURITY_LOG.append(event)
    
    # Keep only last 1000 events
    if len(SECURITY_LOG) > 1000:
        SECURITY_LOG.pop(0)
    
    print(f"[SECURITY] {event_type}: {ip_address} - {details}")

# CSRF Protection
def generate_csrf_token():
    """Generate CSRF token"""
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)
    return session['_csrf_token']

def validate_csrf_token():
    """Validate CSRF token"""
    token = session.get('_csrf_token', '')
    request_token = request.form.get('_csrf_token', '')
    return secrets.compare_digest(token, request_token)

app.jinja_env.globals['csrf_token'] = generate_csrf_token

# Old login_required removed - using Supabase-aware version defined above

@app.before_request
def before_request():
    """Apply security headers and basic checks"""
    # Add security headers
    if request.endpoint and request.endpoint != 'static':
        pass  # Headers will be added in after_request
    
    # Log suspicious user agents
    user_agent = request.headers.get('User-Agent', '')
    if any(suspicious in user_agent.lower() for suspicious in ['bot', 'crawler', 'scanner', 'sqlmap']):
        log_security_event('SUSPICIOUS_USER_AGENT', request.remote_addr, f'UA: {user_agent[:100]}')

@app.after_request
def after_request(response):
    """Add security headers to responses"""
    return add_security_headers(response)

@app.route('/')
@rate_limit(max_requests=60, window_seconds=300)  # 60 requests per 5 minutes
def index():
    """Main landing page"""
    try:
        return render_template('landing.html')
    except:
        return "Landing page not found. Please ensure landing.html exists in templates folder.", 404

@app.route('/login', methods=['GET', 'POST'])
@rate_limit(max_requests=50, window_seconds=300)  # 50 requests per 5 minutes
@brute_force_protection(max_attempts=15, lockout_minutes=5)
def login():
    """Login page with security protections"""
    if request.method == 'POST':
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        # Validate CSRF token
        if not validate_csrf_token():
            log_security_event('CSRF_FAILURE', client_ip, 'Invalid CSRF token in login')
            return render_template('login.html', error='Security validation failed')
        
        # Input validation
        if not email or not password or len(email) > 50 or len(password) > 100:
            log_security_event('INVALID_INPUT', client_ip, f'Invalid login input: email={len(email)}, password={len(password)}')
            return render_template('login.html', error='Invalid input')
        
        # Supabase authentication
        success, user_profile = authenticate_user(email, password)
        
        if success:
            session.permanent = True
            # Clear rate limit and failed attempts on successful login
            if client_ip in FAILED_LOGIN_ATTEMPTS:
                FAILED_LOGIN_ATTEMPTS[client_ip].clear()
            if client_ip in BLOCKED_IPS:
                BLOCKED_IPS.discard(client_ip)
            log_security_event('SUCCESSFUL_LOGIN', client_ip, f'User {email} logged in')
            return redirect(url_for('dashboard'))
        else:
            log_security_event('FAILED_LOGIN', client_ip, f'Failed login attempt for user: {email}')
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    user_email = session.get('user', {}).get('email', 'unknown')
    
    session.clear()
    log_security_event('LOGOUT', client_ip, f'User {user_email} logged out')
    
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
@rate_limit(max_requests=30, window_seconds=300)  # 30 requests per 5 minutes
def dashboard():
    """Serve role-based dashboard"""
    try:
        # Check if session is still valid
        if not is_session_valid():
            return redirect(url_for('login'))
        
        # Get user profile
        user_id = session['user']['id']
        user_profile = get_user_profile(user_id)
        
        if not user_profile:
            return redirect(url_for('login'))
        
        auth_token = session['user'].get('access_token')
        role = user_profile.get('role', '')
        
        # Get dashboard stats
        stats = get_dashboard_stats(role, auth_token, user_id)
        
        if role == 'admin':
            return render_template('dashboard_admin.html', 
                user=user_profile, 
                stats=stats)
        elif role == 'doctor':
            return render_template('dashboard_doctor.html', 
                user=user_profile, 
                stats=stats)
        else:
            return redirect(url_for('login'))
            
    except Exception as e:
                return redirect(url_for('login'))

def get_dashboard_stats(role, auth_token=None, user_id=None):
    """Get dashboard statistics based on role"""
    stats = {
        'total_screenings': 0,
        'today_sessions': 0,
        'total_patients': 0,
        'total_doctors': 0,
        'high_risk_count': 0,
        'moderate_risk_count': 0,
        'low_risk_count': 0
    }
    try:
        # For doctors, filter by their assigned patients
        if role == 'doctor':
            # Look up doctor's record id from doctors table using auth user_id
            from supabase_client_working import SUPABASE_SERVICE_KEY
            doctor_resp = supabase.table('doctors').select('id,name').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
            if doctor_resp.status_code == 200:
                doctors = doctor_resp.json()
                if doctors:
                    doctor_id = doctors[0]['id']
                    
                    # Filter patients by assigned_doctor_id = doctor_id (from doctors table)
                    patients_resp = supabase.table('patients').select('id,name').eq('assigned_doctor_id', doctor_id).execute(auth_token=auth_token)
                    if patients_resp.status_code == 200:
                        patients = patients_resp.json()
                        stats['total_patients'] = len(patients)
            
            # Get screenings for this doctor
            screenings_resp = supabase.table('screenings').select('id,severity,created_at').eq('doctor_id', user_id).execute(auth_token=auth_token)
            if screenings_resp.status_code == 200:
                screenings = screenings_resp.json()
                stats['total_screenings'] = len(screenings)
                for s in screenings:
                    sev = s.get('severity', '').lower()
                    if sev == 'high':
                        stats['high_risk_count'] += 1
                    elif sev == 'moderate':
                        stats['moderate_risk_count'] += 1
                    elif sev == 'low':
                        stats['low_risk_count'] += 1
                    created = s.get('created_at', '')
                    if created and created.startswith(time.strftime('%Y-%m-%d')):
                        stats['today_sessions'] += 1
        
        # For admins, get all data
        elif role == 'admin':
            # Get all screenings
            screenings_resp = supabase.table('screenings').select('id,severity,created_at').execute(auth_token=auth_token)
            if screenings_resp.status_code == 200:
                screenings = screenings_resp.json()
                stats['total_screenings'] = len(screenings)
                for s in screenings:
                    sev = s.get('severity', '').lower()
                    if sev == 'high':
                        stats['high_risk_count'] += 1
                    elif sev == 'moderate':
                        stats['moderate_risk_count'] += 1
                    elif sev == 'low':
                        stats['low_risk_count'] += 1
                    # Count today's sessions
                    created = s.get('created_at', '')
                    if created and created.startswith(time.strftime('%Y-%m-%d')):
                        stats['today_sessions'] += 1
            
            # Get all patients
            patients_resp = supabase.table('patients').select('id').execute(auth_token=auth_token)
            if patients_resp.status_code == 200:
                stats['total_patients'] = len(patients_resp.json())
            
            # Get all doctors
            doctors_resp = supabase.table('doctors').select('id').execute(auth_token=auth_token)
            if doctors_resp.status_code == 200:
                stats['total_doctors'] = len(doctors_resp.json())
    except Exception as e:
            return stats

@app.route('/api/dashboard/stats')
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        user_id = session['user']['id']
        user_profile = get_user_profile(user_id)
        if not user_profile:
            return {'error': 'Profile not found'}, 404
        auth_token = session['user'].get('access_token')
        stats = get_dashboard_stats(user_profile.get('role', ''), auth_token)
        return {'stats': stats, 'user': user_profile}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/patients')
@login_required
def api_patients():
    """API endpoint to get patients list"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        auth_token = session['user'].get('access_token')
        role = session['user'].get('role', '')
        
        if role == 'admin':
            # Use service role key to join patients with doctors
            from supabase_client_working import SUPABASE_SERVICE_KEY
            resp = supabase.client.get(
                f"{supabase.url}/rest/v1/patients?select=*,doctors(name)&order=created_at.desc",
                headers={
                    'apikey': SUPABASE_SERVICE_KEY,
                    'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                    'Content-Type': 'application/json'
                }
            )
        else:
            # Doctor sees only their patients - lookup doctor.id from doctors table first
            user_id = session['user']['id']
            from supabase_client_working import SUPABASE_SERVICE_KEY
            doctor_resp = supabase.table('doctors').select('id,name').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
            if doctor_resp.status_code == 200:
                doctors = doctor_resp.json()
                if doctors:
                    doctor_id = doctors[0]['id']
                    resp = supabase.table('patients').select('*').eq('assigned_doctor_id', doctor_id).execute(auth_token=auth_token)
                else:
                    resp = type('obj', (), {'status_code': 200, 'json': lambda: []})()
            else:
                resp = type('obj', (), {'status_code': 200, 'json': lambda: []})()
        
        if resp.status_code == 200:
            return {'patients': resp.json()}
        return {'patients': []}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/screenings')
@login_required
def api_screenings():
    """API endpoint to get screenings list"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        auth_token = session['user'].get('access_token')
        role = session['user'].get('role', '')
        
        if role == 'admin':
            resp = supabase.table('screenings').select('*').order('created_at', desc=True).execute(auth_token=auth_token)
        else:
            user_id = session['user']['id']
            resp = supabase.table('screenings').select('*').eq('doctor_id', user_id).order('created_at', desc=True).execute(auth_token=auth_token)
        
        if resp.status_code == 200:
            return {'screenings': resp.json()}
        return {'screenings': []}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/patients/<patient_id>')
@login_required
def api_patient_detail(patient_id):
    """API endpoint to get patient detail with screening history"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        auth_token = session['user'].get('access_token')
        role = session['user'].get('role', '')
        from supabase_client_working import SUPABASE_SERVICE_KEY
        
        # Get patient details
        patient_resp = supabase.table('patients').select('*').eq('id', patient_id).execute(auth_token=auth_token)
        if patient_resp.status_code != 200 or not patient_resp.json():
            return {'error': 'Patient not found'}, 404
        
        patient = patient_resp.json()[0]
        
        # For doctors, verify this patient is assigned to them
        if role == 'doctor':
            user_id = session['user']['id']
            doctor_resp = supabase.table('doctors').select('id').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
            if doctor_resp.status_code == 200 and doctor_resp.json():
                doctor_id = doctor_resp.json()[0]['id']
                if patient.get('assigned_doctor_id') != doctor_id:
                    return {'error': 'Unauthorized - not your patient'}, 403
        
        # Get screening history for this patient
        screenings_resp = supabase.table('screenings').select('*').eq('patient_id', patient_id).order('created_at', desc=True).execute(auth_token=auth_token)
        screenings = screenings_resp.json() if screenings_resp.status_code == 200 else []
        
        patient['screenings'] = screenings
        return {'patient': patient}
    except Exception as e:
                return {'error': str(e)}, 500

@app.route('/api/patients/<patient_id>/remarks', methods=['PUT'])
@login_required
def api_patient_remarks(patient_id):
    """API endpoint to add/update patient remarks (doctor only)"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        role = session['user'].get('role', '')
        if role != 'doctor':
            return {'error': 'Unauthorized'}, 403
        
        data = request.get_json()
        remarks = data.get('remarks', '').strip()
        auth_token = session['user'].get('access_token')
        from supabase_client_working import SUPABASE_SERVICE_KEY
        
        # Verify patient belongs to this doctor
        user_id = session['user']['id']
        doctor_resp = supabase.table('doctors').select('id').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
        if doctor_resp.status_code == 200 and doctor_resp.json():
            doctor_id = doctor_resp.json()[0]['id']
            patient_resp = supabase.table('patients').select('assigned_doctor_id').eq('id', patient_id).execute(auth_token=auth_token)
            if patient_resp.status_code == 200 and patient_resp.json():
                if patient_resp.json()[0].get('assigned_doctor_id') != doctor_id:
                    return {'error': 'Unauthorized - not your patient'}, 403
        
        # Update remarks using direct REST API call
        update_resp = supabase.client.patch(
            f"{supabase.url}/rest/v1/patients?id=eq.{patient_id}",
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            json={'remarks': remarks}
        )
        if update_resp.status_code == 200:
            return {'success': True, 'remarks': remarks}
        else:
                        return {'error': f'Failed to update remarks: {update_resp.text[:200]}'}, 400
    except Exception as e:
                return {'error': str(e)}, 500

@app.route('/api/screenings/save', methods=['POST'])
@login_required
def api_save_screening():
    """API endpoint to save screening result"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        role = session['user'].get('role', '')
        if role != 'doctor':
            return {'error': 'Unauthorized - only doctors can save screenings'}, 403
        
        data = request.get_json()
        patient_id = data.get('patient_id')
        severity = data.get('severity', '').strip()
        confidence_score = data.get('confidence_score')
        notes = data.get('notes', '').strip()
        
        if not patient_id or not severity:
            return {'error': 'Patient ID and severity are required'}, 400
        
        auth_token = session['user'].get('access_token')
        user_id = session['user']['id']
        from supabase_client_working import SUPABASE_SERVICE_KEY
        
        # Verify patient belongs to this doctor and get doctor record ID
        doctor_resp = supabase.table('doctors').select('id').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
        if not (doctor_resp.status_code == 200 and doctor_resp.json()):
            return {'error': 'Doctor record not found'}, 404
        doctor_id = doctor_resp.json()[0]['id']
        
        patient_resp = supabase.table('patients').select('assigned_doctor_id').eq('id', patient_id).execute(auth_token=auth_token)
        if patient_resp.status_code == 200 and patient_resp.json():
            if patient_resp.json()[0].get('assigned_doctor_id') != doctor_id:
                return {'error': 'Unauthorized - not your patient'}, 403
        
        screening_data = {
            'patient_id': patient_id,
            'doctor_id': doctor_id,
            'severity': severity,
            'notes': notes
        }
        if confidence_score is not None:
            screening_data['confidence_score'] = float(confidence_score) / 100.0 if float(confidence_score) > 1 else float(confidence_score)
        
        resp = supabase.table('screenings').insert(screening_data).execute(auth_token=SUPABASE_SERVICE_KEY)
        if resp.status_code in [200, 201]:
            return {'success': True, 'screening': resp.json()}, 201
        return {'error': f'Failed to save screening: {resp.text[:200]}'}, 400
    except Exception as e:
                return {'error': str(e)}, 500

@app.route('/api/doctors')
@login_required
def api_doctors():
    """API endpoint to get doctors list (admin only)"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        role = session['user'].get('role', '')
        if role != 'admin':
            return {'error': 'Unauthorized'}, 403
        auth_token = session['user'].get('access_token')
        from supabase_client_working import SUPABASE_SERVICE_KEY
        
        # Get doctors from doctors table first
        doctors_resp = supabase.table('doctors').select('*').order('name', desc=False).execute(auth_token=SUPABASE_SERVICE_KEY)
        if doctors_resp.status_code != 200:
            return {'doctors': []}
        
        doctors = doctors_resp.json()
        
        # Get user profiles for each doctor
        doctors_data = []
        for doctor in doctors:
            # Get user profile for this doctor
            user_profile_resp = supabase.table('user_profiles').select('email,role,position').eq('id', doctor.get('user_id')).execute(auth_token=SUPABASE_SERVICE_KEY)
            
            user_profile = {}
            if user_profile_resp.status_code == 200 and user_profile_resp.json():
                user_profile = user_profile_resp.json()[0]
            
            # Combine doctor and user profile data
            # Use position from doctors table first, then fallback to user_profiles
            position = doctor.get('position', user_profile.get('position', '-'))
            role = user_profile.get('role', doctor.get('role', '-'))
            
            doctors_data.append({
                'id': doctor.get('id'),
                'name': doctor.get('name'),
                'specialization': doctor.get('specialization'),
                'position': position,
                'role': role,
                'email': user_profile.get('email', '-'),
                'hospital_id': doctor.get('hospital_id', doctor.get('id', '-'))  # Use hospital_id field first
            })
        
        return {'doctors': doctors_data}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/doctors/create', methods=['POST'])
@login_required
def api_create_doctor():
    """API endpoint to create a new doctor account (admin only)"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        role = session['user'].get('role', '')
        if role != 'admin':
            return {'error': 'Unauthorized'}, 403
        
        data = request.get_json()
        if not data:
            return {'error': 'No data received'}, 400
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        position = data.get('position', '').strip()
        role = data.get('role', '').strip()
        specialization = data.get('specialization', '').strip()
        license_number = data.get('license_number', '').strip()
        
        if not name or not email or not password:
            return {'error': 'Name, email, and password are required'}, 400
        if len(password) < 6:
            return {'error': 'Password must be at least 6 characters'}, 400
        
        # Create auth user via Supabase Admin API using service_role key
        from supabase_client_working import SUPABASE_SERVICE_KEY
        if not SUPABASE_SERVICE_KEY:
            return {'error': 'Service role key not configured. Please add SUPABASE_SERVICE_KEY to your .env file.'}, 500
        
        auth_resp = supabase.client.post(
            f"{supabase.url}/auth/v1/admin/users",
            json={
                'email': email,
                'password': password,
                'email_confirm': True,
                'user_metadata': {
                    'name': name,
                    'role': 'doctor'
                }
            },
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json'
            }
        )
        
        if auth_resp.status_code not in [200, 201]:
            error_text = auth_resp.text[:200]
            if 'email_exists' in error_text:
                return {'error': 'A user with this email address already exists'}, 400
            return {'error': f'Failed to create auth user: {error_text}'}, 400
        
        auth_data = auth_resp.json()
        user_id = auth_data.get('id')
        
        if not user_id:
            return {'error': 'Failed to get user ID from auth response'}, 500
        
        # Create user profile using service_role key (bypass RLS)
        profile_resp = supabase.client.post(
            f"{supabase.url}/rest/v1/user_profiles",
            json={
                'id': user_id,
                'email': email,
                'role': 'doctor',
                'name': name
            },
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=minimal'
            }
        )
        
        # Generate sequential 4-digit hospital_id
        # First, get the current max hospital_id from existing doctors that have hospital_id not null
        max_id_resp = supabase.client.get(
            f"{supabase.url}/rest/v1/doctors?select=hospital_id&hospital_id=not.is.null&order=hospital_id.desc&limit=1",
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json'
            }
        )
        
        next_id = 0
        if max_id_resp.status_code == 200:
            existing_doctors = max_id_resp.json()
            if existing_doctors and len(existing_doctors) > 0:
                # Get the highest non-None hospital_id
                for doctor in existing_doctors:
                    hospital_id = doctor.get('hospital_id')
                    if hospital_id is not None:
                        try:
                            last_id = int(hospital_id)
                            next_id = last_id + 1
                            break
                        except (ValueError, TypeError):
                            continue
        
        # Format as 4-digit string with leading zeros
        hospital_id = f"{next_id:04d}"
        
        # Create doctor record (use user_id instead of id, no email column)
        doctor_data = {'user_id': user_id, 'name': name, 'hospital_id': hospital_id}
        if position:
            doctor_data['position'] = position
        if role:
            doctor_data['role'] = role
        if specialization:
            doctor_data['specialization'] = specialization
        if license_number:
            doctor_data['license_number'] = license_number
        
        doctor_resp = supabase.client.post(
            f"{supabase.url}/rest/v1/doctors",
            json=doctor_data,
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=minimal'
            }
        )
        
        return {'success': True, 'doctor_id': user_id}, 201
    except Exception as e:
                return {'error': str(e)}, 500

@app.route('/api/patients/create', methods=['POST'])
@login_required
def api_create_patient():
    """API endpoint to create a new patient record (admin only)"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        role = session['user'].get('role', '')
        if role != 'admin':
            return {'error': 'Unauthorized'}, 403
        
        data = request.get_json()
        name = data.get('name', '').strip()
        date_of_birth = data.get('date_of_birth', '').strip()
        contact_number = data.get('contact_number', '').strip()
        emergency_contact = data.get('emergency_contact', '').strip()
        identification_number = data.get('identification_number', '').strip()
        occupation = data.get('occupation', '').strip()
        ethnicity = data.get('ethnicity', '').strip()
        assigned_doctor_id = data.get('doctor_id', '').strip()
        
        if not name or not date_of_birth or not contact_number or not emergency_contact or not identification_number:
            return {'error': 'Patient name, date of birth, contact number, emergency contact, and identification number are required'}, 400
        
        # Generate sequential medical_id (similar to hospital_id)
        from supabase_client_working import SUPABASE_SERVICE_KEY
        max_id_resp = supabase.client.get(
            f"{supabase.url}/rest/v1/patients?select=medical_id&medical_id=not.is.null&order=medical_id.desc&limit=1",
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json'
            }
        )
        
        next_id = 0
        if max_id_resp.status_code == 200:
            existing_patients = max_id_resp.json()
            if existing_patients and len(existing_patients) > 0:
                for patient in existing_patients:
                    medical_id = patient.get('medical_id')
                    if medical_id is not None:
                        try:
                            last_id = int(medical_id)
                            next_id = last_id + 1
                            break
                        except (ValueError, TypeError):
                            continue
        
        # Format as 6-digit medical ID
        medical_id = f"{next_id:06d}"
        
        patient_data = {
            'name': name, 
            'medical_id': medical_id, 
            'date_of_birth': date_of_birth,
            'contact_number': contact_number,
            'emergency_contact': emergency_contact,
            'identification_number': identification_number
        }
        if occupation:
            patient_data['occupation'] = occupation
        if ethnicity:
            patient_data['ethnicity'] = ethnicity
        if assigned_doctor_id:
            patient_data['assigned_doctor_id'] = assigned_doctor_id
        
        # Use service role key for admin operations (like doctor creation)
        resp = supabase.client.post(
            f"{supabase.url}/rest/v1/patients",
            json=patient_data,
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            }
        )
        
        if resp.status_code in [200, 201]:
            return {'success': True, 'patient': resp.json()}, 201
        return {'error': f'Failed to create patient: {resp.text[:200]}'}, 400
    except Exception as e:
                return {'error': str(e)}, 500

@app.route('/portal')
@app.route('/portal/<patient_id>')
@login_required
@rate_limit(max_requests=30, window_seconds=300)  # 30 requests per 5 minutes
def portal(patient_id=None):
    """Serve working ensemble portal, optionally with patient context"""
    try:
        return render_template('working_ensemble.html', patient_id=patient_id, user=session.get('user'))
    except:
        return "Portal file not found. Please ensure working_ensemble.html exists in templates folder.", 404

@app.route('/models/<path:filename>')
@login_required
def serve_models(filename):
    """Serve model files for real model loading"""
    try:
        # Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            log_security_event('SUSPICIOUS_FILE_ACCESS', request.remote_addr, f'Attempted access to: {filename}')
            return make_response('Invalid file path', 400)
        
        return send_from_directory('models', filename)
    except:
        return "Model file not found.", 404

@app.route('/css/<path:filename>')
@rate_limit(max_requests=20, window_seconds=300)
def serve_css(filename):
    """Serve CSS files"""
    try:
        # Validate filename
        if '..' in filename or '/' in filename or '\\' in filename:
            log_security_event('SUSPICIOUS_FILE_ACCESS', request.remote_addr, f'Attempted CSS access to: {filename}')
            return make_response('Invalid file path', 400)
        
        return send_from_directory('static/css', filename)
    except:
        return f"CSS file {filename} not found.", 404

@app.route('/js/<path:filename>')
@rate_limit(max_requests=20, window_seconds=300)
def serve_js(filename):
    """Serve JavaScript files"""
    try:
        # Validate filename
        if '..' in filename or '/' in filename or '\\' in filename:
            log_security_event('SUSPICIOUS_FILE_ACCESS', request.remote_addr, f'Attempted JS access to: {filename}')
            return make_response('Invalid file path', 400)
        
        return send_from_directory('static/js', filename)
    except:
        return f"JavaScript file {filename} not found.", 404

@app.route('/security-info')
@login_required
def security_info():
    """Security monitoring dashboard (admin only)"""
    if session.get('username') != 'admin':
        return make_response('Unauthorized', 403)
    
    return {
        'blocked_ips': list(BLOCKED_IPS),
        'failed_attempts': dict(FAILED_LOGIN_ATTEMPTS),
        'rate_limits': dict(RATE_LIMIT_STORAGE),
        'security_log': SECURITY_LOG[-50:],  # Last 50 events
        'session_info': {
            'active_sessions': len([s for s in SECURITY_LOG if 'SUCCESSFUL_LOGIN' in s.get('type', '')]),
            'blocked_ips_count': len(BLOCKED_IPS)
        }
    }

def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("Starting Professional Flask Server with Security Features...")
    print("Server Information:")
    print("   - Port: 5000")
    print("   - Login: http://localhost:5000/login")
    print("   - Portal: http://localhost:5000/portal")
    print("   - Security: Rate limiting, CSRF protection, Brute force protection")
    print("   - Protocol: HTTP")
    print(" Opening browser in 1.5 seconds...")
    
    # Start browser in background
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
