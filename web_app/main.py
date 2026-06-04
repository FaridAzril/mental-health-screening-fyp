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
import json
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
from supabase_client_working import supabase, SUPABASE_SERVICE_KEY

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

def get_doctor_status(user_id):
    try:
        response = supabase.table('doctors').select('status').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0].get('status', 'active')
        return None
    except Exception as e:
                return None

def is_inactive_doctor(user_profile, user_id):
    if not user_profile or user_profile.get('role') != 'doctor':
        return False
    return get_doctor_status(user_id) == 'inactive'

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
                    if is_inactive_doctor(user_profile, session['user']['id']):
                        session.clear()
                        return False, None
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
    
    # Check if session is older than 8 hours
    last_activity = session.get('last_activity', 0)
    current_time = time.time()
    
    # 8 hours = 28800 seconds
    if current_time - last_activity > 28800:
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

# IP-based Rate Limiting (per-endpoint)
def rate_limit(max_requests=100, window_seconds=3600):
    """Rate limiting decorator - uses endpoint-specific keys"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            
            # Check if IP is blocked
            if client_ip in BLOCKED_IPS:
                return make_response('IP blocked due to suspicious activity', 403)
            
            # Use endpoint-specific key so different endpoints don't share limits
            endpoint_key = f"{client_ip}:{f.__name__}"
            
            # Clean old requests
            now = time.time()
            RATE_LIMIT_STORAGE[endpoint_key] = deque(
                [req_time for req_time in RATE_LIMIT_STORAGE[endpoint_key] 
                 if now - req_time < window_seconds],
                maxlen=max_requests
            )
            
            # Check rate limit
            if len(RATE_LIMIT_STORAGE[endpoint_key]) >= max_requests:
                return make_response('Rate limit exceeded', 429)
            
            # Add current request
            RATE_LIMIT_STORAGE[endpoint_key].append(now)
            
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
        if is_inactive_doctor(user_profile, user_id):
            session.clear()
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
            # Get all screenings with patient and doctor names
            screenings_resp = supabase.table('screenings').select('id,severity,created_at,patients!inner(name),doctors!inner(name)').execute(auth_token=auth_token)
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

@app.route('/api/session/status')
def api_session_status():
    try:
        if not is_session_valid():
            session.clear()
            return {'authenticated': False}, 401

        user_id = session['user']['id']
        user_profile = get_user_profile(user_id)
        if not user_profile:
            session.clear()
            return {'authenticated': False}, 401

        if is_inactive_doctor(user_profile, user_id):
            session.clear()
            return {'authenticated': False, 'account_status': 'inactive'}, 403

        return {
            'authenticated': True,
            'role': user_profile.get('role'),
            'account_status': 'active'
        }, 200
    except Exception as e:
        return {'authenticated': False}, 401

@app.route('/api/patients')
@login_required
def api_patients():
    """API endpoint to get patients list"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        
        role = session['user'].get('role', '')
        from supabase_client_working import SUPABASE_SERVICE_KEY
        print(f"[api_patients] role={role}, service_key_exists={bool(SUPABASE_SERVICE_KEY)}")
        
        svc_headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json'
        }
        
        if role == 'admin':
            resp = supabase.client.get(
                f"{supabase.url}/rest/v1/patients?select=*,doctors(id,name)&order=created_at.desc",
                headers=svc_headers, timeout=10.0
            )
            print(f"[api_patients] admin query status={resp.status_code}")
            if resp.status_code == 200:
                return {'patients': resp.json()}
            print(f"[api_patients] admin query failed: {resp.text[:200]}")
            return {'patients': []}
        else:
            # Doctor: get all patients assigned to this doctor
            user_id = session['user']['id']
            print(f"[api_patients] doctor user_id={user_id}")
            
            # First get doctor record ID
            doctor_resp = supabase.client.get(
                f"{supabase.url}/rest/v1/doctors?select=id&user_id=eq.{user_id}",
                headers=svc_headers, timeout=10.0
            )
            print(f"[api_patients] doctor lookup status={doctor_resp.status_code}")
            
            if doctor_resp.status_code == 200:
                doctors = doctor_resp.json()
                print(f"[api_patients] doctors found: {doctors}")
                if doctors and len(doctors) > 0:
                    doctor_id = doctors[0]['id']
                    patients_resp = supabase.client.get(
                        f"{supabase.url}/rest/v1/patients?select=*&assigned_doctor_id=eq.{doctor_id}&order=created_at.desc",
                        headers=svc_headers, timeout=10.0
                    )
                    print(f"[api_patients] patients query status={patients_resp.status_code}")
                    if patients_resp.status_code == 200:
                        patients = patients_resp.json()
                        print(f"[api_patients] returning {len(patients)} patients")
                        return {'patients': patients}
                    print(f"[api_patients] patients query failed: {patients_resp.text[:200]}")
                else:
                    print(f"[api_patients] no doctor record found for user_id={user_id}")
            else:
                print(f"[api_patients] doctor lookup failed: {doctor_resp.text[:200]}")
            
            return {'patients': []}
    except Exception as e:
        print(f"[api_patients] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
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
        from supabase_client_working import SUPABASE_SERVICE_KEY
        
        if role == 'admin':
            resp = supabase.table('screenings').select('*, patients!inner(name), doctors!inner(name)').order('created_at', desc=True).execute(auth_token=SUPABASE_SERVICE_KEY)
        else:
            # Get doctor record ID from doctors table
            user_id = session['user']['id']
            doctor_resp = supabase.table('doctors').select('id').eq('user_id', user_id).execute(auth_token=SUPABASE_SERVICE_KEY)
            if doctor_resp.status_code == 200 and doctor_resp.json():
                doctor_id = doctor_resp.json()[0]['id']
                resp = supabase.table('screenings').select('*, patients!inner(name)').eq('doctor_id', doctor_id).order('created_at', desc=True).execute(auth_token=SUPABASE_SERVICE_KEY)
            else:
                return {'screenings': []}
        
        if resp.status_code == 200:
            screenings = resp.json()
            return {'screenings': screenings}
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
        screenings_resp = supabase.table('screenings').select('*').eq('patient_id', patient_id).order('created_at', desc=True).execute(auth_token=SUPABASE_SERVICE_KEY)
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
        new_remark = data.get('remarks', '').strip()
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
        
        # Get existing remarks to append
        patient_data = supabase.table('patients').select('remarks').eq('id', patient_id).execute(auth_token=SUPABASE_SERVICE_KEY)
        existing_remarks = ''
        if patient_data.status_code == 200 and patient_data.json():
            existing_remarks = patient_data.json()[0].get('remarks', '') or ''
        
        # Append new remark with newline separator (no date/time)
        if existing_remarks:
            updated_remarks = existing_remarks + '\n\n' + new_remark
        else:
            updated_remarks = '\n\n' + new_remark  # Add spacing for first remark too
        
        # Update remarks using direct REST API call
        update_resp = supabase.client.patch(
            f"{supabase.url}/rest/v1/patients?id=eq.{patient_id}",
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            json={'remarks': updated_remarks}
        )
        if update_resp.status_code == 200:
            return {'success': True, 'remarks': updated_remarks}
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
        
        # Debug: Check if request body is being received
        print(f"DEBUG: Raw request data: {request.data}")
        print(f"DEBUG: Request content type: {request.content_type}")
        
        data = request.get_json()
        print(f"DEBUG: Parsed JSON data: {data}")
        
        if not data:
            return {'error': 'No data received in request'}, 400
            
        patient_id = data.get('patient_id')
        severity = data.get('severity', '').strip()
        confidence_score = data.get('confidence_score')
        notes = data.get('notes', '').strip()
        clinical_note = data.get('clinical_note', '').strip()
        session_data = data.get('sessionData', [])  # Get session data from frontend
        
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
        
        # Map severity string to integer for severity_level column (1=Low, 2=Moderate, 3=High)
        severity_map = {'Low': 1, 'Moderate': 2, 'High': 3}
        severity_level = severity_map.get(severity, 2)  # Default to 2 (Moderate) if unknown
        
        screening_data = {
            'patient_id': patient_id,
            'doctor_id': doctor_id,
            'severity': severity,
            'severity_level': severity_level,
            'notes': notes,
            'clinical_note': clinical_note if clinical_note else None,
            'session_data': json.dumps(session_data) if session_data else None
        }
        if confidence_score is not None:
            try:
                conf_score = float(confidence_score)
                if conf_score > 1:
                    conf_score = conf_score / 100.0
                screening_data['confidence_score'] = conf_score
                print(f"DEBUG: Confidence score processed: {conf_score}")
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Error processing confidence score: {e}")
                pass
        
        # Debug logging
        print(f"DEBUG: Sending screening data: {screening_data}")
        print(f"DEBUG: Patient ID: {patient_id} (type: {type(patient_id)})")
        print(f"DEBUG: Doctor ID: {doctor_id} (type: {type(doctor_id)})")
        print(f"DEBUG: Severity: {severity} (type: {type(severity)})")
        
        try:
            resp = supabase.table('screenings').insert(screening_data).execute(auth_token=SUPABASE_SERVICE_KEY)
            print(f"DEBUG: Supabase response status: {resp.status_code}")
            print(f"DEBUG: Supabase response text: {resp.text[:200] if hasattr(resp, 'text') else str(resp)}")
            
            if resp.status_code in [200, 201]:
                # Only try to parse JSON if there's actual content
                if resp.text and resp.text.strip():
                    return {'success': True, 'screening': resp.json()}, 201
                else:
                    # Successful insert but empty response (common for Supabase)
                    return {'success': True, 'screening': screening_data}, 201
            return {'error': f'Failed to save screening: {resp.text[:200]}'}, 400
        except Exception as supabase_error:
            print(f"DEBUG: Supabase exception: {supabase_error}")
            print(f"DEBUG: Exception type: {type(supabase_error)}")
            return {'error': f'Supabase error: {str(supabase_error)}'}, 500
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
                'hospital_id': doctor.get('hospital_id', doctor.get('id', '-')),
                'license_number': doctor.get('license_number', ''),
                'status': doctor.get('status', 'active')
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
        
        admin_email = session['user'].get('email', 'unknown')
        log_security_event('DOCTOR_CREATED', request.remote_addr, f'Admin {admin_email} created doctor account: {name} ({email})')
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
            admin_email = session['user'].get('email', 'unknown')
            log_security_event('PATIENT_CREATED', request.remote_addr, f'Admin {admin_email} created patient: {name} (ID: {medical_id})')
            return {'success': True, 'patient': resp.json()}, 201
        return {'error': f'Failed to create patient: {resp.text[:200]}'}, 400
    except Exception as e:
                return {'error': str(e)}, 500

@app.route('/api/doctors/update', methods=['POST'])
@login_required
def api_update_doctor():
    """Update doctor account details"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        if session['user'].get('role') != 'admin':
            return {'error': 'Unauthorized'}, 403
        
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        if not doctor_id:
            return {'error': 'Doctor ID required'}, 400
        
        from supabase_client_working import SUPABASE_SERVICE_KEY
        update_data = {}
        if data.get('name'):
            update_data['name'] = data['name'].strip()
        if data.get('position'):
            update_data['position'] = data['position'].strip()
        if data.get('specialization'):
            update_data['specialization'] = data['specialization'].strip()
        if data.get('license_number'):
            update_data['license_number'] = data['license_number'].strip()
        
        if not update_data:
            return {'error': 'No fields to update'}, 400
        
        resp = supabase.client.patch(
            f"{supabase.url}/rest/v1/doctors?id=eq.{doctor_id}",
            json=update_data,
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            }
        )
        
        if resp.status_code in [200, 204]:
            admin_email = session['user'].get('email', 'unknown')
            log_security_event('DOCTOR_UPDATED', request.remote_addr, f'Admin {admin_email} updated doctor {doctor_id}: {update_data}')
            return {'success': True}, 200
        return {'error': f'Failed to update doctor: {resp.text[:200]}'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/doctors/update-status', methods=['POST'])
@login_required
def api_update_doctor_status():
    """Activate or deactivate a doctor account"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        if session['user'].get('role') != 'admin':
            return {'error': 'Unauthorized'}, 403
        
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        new_status = data.get('status')
        
        if not doctor_id or not new_status:
            return {'error': 'Doctor ID and status required'}, 400
        if new_status not in ['active', 'inactive']:
            return {'error': 'Status must be active or inactive'}, 400
        
        from supabase_client_working import SUPABASE_SERVICE_KEY
        resp = supabase.client.patch(
            f"{supabase.url}/rest/v1/doctors?id=eq.{doctor_id}",
            json={'status': new_status},
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            }
        )
        
        if resp.status_code in [200, 204]:
            admin_email = session['user'].get('email', 'unknown')
            log_security_event('DOCTOR_STATUS_CHANGED', request.remote_addr, f'Admin {admin_email} changed doctor {doctor_id} status to {new_status}')
            return {'success': True, 'status': new_status}, 200
        return {'error': f'Failed to update status: {resp.text[:200]}'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/patients/reassign', methods=['POST'])
@login_required
def api_reassign_patient():
    """Reassign a patient to a different doctor"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        if session['user'].get('role') != 'admin':
            return {'error': 'Unauthorized'}, 403
        
        data = request.get_json()
        patient_id = data.get('patient_id')
        new_doctor_id = data.get('new_doctor_id')
        
        if not patient_id or not new_doctor_id:
            return {'error': 'Patient ID and new doctor ID required'}, 400
        
        from supabase_client_working import SUPABASE_SERVICE_KEY
        resp = supabase.client.patch(
            f"{supabase.url}/rest/v1/patients?id=eq.{patient_id}",
            json={'assigned_doctor_id': new_doctor_id},
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            }
        )
        
        if resp.status_code in [200, 204]:
            admin_email = session['user'].get('email', 'unknown')
            log_security_event('PATIENT_REASSIGNED', request.remote_addr, f'Admin {admin_email} reassigned patient {patient_id} to doctor {new_doctor_id}')
            return {'success': True}, 200
        return {'error': f'Failed to reassign patient: {resp.text[:200]}'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/audit-log')
@login_required
def api_audit_log():
    """API endpoint to get audit/activity log (admin only)"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        if session['user'].get('role') != 'admin':
            return {'error': 'Unauthorized'}, 403
        
        # Return security log in reverse chronological order
        logs = list(reversed(SECURITY_LOG))
        return {'logs': logs}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/doctor/profile', methods=['GET'])
@login_required
def api_doctor_profile():
    """Get the logged-in doctor's profile (read-only, edits managed by admin)"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        if session['user'].get('role') != 'doctor':
            return {'error': 'Unauthorized'}, 403
        
        user_id = session['user']['id']
        from supabase_client_working import SUPABASE_SERVICE_KEY
        svc_headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json'
        }
        
        resp = supabase.client.get(
            f"{supabase.url}/rest/v1/doctors?select=*&user_id=eq.{user_id}",
            headers=svc_headers, timeout=10.0
        )
        if resp.status_code == 200 and resp.json():
            return {'doctor': resp.json()[0]}, 200
        return {'doctor': None}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/screenings/<screening_id>/note', methods=['PUT'])
@login_required
def api_screening_note(screening_id):
    """Add or update a clinical note on a screening"""
    try:
        if 'user' not in session:
            return {'error': 'Not authenticated'}, 401
        
        data = request.get_json()
        clinical_note = data.get('clinical_note', '').strip()
        
        from supabase_client_working import SUPABASE_SERVICE_KEY
        resp = supabase.client.patch(
            f"{supabase.url}/rest/v1/screenings?id=eq.{screening_id}",
            json={'clinical_note': clinical_note},
            headers={
                'apikey': SUPABASE_SERVICE_KEY,
                'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            }
        )
        
        if resp.status_code in [200, 204]:
            return {'success': True}, 200
        return {'error': f'Failed to save note: {resp.text[:200]}'}, 400
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

@app.route('/extract-au-features', methods=['POST'])
@login_required
@rate_limit(max_requests=1000, window_seconds=60)
def extract_au_features():
    """Extract AU features from video frame using MediaPipe Face Mesh landmarks"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import io
        import base64
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
        
        data = request.get_json()
        if not data or 'frame_data' not in data:
            return {'error': 'Frame data required'}, 400
        
        frame_data = data['frame_data']
        if frame_data.startswith('data:image/'):
            frame_data = frame_data.split(',')[1]
        
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize FaceLandmarker if not cached
        if not hasattr(extract_au_features, '_landmarker'):
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'face_landmarker.task')
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5
            )
            extract_au_features._landmarker = vision.FaceLandmarker.create_from_options(options)
        
        landmarker = extract_au_features._landmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)
        
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return {'au_features': [0.0] * 17, 'face_detected': False, 'landmark_count': 0}
        
        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]
        
        def dist(i, j):
            return np.sqrt((landmarks[i].x - landmarks[j].x)**2 + (landmarks[i].y - landmarks[j].y)**2)
        
        # Face normalization reference (distance between outer eye corners)
        face_width = dist(33, 263)
        if face_width < 0.001:
            face_width = 0.1
        
        # AU1 - Inner Brow Raiser: inner eyebrow height relative to eye
        au1 = (landmarks[107].y - landmarks[159].y) / face_width  # left
        au1_r = (landmarks[336].y - landmarks[386].y) / face_width  # right
        au1_val = np.clip((au1 + au1_r) * 3.0, 0, 1)
        
        # AU2 - Outer Brow Raiser: outer eyebrow height
        au2 = (landmarks[70].y - landmarks[159].y) / face_width
        au2_r = (landmarks[300].y - landmarks[386].y) / face_width
        au2_val = np.clip((au2 + au2_r) * 3.0, 0, 1)
        
        # AU4 - Brow Lowerer: distance between inner brows
        au4_val = np.clip(1.0 - dist(55, 285) / face_width * 3.0, 0, 1)
        
        # AU5 - Upper Lid Raiser: eye openness
        eye_open_l = dist(159, 145) / face_width
        eye_open_r = dist(386, 374) / face_width
        au5_val = np.clip((eye_open_l + eye_open_r) * 5.0, 0, 1)
        
        # AU6 - Cheek Raiser: cheek landmark height
        au6 = (landmarks[117].y - landmarks[50].y) / face_width
        au6_r = (landmarks[346].y - landmarks[280].y) / face_width
        au6_val = np.clip((au6 + au6_r) * 3.0, 0, 1)
        
        # AU7 - Lid Tightener: lower eyelid tension
        au7_val = np.clip(1.0 - (eye_open_l + eye_open_r) * 4.0, 0, 1)
        
        # AU9 - Nose Wrinkler: nose bridge area compression
        au9_val = np.clip(dist(6, 197) / face_width * 2.0, 0, 1)
        
        # AU10 - Upper Lip Raiser
        au10_val = np.clip((landmarks[0].y - landmarks[13].y) / face_width * 5.0, 0, 1)
        
        # AU12 - Lip Corner Puller (smile): mouth width relative to rest
        mouth_width = dist(61, 291)
        au12_val = np.clip(mouth_width / face_width * 2.0 - 0.5, 0, 1)
        
        # AU14 - Dimpler: lip corner depth
        au14_val = np.clip(abs(landmarks[61].z - landmarks[291].z) * 10.0, 0, 1)
        
        # AU15 - Lip Corner Depressor (frown)
        lip_mid_y = (landmarks[13].y + landmarks[14].y) / 2
        lip_corner_y = (landmarks[61].y + landmarks[291].y) / 2
        au15_val = np.clip((lip_corner_y - lip_mid_y) / face_width * 5.0, 0, 1)
        
        # AU17 - Chin Raiser
        au17_val = np.clip((landmarks[152].y - landmarks[17].y) / face_width * 3.0, 0, 1)
        
        # AU20 - Lip Stretcher
        au20_val = np.clip(mouth_width / face_width * 1.5 - 0.3, 0, 1)
        
        # AU23 - Lip Tightener: lip thickness
        lip_thickness = dist(13, 14)
        au23_val = np.clip(1.0 - lip_thickness / face_width * 8.0, 0, 1)
        
        # AU25 - Lips Part: mouth openness vertical
        mouth_open = dist(13, 14)
        au25_val = np.clip(mouth_open / face_width * 6.0, 0, 1)
        
        # AU26 - Jaw Drop
        jaw_drop = dist(13, 17)
        au26_val = np.clip(jaw_drop / face_width * 3.0 - 0.3, 0, 1)
        
        # AU45 - Blink: eye closure
        au45_val = np.clip(1.0 - (eye_open_l + eye_open_r) * 6.0, 0, 1)
        
        au_features = [
            au1_val, au2_val, au4_val, au5_val, au6_val, au7_val,
            au9_val, au10_val, au12_val, au14_val, au15_val, au17_val,
            au20_val, au23_val, au25_val, au26_val, au45_val
        ]
        
        return {
            'au_features': [float(v) for v in au_features],
            'face_detected': True,
            'landmark_count': len(landmarks)
        }
        
    except Exception as e:
        print(f"AU extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500

@app.route('/predict-severity', methods=['POST'])
@login_required
@rate_limit(max_requests=500, window_seconds=60)
def predict_severity():
    """Real ensemble model prediction endpoint"""
    try:
        import tensorflow as tf
        import numpy as np
        
        data = request.get_json()
        if not data or 'au_features' not in data:
            return {'error': 'AU features required'}, 400
        
        au_features = np.array(data['au_features'])
        
        # Reshape to (1, 300, 17) for model input
        if len(au_features.shape) == 2:
            au_features = au_features.reshape(1, au_features.shape[0], au_features.shape[1])
        
        if au_features.shape != (1, 300, 17):
            return {'error': f'Invalid shape. Expected (1, 300, 17), got {au_features.shape}'}, 400
        
        # Load models (cached)
        if not hasattr(predict_severity, 'models_loaded'):
            print("Loading ensemble models...")
            predict_severity.models = []
            for i in range(1, 4):
                model_path = f'models/ensemble_{i}.h5'
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    predict_severity.models.append(model)
                    print(f"Loaded ensemble_{i}.h5")
                except Exception as e:
                    print(f"Failed to load ensemble_{i}.h5: {e}")
                    return {'error': f'Model loading failed: {str(e)}'}, 500
            predict_severity.models_loaded = True
            print("All ensemble models loaded!")
        
        # Standardize input
        au_flat = au_features.reshape(-1, 17)
        mean = np.mean(au_flat, axis=0)
        std = np.std(au_flat, axis=0) + 1e-8
        au_scaled = (au_flat - mean) / std
        au_processed = au_scaled.reshape(1, 300, 17)
        
        # Get predictions from all 3 models
        model_predictions = []
        for i, model in enumerate(predict_severity.models):
            pred = model.predict(au_processed, verbose=0)[0]
            model_predictions.append(pred)
        
        # Ensemble voting with weights [1.0, 1.0, 1.5]
        weights = [1.0, 1.0, 1.5]
        weighted_results = np.zeros(3)
        for i, pred in enumerate(model_predictions):
            for j in range(3):
                weighted_results[j] += pred[j] * weights[i]
        
        total_weight = sum(weights)
        final_probs = weighted_results / total_weight
        
        # Pure argmax - pick the class with highest probability
        low, moderate, high = final_probs
        severity_classes = ['Low', 'Moderate', 'High']
        severity = severity_classes[int(np.argmax(final_probs))]
        
        return {
            'severity': severity,
            'confidence': float(np.max(final_probs)),
            'probabilities': {
                'low': float(low),
                'moderate': float(moderate),
                'high': float(high)
            },
            'model_predictions': [
                {'model': f'Model {i+1}', 'probabilities': {'low': float(p[0]), 'moderate': float(p[1]), 'high': float(p[2])}}
                for i, p in enumerate(model_predictions)
            ]
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {'error': str(e)}, 500

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
@rate_limit(max_requests=500, window_seconds=300)
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
@rate_limit(max_requests=500, window_seconds=300)
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
    
    # Preload ensemble models at startup
    try:
        import tensorflow as tf
        print(" Loading ensemble models...")
        predict_severity.models = []
        for i in range(1, 4):
            model_path = f'models/ensemble_{i}.h5'
            model = tf.keras.models.load_model(model_path, compile=False)
            predict_severity.models.append(model)
            print(f"   ✓ Loaded ensemble_{i}.h5")
        predict_severity.models_loaded = True
        print(" All 3 ensemble models ready!")
    except Exception as e:
        print(f" Warning: Could not preload models: {e}")
    
    print(" Opening browser in 1.5 seconds...")
    
    # Start browser in background
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
