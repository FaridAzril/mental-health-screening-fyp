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

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            log_security_event('UNAUTHORIZED_ACCESS', request.remote_addr, 'Attempted to access protected route')
            return redirect(url_for('login'))
        
        # Check session timeout
        last_activity = session.get('last_activity', time.time())
        if time.time() - last_activity > app.config['PERMANENT_SESSION_LIFETIME'].total_seconds():
            session.clear()
            log_security_event('SESSION_TIMEOUT', request.remote_addr, 'Session expired')
            return redirect(url_for('login'))
        
        # Update last activity
        session['last_activity'] = time.time()
        return f(*args, **kwargs)
    return decorated_function

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
@rate_limit(max_requests=10, window_seconds=300)  # 10 login attempts per 5 minutes
@brute_force_protection(max_attempts=5, lockout_minutes=15)
def login():
    """Login page with security protections"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        # Validate CSRF token
        if not validate_csrf_token():
            log_security_event('CSRF_FAILURE', client_ip, 'Invalid CSRF token in login')
            return render_template('login.html', error='Security validation failed')
        
        # Input validation
        if not username or not password or len(username) > 50 or len(password) > 100:
            log_security_event('INVALID_INPUT', client_ip, f'Invalid login input: username={len(username)}, password={len(password)}')
            return render_template('login.html', error='Invalid input')
        
        # Demo authentication (in production, use proper database authentication)
        if username == 'admin' and password == 'demo123':
            session['user_id'] = username
            session['username'] = username
            session['login_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            session['last_activity'] = time.time()
            session.permanent = True
            
            # Clear failed attempts on successful login
            if client_ip in FAILED_LOGIN_ATTEMPTS:
                del FAILED_LOGIN_ATTEMPTS[client_ip]
            
            log_security_event('SUCCESSFUL_LOGIN', client_ip, f'User {username} logged in')
            
            # Return HTML with sessionStorage setting
            return render_template('login_success.html', username=username)
        else:
            # Log failed attempt
            FAILED_LOGIN_ATTEMPTS[client_ip].append({
                'timestamp': time.time(),
                'username': username
            })
            
            log_security_event('FAILED_LOGIN', client_ip, f'Failed login attempt for user: {username}')
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    username = session.get('username', 'unknown')
    
    session.clear()
    log_security_event('LOGOUT', client_ip, f'User {username} logged out')
    
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
@rate_limit(max_requests=30, window_seconds=300)  # 30 requests per 5 minutes
def dashboard():
    """Serve dashboard page"""
    try:
        return render_template('dashboard.html')
    except:
        return "Dashboard file not found. Please ensure dashboard.html exists in templates folder.", 404

@app.route('/portal')
@login_required
@rate_limit(max_requests=30, window_seconds=300)  # 30 requests per 5 minutes
def portal():
    """Serve working ensemble portal"""
    try:
        return render_template('working_ensemble.html')
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
