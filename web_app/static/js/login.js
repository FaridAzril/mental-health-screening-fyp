// Login page JavaScript functionality

// Add enter key support for form
document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        const form = document.getElementById('login-form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
});

// Form focus management
document.addEventListener('DOMContentLoaded', function() {
    // Auto-focus on username field
    const usernameInput = document.getElementById('username');
    if (usernameInput) {
        usernameInput.focus();
    }
    
    // Add input validation feedback
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.value.trim() === '') {
                this.style.borderColor = '#ef4444';
            } else {
                this.style.borderColor = '#d1d5db';
            }
        });
        
        input.addEventListener('focus', function() {
            this.style.borderColor = '#3b82f6';
        });
    });
});

// Button loading state - only activate if form is not submitting
const loginButton = document.querySelector('.login-button');
if (loginButton) {
    loginButton.addEventListener('click', function(e) {
        // Don't interfere with form submission
        // This is handled by the form's onsubmit attribute
    });
    
    // Handle form submission for loading state
    document.getElementById('login-form').addEventListener('submit', function() {
        if (loginButton) {
            loginButton.disabled = true;
            loginButton.innerHTML = '<span class="button-text">Authenticating...</span>';
            
            // Re-enable after 3 seconds (fallback)
            setTimeout(() => {
                if (loginButton) {
                    loginButton.disabled = false;
                    loginButton.innerHTML = '<span class="button-text">Access Portal</span>';
                }
            }, 3000);
        }
    });
}

// Handle successful authentication
function handleSuccessfulAuthentication() {
    // Set session storage for client-side checks
    sessionStorage.setItem('authenticated', 'true');
    sessionStorage.setItem('username', document.getElementById('username').value);
    sessionStorage.setItem('loginTime', new Date().toISOString());
}
