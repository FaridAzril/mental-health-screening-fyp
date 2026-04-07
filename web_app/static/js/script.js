// Professional Mental Health Screening Portal JavaScript

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeLandingPage();
});

// Initialize landing page functionality
function initializeLandingPage() {
    setupLaunchButton();
    addMicroInteractions();
    setupAccessibilityFeatures();
}

// Setup launch button functionality
function setupLaunchButton() {
    const launchButton = document.getElementById('launch-portal');
    
    if (launchButton) {
        // Handle launch portal button
        function handleLaunchPortal() {
            // Add loading state
            launchButton.classList.add('loading');
            launchButton.innerHTML = `
                <span class="button-text">Launching Portal...</span>
            `;
            
            // Disable button during launch
            launchButton.disabled = true;
            
            // Open portal in same tab
            setTimeout(() => {
                window.open('/portal', '_self');
                
                // Reset button after delay
                setTimeout(() => {
                    launchButton.classList.remove('loading');
                    launchButton.innerHTML = `
                        <span class="button-text">Launch Screening Portal</span>
                    `;
                    launchButton.disabled = false;
                }, 1500);
            }, 500);
        }

        // Event listeners
        launchButton.addEventListener('click', handleLaunchPortal);
        
        // Add keyboard support
        launchButton.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                handleLaunchPortal();
            }
        });
    }
}

// Handle portal launch
function handleLaunchPortal() {
    const launchButton = document.getElementById('launch-portal');
    
    // Add loading state
    launchButton.classList.add('loading');
    launchButton.innerHTML = `
        <span class="button-text">Launching Portal...</span>
    `;
    
    // Disable button during launch
    launchButton.disabled = true;
    
    // Open portal in new window
    setTimeout(() => {
        window.open('http://localhost:5000/portal', '_blank', 'noopener,noreferrer');
        
        // Reset button after delay
        setTimeout(() => {
            launchButton.classList.remove('loading');
            launchButton.innerHTML = `
                <span class="button-text">Launch Screening Portal</span>
            `;
            launchButton.disabled = false;
        }, 1500);
    }, 500);
}

// Add micro-interactions
function addMicroInteractions() {
    // Logo hover effect
    const logoContainer = document.querySelector('.logo-container');
    if (logoContainer) {
        logoContainer.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05) rotate(5deg)';
        });
        
        logoContainer.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1) rotate(0deg)';
        });
    }
    
    // Feature cards hover effects
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
        
        // Stagger animation on load
        setTimeout(() => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.5s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100 * (index + 1));
        }, 300);
    });
}

// Setup accessibility features
function setupAccessibilityFeatures() {
    // Add ARIA labels
    const launchButton = document.getElementById('launch-portal');
    if (launchButton) {
        launchButton.setAttribute('aria-label', 'Launch Mental Health Screening Portal');
        launchButton.setAttribute('role', 'button');
    }
    
    // Add skip to content link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'skip-link';
    skipLink.setAttribute('aria-label', 'Skip to main content');
    document.body.insertBefore(skipLink, document.body.firstChild);
    
    // Add focus management
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });
    
    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });
}

// Performance monitoring
function trackPagePerformance() {
    // Track page load time
    window.addEventListener('load', function() {
        const loadTime = performance.now();
        console.log('Page loaded in', loadTime.toFixed(2), 'milliseconds');
        
        // Send analytics if needed
        if (window.gtag) {
            gtag('event', 'page_load', {
                'load_time': loadTime
            });
        }
    });
    
    // Track button clicks
    const launchButton = document.getElementById('launch-portal');
    if (launchButton) {
        launchButton.addEventListener('click', function() {
            if (window.gtag) {
                gtag('event', 'portal_launch', {
                    'button_text': 'Launch Screening Portal'
                });
            }
        });
    }
}

// Error handling
function setupErrorHandling() {
    window.addEventListener('error', function(event) {
        console.error('JavaScript error:', event.error);
        
        // Show user-friendly error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = 'An error occurred. Please refresh the page.';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            z-index: 1000;
        `;
        
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    });
}

// Initialize everything
trackPagePerformance();
setupErrorHandling();

// Add loading styles
const loadingStyles = `
    .launch-button.loading {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        cursor: not-allowed;
    }
    
    .launch-button.loading .button-icon {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .skip-link {
        position: absolute;
        top: -40px;
        left: 0;
        background: #1f2937;
        color: white;
        padding: 0.5rem 1rem;
        text-decoration: none;
        border-radius: 0 0 0.5rem 0;
        z-index: 1000;
        transition: all 0.2s ease;
    }
    
    .skip-link:focus {
        top: 0;
    }
    
    .keyboard-navigation *:focus {
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
    }
    
    .error-message {
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;

// Add loading styles to head
const styleSheet = document.createElement('style');
styleSheet.textContent = loadingStyles;
document.head.appendChild(styleSheet);
