(function () {
    const POLL_INTERVAL_MS = 10000;
    let redirecting = false;

    async function checkSessionStatus() {
        if (redirecting) return;

        try {
            const response = await fetch('/api/session/status', {
                method: 'GET',
                credentials: 'same-origin',
                cache: 'no-store'
            });

            if (response.status === 401 || response.status === 403) {
                redirecting = true;
                window.location.href = '/login';
                return;
            }

            const data = await response.json();
            if (!data.authenticated || data.account_status === 'inactive') {
                redirecting = true;
                window.location.href = '/login';
            }
        } catch (error) {
            console.warn('Session polling failed:', error);
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        checkSessionStatus();
        setInterval(checkSessionStatus, POLL_INTERVAL_MS);
    });
})();
