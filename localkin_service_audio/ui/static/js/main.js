// LocalKin Service Audio Web Interface - Main JavaScript

// Global variables
let statusInterval;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set up global error handling
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showError('An unexpected error occurred. Please refresh the page.');
    });

    // Initialize status checking
    checkStatus();

    console.log('LocalKin Service Audio Web Interface initialized');
}

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.status === 'running') {
            updateStatusIndicator('online');
            console.log('System status:', data);
        } else {
            updateStatusIndicator('offline');
        }
    } catch (error) {
        console.error('Status check failed:', error);
        updateStatusIndicator('error');
    }
}

function updateStatusIndicator(status) {
    // Update navbar status indicator if it exists
    const statusBtn = document.querySelector('.navbar .btn-outline-light');
    if (statusBtn) {
        statusBtn.className = `btn btn-outline-${getStatusColor(status)}`;
        statusBtn.innerHTML = `<i class="fas ${getStatusIcon(status)}"></i>`;
    }
}

function getStatusColor(status) {
    switch (status) {
        case 'online': return 'success';
        case 'offline': return 'danger';
        case 'error': return 'warning';
        default: return 'secondary';
    }
}

function getStatusIcon(status) {
    switch (status) {
        case 'online': return 'fa-check-circle';
        case 'offline': return 'fa-times-circle';
        case 'error': return 'fa-exclamation-triangle';
        default: return 'fa-question-circle';
    }
}

// Utility functions
function showError(message) {
    const alert = document.getElementById('errorAlert');
    const messageSpan = document.getElementById('errorMessage');

    if (alert && messageSpan) {
        messageSpan.textContent = message;
        alert.style.display = 'block';
        alert.scrollIntoView({ behavior: 'smooth' });
    } else {
        alert(message);
    }
}

function hideError() {
    const alert = document.getElementById('errorAlert');
    if (alert) {
        alert.style.display = 'none';
    }
}

function showSuccess(message) {
    // Create a temporary success alert
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show position-fixed';
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alert);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatConfidence(confidence) {
    if (typeof confidence === 'number') {
        return (confidence * 100).toFixed(1) + '%';
    }
    return 'N/A';
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showSuccess('Text copied to clipboard!');
    }).catch(() => {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showSuccess('Text copied to clipboard!');
    });
}

function downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Export utilities for use in other scripts
window.LocalKinAudioUtils = {
    showError,
    hideError,
    showSuccess,
    formatFileSize,
    formatConfidence,
    copyToClipboard,
    downloadBlob
};
