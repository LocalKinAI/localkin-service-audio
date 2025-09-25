// LocalKin Service Audio Transcription Interface

let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeTranscription();
});

function initializeTranscription() {
    const fileInput = document.getElementById('audioFile');
    const transcribeBtn = document.getElementById('transcribeBtn');
    const modelSelect = document.getElementById('modelSelect');

    // File input handler
    fileInput.addEventListener('change', handleFileSelection);

    // Transcribe button handler
    transcribeBtn.addEventListener('click', startTranscription);

    // Model selection handler
    modelSelect.addEventListener('change', validateForm);

    console.log('Transcription interface initialized');
}

function handleFileSelection(event) {
    const file = event.target.files[0];
    selectedFile = file;

    if (file) {
        // Validate file size (50MB limit)
        const maxSize = 50 * 1024 * 1024; // 50MB in bytes
        if (file.size > maxSize) {
            LocalKinAudioUtils.showError('File size exceeds 50MB limit. Please choose a smaller file.');
            selectedFile = null;
            event.target.value = '';
            return;
        }

        // Validate file type
        const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/x-m4a', 'audio/flac', 'audio/ogg'];
        if (!allowedTypes.some(type => file.type.includes(type.split('/')[1]))) {
            LocalKinAudioUtils.showError('Unsupported file format. Please use WAV, MP3, M4A, FLAC, or OGG files.');
            selectedFile = null;
            event.target.value = '';
            return;
        }

        console.log('File selected:', file.name, LocalKinAudioUtils.formatFileSize(file.size));
    }

    validateForm();
}

function validateForm() {
    const transcribeBtn = document.getElementById('transcribeBtn');
    const hasFile = selectedFile !== null;
    const hasModel = document.getElementById('modelSelect').value !== '';

    transcribeBtn.disabled = !(hasFile && hasModel);
}

async function startTranscription() {
    if (!selectedFile) {
        LocalKinAudioUtils.showError('Please select an audio file first.');
        return;
    }

    const modelName = document.getElementById('modelSelect').value;
    if (!modelName) {
        LocalKinAudioUtils.showError('Please select a transcription model.');
        return;
    }

    // Hide error and results
    LocalKinAudioUtils.hideError();
    document.getElementById('resultsContainer').style.display = 'none';

    // Show progress
    showProgress(true, 'Uploading file...');

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_name', modelName);

        // Update progress
        updateProgress(25, 'Processing audio...');

        // Send transcription request
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        updateProgress(75, 'Analyzing results...');

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Transcription failed');
        }

        const result = await response.json();

        updateProgress(100, 'Complete!');
        setTimeout(() => {
            showProgress(false);
            displayResults(result);
        }, 500);

    } catch (error) {
        console.error('Transcription error:', error);
        showProgress(false);
        LocalKinAudioUtils.showError(error.message || 'Transcription failed. Please try again.');
    }
}

function showProgress(show, text = '') {
    const container = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    if (show) {
        container.style.display = 'block';
        progressText.textContent = text;
        progressBar.style.width = '0%';
    } else {
        container.style.display = 'none';
    }
}

function updateProgress(percent, text) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    progressBar.style.width = percent + '%';
    progressText.textContent = text;
}

function displayResults(result) {
    // Update result fields
    document.getElementById('resultModel').textContent = result.model || 'Unknown';
    document.getElementById('resultLanguage').textContent = result.language || 'Unknown';

    // Update confidence
    const confidencePercent = result.confidence ? (result.confidence * 100) : 0;
    document.getElementById('confidenceBar').style.width = confidencePercent + '%';
    document.getElementById('confidenceText').textContent = LocalKinAudioUtils.formatConfidence(result.confidence);

    // Update transcription text
    document.getElementById('transcriptionText').textContent = result.text || 'No transcription available';

    // Show results container
    document.getElementById('resultsContainer').style.display = 'block';
    document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });

    LocalKinAudioUtils.showSuccess('Transcription completed successfully!');
}

// Global functions for button handlers
function copyToClipboard() {
    const text = document.getElementById('transcriptionText').textContent;
    LocalKinAudioUtils.copyToClipboard(text);
}

function downloadText() {
    const text = document.getElementById('transcriptionText').textContent;
    const model = document.getElementById('resultModel').textContent;

    const blob = new Blob([text], { type: 'text/plain' });
    const filename = `transcription_${model}_${new Date().toISOString().split('T')[0]}.txt`;

    LocalKinAudioUtils.downloadBlob(blob, filename);
}
