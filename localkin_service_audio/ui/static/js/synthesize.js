// LocalKin Service Audio Speech Synthesis Interface

document.addEventListener('DOMContentLoaded', function() {
    initializeSynthesis();
});

function initializeSynthesis() {
    const textInput = document.getElementById('textInput');
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const modelSelect = document.getElementById('modelSelect');

    // Text input handler
    textInput.addEventListener('input', validateForm);

    // Synthesize button handler
    synthesizeBtn.addEventListener('click', startSynthesis);

    // Model selection handler
    modelSelect.addEventListener('change', validateForm);

    console.log('Synthesis interface initialized');
}

function validateForm() {
    const textInput = document.getElementById('textInput');
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const hasText = textInput.value.trim().length > 0;
    const hasModel = document.getElementById('modelSelect').value !== '';

    synthesizeBtn.disabled = !(hasText && hasModel);
}

async function startSynthesis() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) {
        LocalKinAudioUtils.showError('Please enter some text to synthesize.');
        return;
    }

    if (text.length > 5000) {
        LocalKinAudioUtils.showError('Text is too long. Maximum 5000 characters allowed.');
        return;
    }

    const modelName = document.getElementById('modelSelect').value;
    if (!modelName) {
        LocalKinAudioUtils.showError('Please select a TTS model.');
        return;
    }

    // Hide error and results
    LocalKinAudioUtils.hideError();
    document.getElementById('resultsContainer').style.display = 'none';

    // Show progress
    showProgress(true, 'Initializing TTS model...');

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('text', text);
        formData.append('model_name', modelName);

        // Update progress
        updateProgress(25, 'Generating speech...');

        // Send synthesis request
        const response = await fetch('/api/synthesize', {
            method: 'POST',
            body: formData
        });

        updateProgress(75, 'Processing audio...');

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Speech synthesis failed');
        }

        const result = await response.json();

        updateProgress(100, 'Complete!');
        setTimeout(() => {
            showProgress(false);
            displayResults(result);
        }, 500);

    } catch (error) {
        console.error('Synthesis error:', error);
        showProgress(false);
        LocalKinAudioUtils.showError(error.message || 'Speech synthesis failed. Please try again.');
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
    document.getElementById('resultTextLength').textContent = `${result.text_length || 0} characters`;
    document.getElementById('resultFileSize').textContent = LocalKinAudioUtils.formatFileSize(result.file_size || 0);

    // Update audio player
    const audioSource = document.getElementById('audioSource');
    const audioPlayer = document.getElementById('audioPlayer');

    if (result.audio_url) {
        audioSource.src = result.audio_url;
        audioPlayer.load();
    }

    // Show results container
    document.getElementById('resultsContainer').style.display = 'block';
    document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });

    LocalKinAudioUtils.showSuccess('Speech generated successfully!');
}

// Global functions for button handlers
function playAudio() {
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.play();
}

function downloadAudio() {
    const audioSource = document.getElementById('audioSource');
    if (audioSource.src) {
        // Create download link
        const a = document.createElement('a');
        a.href = audioSource.src;
        a.download = `speech_${new Date().toISOString().split('T')[0]}.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

function copyUrl() {
    const audioSource = document.getElementById('audioSource');
    if (audioSource.src) {
        LocalKinAudioUtils.copyToClipboard(audioSource.src);
    }
}

function setExampleText(text) {
    document.getElementById('textInput').value = text;
    validateForm();
}
