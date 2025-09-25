// LocalKin Service Audio Speech Synthesis Interface

document.addEventListener('DOMContentLoaded', function() {
    initializeSynthesis();
});

function initializeSynthesis() {
    const textInput = document.getElementById('textInput');
    const fileInput = document.getElementById('fileInput');
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const modelSelect = document.getElementById('modelSelect');

    // Input type toggle
    document.getElementById('textInputRadio').addEventListener('change', toggleInputType);
    document.getElementById('fileInputRadio').addEventListener('change', toggleInputType);

    // Text input handler
    textInput.addEventListener('input', validateForm);

    // File input handler
    fileInput.addEventListener('change', handleFileSelect);

    // Synthesize button handler
    synthesizeBtn.addEventListener('click', startSynthesis);

    // Model selection handler
    modelSelect.addEventListener('change', function() {
        updateVoiceOptions();
        validateForm();
    });

    // Initialize voice options for default model
    updateVoiceOptions();

    console.log('Synthesis interface initialized');
}

function toggleInputType() {
    const isTextInput = document.getElementById('textInputRadio').checked;
    const textSection = document.getElementById('textInputSection');
    const fileSection = document.getElementById('fileInputSection');

    if (isTextInput) {
        textSection.style.display = 'block';
        fileSection.style.display = 'none';
        document.getElementById('fileInput').value = '';
        document.getElementById('fileInfo').style.display = 'none';
    } else {
        textSection.style.display = 'none';
        fileSection.style.display = 'block';
        document.getElementById('textInput').value = '';
    }

    validateForm();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        // Validate file type
        const allowedTypes = ['text/plain', 'text/markdown', 'text/rtf'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(txt|md|rtf)$/i)) {
            LocalKinAudioUtils.showError('Please select a valid text file (.txt, .md, .rtf)');
            event.target.value = '';
            return;
        }

        // Validate file size (1MB limit)
        if (file.size > 1024 * 1024) {
            LocalKinAudioUtils.showError('File size too large. Maximum 1MB allowed.');
            event.target.value = '';
            return;
        }

        // Read file content
        const reader = new FileReader();
        reader.onload = function(e) {
            const content = e.target.result;
            updateFileInfo(file, content);
        };
        reader.readAsText(file);
    } else {
        document.getElementById('fileInfo').style.display = 'none';
    }
    validateForm();
}

function updateFileInfo(file, content) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileChars = document.getElementById('fileChars');

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileChars.textContent = content.length;
    fileInfo.style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function updateVoiceOptions() {
    const modelName = document.getElementById('modelSelect').value;
    const voiceOptions = document.getElementById('voiceOptions');
    const voiceSelector = document.getElementById('voiceSelector');

    // Define voice options for different models
    const voiceConfigs = {
        'kokoro-82m': {
            label: 'Kokoro Voice',
            options: [
                { value: 'female', label: 'Female (af_sarah)' },
                { value: 'male', label: 'Male (am_adam)' },
                { value: 'af_sarah', label: 'Sarah (Female)' },
                { value: 'am_adam', label: 'Adam (Male)' }
            ]
        },
        'xtts-v2': {
            label: 'XTTS Speaker',
            options: [
                { value: 'Claribel Dervla', label: 'Claribel Dervla (Female)' },
                { value: 'Daisy Studious', label: 'Daisy Studious (Female)' },
                { value: 'Annie Angel', label: 'Annie Angel (Female)' },
                { value: 'Elizabeth', label: 'Elizabeth (Female)' },
                { value: 'Ella Dervla', label: 'Ella Dervla (Female)' },
                { value: 'Harry Potter', label: 'Harry Potter (Male)' },
                { value: 'Alan Rickman', label: 'Alan Rickman (Male)' },
                { value: 'Eric', label: 'Eric (Male)' },
                { value: 'Josh', label: 'Josh (Male)' }
            ]
        },
        'speecht5-tts': {
            label: 'SpeechT5 Speaker',
            options: [
                { value: 'default', label: 'Default Speaker' }
            ]
        },
        'tortoise-tts': {
            label: 'Tortoise Speaker',
            options: [
                { value: 'default', label: 'Default Speaker' }
            ]
        }
    };

    const config = voiceConfigs[modelName];
    if (config) {
        let html = `<label for="voiceSelect" class="form-label small">${config.label}</label>`;
        html += '<select class="form-select" id="voiceSelect">';

        config.options.forEach(option => {
            html += `<option value="${option.value}">${option.label}</option>`;
        });

        html += '</select>';
        voiceSelector.innerHTML = html;
        voiceOptions.style.display = 'block';
    } else {
        voiceSelector.innerHTML = '';
        voiceOptions.style.display = 'none';
    }
}

function validateForm() {
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const isTextInput = document.getElementById('textInputRadio').checked;
    const textInput = document.getElementById('textInput');
    const fileInput = document.getElementById('fileInput');
    const modelSelect = document.getElementById('modelSelect');

    let hasContent = false;
    if (isTextInput) {
        hasContent = textInput.value.trim().length > 0 && textInput.value.length <= 10000;
    } else {
        hasContent = fileInput.files && fileInput.files.length > 0;
    }

    const hasModel = modelSelect.value !== '';
    synthesizeBtn.disabled = !(hasContent && hasModel);
}

async function startSynthesis() {
    const isTextInput = document.getElementById('textInputRadio').checked;
    const text = document.getElementById('textInput').value.trim();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files ? fileInput.files[0] : null;

    if (isTextInput) {
        if (!text) {
            LocalKinAudioUtils.showError('Please enter some text to synthesize.');
            return;
        }

        if (text.length > 10000) {
            LocalKinAudioUtils.showError('Text is too long. Maximum 10000 characters allowed.');
            return;
        }
    } else {
        if (!file) {
            LocalKinAudioUtils.showError('Please select a file to upload.');
            return;
        }
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
    showProgress(true, 'Preparing synthesis request...');

    try {
        // Prepare form data
        const formData = new FormData();

        if (isTextInput) {
            formData.append('text', text);
        } else {
            formData.append('file', file);
        }

        formData.append('model_name', modelName);

        // Add voice selection if available
        const voiceSelect = document.getElementById('voiceSelect');
        if (voiceSelect && voiceSelect.value) {
            formData.append('voice', voiceSelect.value);
        }

        // Update progress
        updateProgress(25, 'Sending to TTS model...');

        // Send synthesis request
        const response = await fetch('/api/synthesize', {
            method: 'POST',
            body: formData
        });

        updateProgress(75, 'Generating speech...');

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Speech synthesis failed');
        }

        const result = await response.json();

        updateProgress(100, 'Synthesis complete!');
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
    document.getElementById('resultVoice').textContent = result.voice || 'default';
    document.getElementById('resultInputType').textContent = result.input_type || 'text';
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
