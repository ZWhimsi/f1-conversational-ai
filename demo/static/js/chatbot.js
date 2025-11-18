// Chatbot JavaScript
const chatbotToggle = document.getElementById('chatbotToggle');
const chatbotContainer = document.getElementById('chatbotContainer');
const closeChatbot = document.getElementById('closeChatbot');
const chatbotInput = document.getElementById('chatbotInput');
const sendButton = document.getElementById('sendButton');
const chatbotMessages = document.getElementById('chatbotMessages');

// Toggle chatbot
chatbotToggle.addEventListener('click', () => {
    chatbotContainer.classList.toggle('active');
    if (chatbotContainer.classList.contains('active')) {
        chatbotInput.focus();
    }
});

// Close chatbot
closeChatbot.addEventListener('click', () => {
    chatbotContainer.classList.remove('active');
});

// Send message
function sendMessage() {
    const message = chatbotInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    chatbotInput.value = '';
    sendButton.disabled = true;
    sendButton.textContent = 'Sending...';
    
    // Show typing indicator
    const typingIndicator = addMessage('Thinking...', 'bot');
    
    // Send to API
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove typing indicator
        typingIndicator.remove();
        
        if (data.status === 'success') {
            addMessage(data.response, 'bot');
        } else {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    })
    .catch(error => {
        typingIndicator.remove();
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        console.error('Error:', error);
    })
    .finally(() => {
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        chatbotInput.focus();
    });
}

// Add message to chat
function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const p = document.createElement('p');
    p.textContent = text;
    messageDiv.appendChild(p);
    
    chatbotMessages.appendChild(messageDiv);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    
    return messageDiv;
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

chatbotInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !sendButton.disabled) {
        sendMessage();
    }
});

// Check API health on load
fetch('/api/health')
    .then(response => response.json())
    .then(data => {
        if (!data.model_loaded) {
            addMessage('⚠️ Model not loaded. Please check the configuration.', 'bot');
        }
    })
    .catch(error => {
        console.error('Health check failed:', error);
    });

