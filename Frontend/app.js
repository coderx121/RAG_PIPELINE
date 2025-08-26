const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const fileUpload = document.getElementById("file-upload");
const chatBox = document.getElementById("chat-box");
const themeToggle = document.getElementById("theme-toggle");

// Load chat history from localStorage
function loadHistory() {
  const history = JSON.parse(localStorage.getItem("ragChatHistory") || "[]");
  history.forEach(msg => addMessage(msg.text, msg.sender));
}

// Save message to history
function saveToHistory(text, sender) {
  const history = JSON.parse(localStorage.getItem("ragChatHistory") || "[]");
  history.push({ text, sender });
  localStorage.setItem("ragChatHistory", JSON.stringify(history));
}

// Theme toggle
themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  themeToggle.textContent = document.body.classList.contains("dark") ? "☀️" : "🌙";
});

// Clear history button (assuming added to HTML)
document.getElementById('clear-history').addEventListener('click', () => {
  localStorage.removeItem('ragChatHistory');
  chatBox.innerHTML = '';
});

// Handle form submit
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const userMessage = input.value.trim();
  const file = fileUpload.files[0];
  if (!userMessage && !file) return;

  // Display user message (include file name if uploaded)
  const displayMessage = file ? `${userMessage} (Attached: ${file.name})` : userMessage;
  addMessage(displayMessage, "user");
  saveToHistory(displayMessage, "user");
  input.value = "";
  fileUpload.value = "";

  // Typing placeholder with stages
  const typingMsg = addMessage("Retrieving context...", "bot", true);

  try {
    const botResponse = await getBotResponse(userMessage, file);
    updateMessage(typingMsg, "Generating response...", "bot"); // Update to next stage
    await new Promise(r => setTimeout(r, 500)); // Simulate generation delay
    updateMessage(typingMsg, botResponse, "bot");
    saveToHistory(botResponse, "bot");
  } catch (error) {
    updateMessage(typingMsg, `Error: ${error.message}. Please try again.`, "bot");
  }
});

// Add message to chat
function addMessage(text, sender, isTyping = false) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  if (isTyping) msg.classList.add("typing");

  const avatar = document.createElement("div");
  avatar.classList.add("avatar");
  avatar.textContent = sender === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  if (isTyping) {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = marked.parse(text); // Markdown support
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);

  // Copy button for bot
  if (sender === "bot" && !isTyping) {
    const copyBtn = document.createElement("button");
    copyBtn.classList.add("copy-btn");
    copyBtn.innerHTML = "📋";
    copyBtn.onclick = () => navigator.clipboard.writeText(bubble.innerText);
    msg.appendChild(copyBtn);
  }

  // Timestamp
  const timestamp = document.createElement("div");
  timestamp.classList.add("timestamp");
  timestamp.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  msg.appendChild(timestamp);

  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;

  return msg;
}

// Update message (replace typing with bot response)
function updateMessage(msg, newText, sender) {
  msg.classList.remove("typing");
  const bubble = msg.querySelector(".bubble");
  bubble.innerHTML = marked.parse(newText);

  if (sender === "bot" && !msg.querySelector(".copy-btn")) {
    const copyBtn = document.createElement("button");
    copyBtn.classList.add("copy-btn");
    copyBtn.innerHTML = "📋";
    copyBtn.onclick = () => navigator.clipboard.writeText(bubble.innerText);
    msg.appendChild(copyBtn);
  }
}

// Backend response (with file support)
async function getBotResponse(query, file) {
  const formData = new FormData();
  formData.append("query", query);
  if (file) formData.append("file", file);

  try {
    const response = await fetch('http://localhost:8000/api/rag', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('API error:', error);
    return `Error: ${error.message}. Please check if the backend is running.`;
  }
}

// Load history on start
loadHistory();