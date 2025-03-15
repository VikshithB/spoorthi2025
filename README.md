# Spoorthi Chatbot

This project is a real-time interactive voice chatbot that integrates dynamic event information from Google Docs with an AI backend. It leverages OpenAI's Whisper (medium.en) for speech-to-text transcription, a FAISS vector store (using LangChain Community embeddings) for efficient document retrieval, and a custom Ollama LLM wrapper to generate responses. The final response is converted to speech using ElevenLabs TTS and played via mpv. A simple Tkinter-based UI displays the real-time conversation.

## Features

- **Real-Time Voice Input:** Continuously records and transcribes speech with Whisper (medium.en) at 48 kHz.
- **Dynamic Knowledge Base:** Retrieves event information from a Google Doc using the Google Docs API.
- **Efficient Retrieval:** Uses FAISS with HuggingFace embeddings for similarity search.
- **AI-Powered Responses:** Generates concise responses using a custom Ollama LLM wrapper.
- **Text-to-Speech (TTS):** Converts final answers to speech using ElevenLabs TTS, played with mpv.
- **Session-Based Conversation:** Begins with a greeting ("hi", "hello", or "namaste") and maintains a conversation session. The session ends if no input is received for a set period or if empty inputs accumulate.
- **Graphical User Interface (GUI):** A Tkinter-based UI displays real-time transcriptions and agent responses.

## Requirements

### Python Packages

Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt