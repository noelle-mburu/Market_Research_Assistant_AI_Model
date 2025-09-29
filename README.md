# Market_Research_Assistant_AI_Model
This is a project to create a Market Research Assistant AI Model that will gather, analyze, and predict trends in Roblox and other gaming platforms by retrieving public data from the internet.

It is a voice-enabled AI chatbot for gaming market research with real-time data analysis, PDF processing, and web search capabilities.

## Features

- **Voice Input**: Speak your questions instead of typing
- **Text-to-Speech**: Listen to AI responses aloud
- **PDF Processing**: Upload and analyze gaming market reports
- **Web Search**: Real-time search for the latest gaming trends
- **Conversational AI**: Powered by Groq's Llama model
- **Gaming Theme**: Dark theme with gaming aesthetics

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))
- Ngrok account ([Sign up here](https://ngrok.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd gaming-market-research-ai

2. **Install dependancies**
  ```bash
   pip install -r requirements.txt
```

3. **Set up environment variables**
Create a .env file:
  ```bash
GROQ_API_KEY=your_groq_api_key_here
ngrok_token=your_ngrok_token_here
```
## Usage
Option 1: Run locally
```bash
streamlit run app.py
```

Option 2: Run with public URL (ngrok)
```bash
python deploy.py
```


<img width="1357" height="644" alt="market_research_ai" src="https://github.com/user-attachments/assets/d2b4de38-565a-474a-af43-fb354cff3bcf" />

<img width="1352" height="644" alt="market_research_ai2" src="https://github.com/user-attachments/assets/ddb72b7b-d69d-4f0d-91bf-4a9fb7bc83c2" />
