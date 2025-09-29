# Market_Research_Assistant_AI_Model
## Problem Statement

The biggest challenge in the gaming industry is the inaccessibility and fragmentation of up-to-date market data and analysis. Business intelligence requires hours of manual searching, document analysis, and data aggregation.

## Solution
The Gaming Research Agent is a multimodal (Voice/Text/Documents) AI-powered assistant that provides instant, consolidated, and multi-source market research in seconds.

It helps analysts, entrepreneurs, and investors uncover trends, valuations, and actionable insights with speed and precision—eliminating the burden of fragmented manual research.

## Features

- **Voice Input**: Speak your questions instead of typing
- **Text-to-Speech**: Listen to AI responses aloud
- **PDF Processing**: Upload and analyze gaming market reports
- **Web Search**: Real-time search for the latest gaming trends
- **Conversational AI**: Powered by Groq's Llama model
- **Gaming Theme**: Dark theme with gaming aesthetics

## WorkFlow:
flowchart TD
    A[User Input] -->|Text or Voice| B[Input Handler]
    B --> C{Data Aggregation}
    C -->|PDF Uploaded| D[PDF Parsing & Chunking]
    C -->|Web Search| E[DuckDuckGo Search]
    D --> F[Embeddings via HuggingFace]
    E --> F
    F --> G[Chroma DB Vector Store]
    G --> H[LangChain Conversational Retrieval Chain]
    H --> I[Groq LLM Reasoning]
    I --> J{Output Generation}
    J -->|Text| K[Streamlit Chat UI]
    J -->|Voice| L[gTTS Speech Output]
    J -->|Numerical Data| M[Matplotlib Visualization]


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


##  Business Value

The Gaming Research Agent transforms market intelligence by:

 - Saving hours of manual research
 - Delivering consolidated, actionable insights
 - Offering real-time and document-based intelligence
 - Giving businesses a competitive edge in gaming markets
   
## Contact & Collaboration

I’d love to hear from you! 🚀
This project was a joint collarabration:
- Evalyn Njagi
- Noelle Wambui
- Rose Wabere
  
If you’re interested in  contributing, or discussing ideas related to gaming market research and AI-powered analytics kindly reach out at:

 Email: evalynnjagi02@gmail.com, noellemburu@gmail.com, rozw@gmail.com


