# Dual-Agent Chatbot with LangGraph

A conversational AI system that intelligently routes user messages to specialized agents — a **Therapist** for emotional support or a **Logical Assistant** for analytical responses — using [LangGraph](https://github.com/langchain-ai/langgraph) and Google's Gemini model.

---

## How It Works

The graph pipeline processes each message through 4 nodes:
```
User Input → Classifier → Router → Therapist Agent
                                 ↘ Logical Agent
```

1. **Classifier Agent** — Detects whether the message is `emotional` or `logical` using structured output (Pydantic).
2. **Router Agent** — Directs the flow to the appropriate agent based on the classification.
3. **Therapist Agent** — Responds with empathy and compassion to emotional messages.
4. **Logical Agent** — Responds with clear, analytical reasoning to logical messages.

---

## Project Structure
```
.
├── .env                  # Your API keys (not committed)
├── .env.example          # Template for environment variables
├── .gitignore
├── LICENSE
├── main.py               # Main application logic
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your API key:
```bash
cp .env.example .env
```
```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Usage
```bash
python main.py
```

Then type any message at the prompt:
```
Message: I've been feeling really overwhelmed lately.
Assistant (Therapist): It sounds like you're carrying a lot right now...

Message: What is the time complexity of quicksort?
Assistant (Logical): Quicksort has an average time complexity of O(n log n)...

Message: exit
```

Type `exit` to quit the session.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `langgraph` | Agent graph orchestration |
| `langchain` | LLM abstraction & message types |
| `langchain-google-genai` | Gemini model integration |
| `pydantic` | Structured output / message classification |
| `python-dotenv` | Environment variable management |

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
