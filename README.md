# 📈 YouTube-Wrapped-Enhanced

**YouTube Wrapped Analytics with LLM & RAG**

This project lets you analyze your YouTube watch history in a powerful, interactive way—just like a personal "YouTube Wrapped," but with the ability to ask any question you want!  
It combines **data extraction**, **semantic search**, and **Large Language Models (LLMs)** to answer both statistical and open-ended questions about your YouTube habits.

---

## 📂 Project Files

### `youtubewrapped.py`
- Parses your YouTube watch history HTML (from Google Takeout).
- Extracts video, channel, and timestamp info.
- Embeds each entry using Gemini and saves to JSON.

### `queryproc.py`
- Lets you ask questions about your history, such as:
  - "Top 5 channels in 2023"
  - "What did I watch about AI?"
  - "Total videos in 2022"
- Uses LLM-based classification and RAG (Retrieval-Augmented Generation) to answer.

---

## ⚙️ How It Works

### `youtubewrapped.py`
- Parses your YouTube watch history HTML.
- Extracts video/channel/timestamp info.
- Generates text embeddings for each video using Gemini.
- Saves embeddings to JSON for later querying.

### `queryproc.py`
- Classifies your question (aggregate, total, open-ended, hybrid).
- For aggregate/total: uses fast counting and filtering.
- For open-ended: 
  - Finds the top 50 most semantically relevant entries using embeddings.
  - Passes them plus your question to Gemini LLM for contextual answering.

---

## ✨ Features
- 📜 Automatic parsing of YouTube watch history HTML.
- 🧠 Embeds each video entry using Gemini for semantic search.
- 💬 Natural language Q&A:
  - Ask about your most-watched channels, videos, or open-ended questions.
- 📊 Supports aggregate, open-ended, hybrid, and total queries:
  - "Top 10 videos in 2023"
  - "Total channels"
  - "What did I watch about Python?"
- 🔎 Retrieval-Augmented Generation:
  - For open-ended questions, finds the most relevant entries and uses an LLM to answer in context.
