# Youtube-Wrapped-Enhanced
YouTube Wrapped Analytics with LLM & RAG

Project Description

This project lets you analyze your YouTube watch history in a powerful, interactive way—just like a personal "YouTube Wrapped," but with the ability to ask any question you want!
It combines data extraction, semantic search, and large language models (LLMs) to answer both statistical and open-ended questions about your YouTube habits. 

youtubewrapped.py: Extracts and processes your YouTube watch history from Google Takeout, and generates text embeddings for each video using Gemini.

queryproc.py: Lets you ask questions about your history (e.g., "Top 5 channels in 2023", "What did I watch about AI?", "Total videos in 2022") and uses LLMs and RAG (Retrieval-Augmented Generation) to answer.

How it Works

youtubewrapped.py:
Parses your YouTube watch history HTML.
Extracts video/channel/timestamp info.
Embeds each entry using Gemini and saves to JSON.

queryproc.py:
Classifies your question (aggregate, total, open-ended, hybrid).
For aggregate/total: uses fast counting and filtering.
For open-ended: finds the top 50 most relevant entries using semantic search, then sends them (plus your question) to Gemini LLM for a contextual answer.
