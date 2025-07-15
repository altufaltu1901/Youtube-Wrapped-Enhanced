import re
from typing import Tuple, Optional
import os
import json as pyjson
import string
import numpy as np

# Query types
AGGREGATE = 'aggregate'  # most watched channel/video, possibly with year
OPEN_ENDED = 'open_ended'  # open-ended question
HYBRID = 'hybrid'  # most watched about X

# --- LLM-based classification using Together API ---
def classify_query_llm(query: str) -> tuple:
    from together import Together
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing environment variable TOGETHER_API_KEY.\n"
            "Please set it before running, e.g.:\n"
            "  export TOGETHER_API_KEY='YOUR_KEY_HERE' (macOS/Linux)\n"
            "  setx TOGETHER_API_KEY \"YOUR_KEY_HERE\" (Windows)"
        )
    client = Together(api_key=api_key)
    prompt = f"""
Return a JSON object with fields: type, year, topic, number.\
the type to enter will be the number of the type in the list
the year will only be a number if the user says something like this year put thenumber of the year
also for number only put number and try to figure out the number from the query
(1) aggregate (most watched channel/video, possibly with year, top n most watched), for this in topic put channel or video just the words, and number (top n)if in query and put year if in query
(2) open-ended(any question taht doesnt ask about aggregation like top or most ), leave everything empty in json except type and topic in topic mention whats it about (like video about X topic will be X)
(3) hybrid (most watched about X). for this in topic put the topic the query is about and number if in query and year if in query
(4) total (total videos watched/cannels)  for this in topic put the topic put channel or video just the words
Also extract the year (if present) and the topic (if present).
Query: '{query}'
"""
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        result = pyjson.loads(response.choices[0].message.content)
        return result['type'], result.get('year'), result.get('topic'), result.get('number')
    except Exception as e:
        print("LLM classification error:", e)
        print("Raw LLM response:", response.choices[0].message.content)
        return OPEN_ENDED, None, None, None

# --- Gemini embedding and LLM for open-ended queries ---
def get_gemini_embedding(text, model="models/embedding-001"):
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing environment variable GOOGLE_API_KEY.\n"
            "Please set it before running, e.g.\n"
            "  export GOOGLE_API_KEY='YOUR_KEY_HERE' (macOS/Linux)\n"
            "  setx GOOGLE_API_KEY \"YOUR_KEY_HERE\" (Windows)"
        )
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_query"
    )
    return response['embedding']

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    user_query = input("Enter your YouTube analytics query: ")
    qtype, year, topic, number = classify_query_llm(user_query)
    # print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")

    if qtype == 1:
        print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
        with open("altamash.json", "r", encoding="utf-8") as f:
            total_data = pyjson.load(f)
            # Filter by year if present
            if year:
                filtered_data = [entry for entry in total_data if str(year) in entry.get("timestamp", "")]
            else:
                filtered_data = total_data
            from collections import Counter
            n = int(number) if number and str(number).isdigit() else 5
            if topic and topic.lower() == "video":
                video_counter = Counter(entry["video_title"] for entry in filtered_data)
                print(f"Top {n} most watched videos{f' in {year}' if year else ''}:")
                for video, count in video_counter.most_common(n):
                    print(f"{video}: {count} times")
            elif topic and topic.lower() == "channel":
                channel_counter = Counter(entry["channel_name"] for entry in filtered_data)
                print(f"Top {n} most watched channels{f' in {year}' if year else ''}:")
                for channel, count in channel_counter.most_common(n):
                    print(f"{channel}: {count} times")
            else:
                print("Could not determine whether to aggregate videos or channels from the query topic.")
    elif qtype == 2:
        print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        
        genai.configure(api_key=api_key)
        # Load embedded data
        with open("altamash_with_embeddings.json", "r", encoding="utf-8") as f:
            embedded_data = pyjson.load(f)
        # Get embedding for the user query
        query_emb = get_gemini_embedding(topic)
        # Compute similarities
        scored = []
        for entry in embedded_data:
            sim = cosine_similarity(query_emb, entry["embedding"])
            scored.append((sim, entry))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_entries = [entry for _, entry in scored[:50]]
        # Prepare context for Gemini LLM
        context = "\n".join(f"{e['timestamp']} - {e['video_title']} ({e['channel_name']})" for e in top_entries)
        # Send to Gemini LLM
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = f"""
You are a YouTube analytics assistant. Here is some of my YouTube watch history:
{context}

Answer the following question about my YouTube history:
{user_query}
"""
        response = model.generate_content(prompt)
        print("\nAnswer:")
        print(response.text)
    elif qtype == 3:
        print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
    elif qtype == 4:
        print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
        with open("altamash.json", "r", encoding="utf-8") as f:
            total_data = pyjson.load(f)
            # Filter by year if present
            if year:
                filtered_data = [entry for entry in total_data if str(year) in entry.get("timestamp", "")]
            else:
                filtered_data = total_data
            if topic and topic.lower() == "video":
                unique_videos = set(entry["video_title"] for entry in filtered_data)
                print(f"Total videos watched{f' in {year}' if year else ''}: {len(unique_videos)}")
            elif topic and topic.lower() == "channel":
                unique_channels = set(entry["channel_name"] for entry in filtered_data)
                print(f"Total unique channels watched{f' in {year}' if year else ''}: {len(unique_channels)}")
            else:
                print("Could not determine whether to count videos or channels from the query topic.")

