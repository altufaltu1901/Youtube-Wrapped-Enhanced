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
(1) aggregate (most watched channel/video can ONLY be channel or video, possibly with year, top n most watched), for this in topic put channel or video just the words, and number (top n)if in query and put year if in query
(2) open-ended(any question other than the rest ), leave everything empty in json except(year can be mentioned) type and topic in topic mention whats it about( if there isnt a theme and they are asking about their history in general put topic as general)
(3) hybrid (most watched about X(ONLY quantitive queries about topic or else go type 2)). for this in topic put the topic the query is about and number if in query and year if in query
(4) total (total videos watched/cannels)  for this in topic put the topic put channel or video just the words
(5) wrapped(anything about wrapped or summary (or what did i watch/tell me about my usage)as in like everything about youtube)  year if in query only the number
Also extract the year (if present) and the topic (if present).
DO NOT CHOOSE TYPE 1,4 IF ITS NOT ABOUT CHANNEL OR VIDEO
Current year is 2025
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

def answer_general_query_with_latest_videos(query, n=100,year=2025):
    """
    Use the top N latest videos (by timestamp order in altamash.json) as context and send to Together LLM for an answer.
    """
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
    with open("altamash.json", "r", encoding="utf-8") as f:
        data = pyjson.load(f)
    if year:
                filtered_data = [entry for entry in data if str(year) in entry.get("timestamp", "")]
    else:
                filtered_data = data
    latest_entries = filtered_data[0:n]
    context = "\n".join(f"{e['timestamp']} - {e['video_title']} ({e['channel_name']})" for e in latest_entries)
    prompt = f"""
You are a YouTube analytics assistant. Here is some of my latest YouTube watch history:
{context}
Dont output any context only the answer
Answer the following question about my YouTube history:
{query}
"""
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\nAnswer:")
    print(response.choices[0].message.content)

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
    elif qtype == 2 or qtype==3:
        print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
        from together import Together
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        subset_answer=""
        if(topic!="general"):
            subset_prompt = f"Is the following question about my YouTube history in general, or about a specific subset/topic? Answer with 'general' or 'subset' and if subset, also provide the topic.\nQuestion: {user_query}"
            subset_response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": subset_prompt}]
            )

            subset_answer = subset_response.choices[0].message.content.strip().lower()
            print(subset_answer)
        else:
            subset_answer=topic
            
        if subset_answer.startswith("general"):
            answer_general_query_with_latest_videos(user_query)
        else:
           
            topic_extracted = topic
            if "topic:" in subset_answer:
                topic_extracted = subset_answer.split("topic:", 1)[-1].strip()
            with open("altamash_with_embeddings.json", "r", encoding="utf-8") as f:
                embedded_data = pyjson.load(f)
            query_emb = get_gemini_embedding(topic_extracted)
            scored = []
            for entry in embedded_data:
                sim = cosine_similarity(query_emb, entry["embedding"])
                scored.append((sim, entry))
            scored.sort(reverse=True, key=lambda x: x[0])
            top_entries = [entry for _, entry in scored[:50]]
            # Prepare context for Gemini LLM
            client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
            subset_answer=""
            context = "\n".join(f"{e['timestamp']} - {e['video_title']} ({e['channel_name']})" for e in top_entries)
            subset_prompt = f"""
# You are a YouTube analytics assistant. Here is some of my YouTube watch history:
# {context}

# Answer the following question about my YouTube history:
# {user_query}
# """
            subset_response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": subset_prompt}]
            )

            print(subset_response.choices[0].message.content)
#             print(subset_answer)
#             import google.generativeai as genai
#             api_key = os.getenv("GOOGLE_API_KEY")
#             genai.configure(api_key=api_key)
#             context = "\n".join(f"{e['timestamp']} - {e['video_title']} ({e['channel_name']})" for e in top_entries)
#             model = genai.GenerativeModel('gemini-1.5-pro')
#             prompt = f"""
# You are a YouTube analytics assistant. Here is some of my YouTube watch history:
# {context}

# Answer the following question about my YouTube history:
# {user_query}
# """
#             response = model.generate_content(prompt)
#             print("\nAnswer:")
#             print(response.text)
    # elif qtype == 3:
    #     print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
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
    elif qtype == 5:
        print(f"Type: {qtype}\nYear: {year}\nTopic: {topic}\nNumber: {number}")
        with open("altamash.json", "r", encoding="utf-8") as f:
            total_data = pyjson.load(f)
            # Filter by year if present
            if year:
                filtered_data = [entry for entry in total_data if str(year) in entry.get("timestamp", "")]
            else:
                filtered_data = total_data
            from collections import Counter
            video_counter = Counter(entry["video_title"] for entry in filtered_data)
            channel_counter = Counter(entry["channel_name"] for entry in filtered_data)
            unique_videos = set(entry["video_title"] for entry in filtered_data)
            unique_channels = set(entry["channel_name"] for entry in filtered_data)
            print(f"\nYouTube Wrapped Summary{f' for {year}' if year else ''}:")
            print(f"Top 5 Most Watched Videos:")
            for video, count in video_counter.most_common(5):
                print(f"  {video}: {count} times")
            print(f"\nTop 5 Most Watched Channels:")
            for channel, count in channel_counter.most_common(5):
                print(f"  {channel}: {count} times")
            print(f"\nTotal Unique Videos Watched: {len(unique_videos)}")
            print(f"Total Unique Channels Watched: {len(unique_channels)}")
            print("Top 5 Genres Watched:")
            prompt="You are given the 100 latest watched videos in the year give me the top 5 genres with one sentence resoning"
            answer_general_query_with_latest_videos(prompt,100,year)

