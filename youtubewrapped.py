from bs4 import BeautifulSoup
import json
import time
from collections import Counter, defaultdict
import re

INPUT_FILE = "watch-history.html"
OUTPUT_FILE = "altamash.json"

# print("Starting to read HTML file...")
with open(INPUT_FILE, encoding="utf-8") as f:
    html = f.read()

# print(f"HTML file loaded ({len(html)} characters)")
# print("Parsing HTML with BeautifulSoup...")
soup = BeautifulSoup(html, "lxml")

# print("Searching for video entries...")
outer_cells = soup.find_all("div", class_="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp")
# print(f"Found {len(outer_cells)} outer cells to process")

results = []
monthly_counter = defaultdict(int)

for outer in outer_cells:
    main_cells = outer.find_all("div", class_="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1")
    if not main_cells:
        continue

    cell = main_cells[0]
    links = cell.find_all("a")
    if len(links) < 2:
        continue

    video_link = links[0]
    channel_link = links[1]

    # Extract text and URLs
    video_title = video_link.get_text(strip=True)
    video_url = video_link.get("href", "")
    channel_name = channel_link.get_text(strip=True)
    channel_url = channel_link.get("href", "")

    
    cell_text = cell.get_text(strip=True)
    lines = [line.strip() for line in cell_text.split('\n') if line.strip()]
    
    timestamp = ""
    for line in reversed(lines):
        if any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
            timestamp = line
            break

    
    month_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', timestamp)
    if month_match:
        month = month_match.group(1)
        monthly_counter[month] += 1

    results.append({
        "video_title": video_title,
        "video_url": video_url,
        "channel_name": channel_name,
        "channel_url": channel_url,
        "timestamp": timestamp
    })

# print(f"Processing complete! Found {len(results)} video entries.")

# # Count most watched channels and videos
# channel_counter = Counter(entry["channel_name"] for entry in results)
# video_counter = Counter(entry["video_title"] for entry in results)

# print("\nTop 10 Most Watched Channels:")
# for channel, count in channel_counter.most_common(10):
#     print(f"{channel}: {count} times")

# print("\nTop 10 Most Watched Videos:")
# for video, count in video_counter.most_common(10):
#     print(f"{video}: {count} times")

# print("\nMonthly video count (all years combined):")
# months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# for month in months_order:
#     print(f"{month}: {monthly_counter[month]}")

# Write all results to JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
# print(f"\nAll results written to {OUTPUT_FILE}")


import google.generativeai as genai
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="api_key")  

EMBEDDING_INPUT_FILE = OUTPUT_FILE
EMBEDDING_OUTPUT_FILE = "altamash_with_embeddings.json"

with open(EMBEDDING_INPUT_FILE, "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

def get_gemini_embedding(text, model="models/embedding-001"):
    while True:
        try:
            response = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            print("Error:", e)
            time.sleep(5)

for i, entry in enumerate(embedding_data):
    text = entry['video_title'] + ' ' + entry['channel_name']
    entry['embedding'] = get_gemini_embedding(text)
    if (i+1) % 10 == 0:
        # print(f"Embedding processed {i+1}/{len(embedding_data)} entries")

with open(EMBEDDING_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, ensure_ascii=False, indent=2)

# print(f"\nEmbeddings added and saved to {EMBEDDING_OUTPUT_FILE}")