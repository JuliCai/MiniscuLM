import os
import json
import argparse
import shutil
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate synthetic training data.")
parser.add_argument("-resume", action="store_true", help="Resume from where left off")
parser.add_argument("-overwrite", action="store_true", help="Overwrite existing data")
args = parser.parse_args()

# Configuration
TRAINING_DATA_DIR = "Training_Data"
TOPICS_FILE = os.path.join(TRAINING_DATA_DIR, "topics.json")

# API Setup
try:
    githubaikey = open("githubkey.apikeys").read().strip()
except FileNotFoundError:
    print("Error: githubkey.apikeys file not found.")
    exit(1)

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4.1-mini"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(githubaikey),
    api_version="2024-12-01-preview",
)

def get_completion(system_prompt, user_prompt, model=model_name):
    try:
        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            temperature=0.7,
            top_p=1,
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API: {e}")
        time.sleep(5) # Wait and retry logic could be added here
        return None

def parse_json_response(response_text):
    try:
        # Strip markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {response_text}")
        return None

def main():
    # Handle Resume/Overwrite logic
    if os.path.exists(TRAINING_DATA_DIR):
        if args.overwrite:
            print("Overwriting existing Training Data...")
            shutil.rmtree(TRAINING_DATA_DIR)
            os.makedirs(TRAINING_DATA_DIR)
        elif args.resume:
            print("Resuming generation...")
        else:
            # Check if directory is empty
            if os.listdir(TRAINING_DATA_DIR):
                print(f"Warning: '{TRAINING_DATA_DIR}' already exists and is not empty.")
                print("Use -resume to continue or -overwrite to restart.")
                exit(1)
    else:
        os.makedirs(TRAINING_DATA_DIR)

    # 1. Generate Topics
    topics = []
    if args.resume and os.path.exists(TOPICS_FILE):
        with open(TOPICS_FILE, 'r') as f:
            topics = json.load(f)
            print(f"Loaded {len(topics)} topics from file.")
    
    if not topics:
        print("Generating 10 general topics...")
        prompt = "Generate 10 general topics (eg. trivia, math, coding, writing etc.) that span a wide range of knowledge. Return ONLY valid JSON."
        json_format = '{"topics":["topic1","topic2"]}'
        response = get_completion(f"You are a helpful assistant. Respond in this exact json form: {json_format}", prompt, model="gpt-4.1")
        data = parse_json_response(response)
        if data and "topics" in data:
            topics = data["topics"][:10] # Ensure only 10
            with open(TOPICS_FILE, 'w') as f:
                json.dump(topics, f)
        else:
            print("Failed to generate topics.")
            exit(1)

    # 2. Loop over topics
    for topic in topics:
        topic_safe = "".join(x for x in topic if x.isalnum() or x in " -_").strip()
        topic_dir = os.path.join(TRAINING_DATA_DIR, topic_safe)
        os.makedirs(topic_dir, exist_ok=True)
        
        subtopics_file = os.path.join(topic_dir, "subtopics.json")
        subtopics = []

        if args.resume and os.path.exists(subtopics_file):
            with open(subtopics_file, 'r') as f:
                subtopics = json.load(f)
        
        if not subtopics:
            print(f"Generating 10 subtopics for '{topic}'...")
            prompt = f"Generate 10 different subtopics for the general topic '{topic}'. Ensure they cover a wide range within the topic. Return ONLY valid JSON."
            json_format = '{"topics":["subtopic1","subtopic2"]}' # Reusing the key "topics" as per prompt instruction "in the same form as before"
            response = get_completion(f"You are a helpful assistant. Respond in this exact json form: {json_format}", prompt, model="gpt-4.1")
            data = parse_json_response(response)
            if data and "topics" in data:
                subtopics = data["topics"][:10]
                with open(subtopics_file, 'w') as f:
                    json.dump(subtopics, f)
            else:
                print(f"Failed to generate subtopics for {topic}. Skipping...")
                continue

        # 3. Loop over subtopics
        for subtopic in subtopics:
            subtopic_safe = "".join(x for x in subtopic if x.isalnum() or x in " -_").strip()
            subtopic_dir = os.path.join(topic_dir, subtopic_safe)
            os.makedirs(subtopic_dir, exist_ok=True)

            # Check how many files we have
            existing_files = [f for f in os.listdir(subtopic_dir) if f.endswith(".txt")]
            if len(existing_files) >= 80:
                print(f"Skipping {topic}/{subtopic} (already has {len(existing_files)} files).")
                continue

            print(f"Generating examples for '{topic}' -> '{subtopic}'...")
            
            # Generate all 100 at once to avoid duplicates and ensure uniqueness
            # Check if we already have 80 files
            existing_files = [f for f in os.listdir(subtopic_dir) if f.endswith(".txt")]
            if len(existing_files) >= 80:
                print(f"Skipping {topic}/{subtopic} (already has {len(existing_files)} files).")
                continue

            print(f"  Generating 100 examples...")
            
            length_instruction = "Keep the examples very concise (around 10 tokens)."
            if "writing" in topic.lower():
                 length_instruction = "Keep the examples concise but slightly longer (about 20-40 tokens)."

            prompt = f"Generate 100 unique question/response pairs for the subtopic '{subtopic}' (under general topic '{topic}'). {length_instruction} The questions should vary in difficulty and style. Return ONLY valid JSON."
            json_format = '{"examples":[{"question":"...","answer":"..."}, ...]}'
            
            response = get_completion(f"You are a helpful assistant. Respond in this exact json form: {json_format}", prompt)
            data = parse_json_response(response)
            
            if data and "examples" in data:
                examples = data["examples"]
                print(f"  Generated {len(examples)} examples.")
                # Save examples
                for i, example in enumerate(examples):
                    if i >= 100: break
                    filename = f"{topic_safe}_{subtopic_safe}_{i}.txt"
                    filepath = os.path.join(subtopic_dir, filename)
                    
                    # Save as JSON
                    content = json.dumps({
                        "question": example.get('question', ''),
                        "answer": example.get('answer', '')
                    }, indent=4)
                    
                    with open(filepath, 'w') as f:
                        f.write(content)
            else:
                print(f"  Failed to generate examples for {subtopic}.")


if __name__ == "__main__":
    main()
