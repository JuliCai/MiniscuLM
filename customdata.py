import os
import json
import argparse
import time
import shutil
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate synthetic training data for a custom topic.")
parser.add_argument("subtopic", type=str, help="The name of the custom subtopic (e.g., 'greetings')")
parser.add_argument("context", type=str, nargs='?', help="Optional context to guide generation (e.g., 'the block based programming language')")
parser.add_argument("-casual", action="store_true", help="Generate casual conversation examples and overwrite existing")
parser.add_argument("-raw", action="store_true", help="Generate raw text data (articles, stories) instead of Q/A pairs")
args = parser.parse_args()

# Configuration
TRAINING_DATA_DIR = "Training_Data"
CUSTOM_TOPIC_NAME = "Custom"

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
        time.sleep(5)
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
    if args.raw:
        raw_dir = os.path.join(TRAINING_DATA_DIR, "Raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        filename_safe = "".join(x for x in args.subtopic if x.isalnum() or x in " -_").strip()
        filepath = os.path.join(raw_dir, f"{filename_safe}.txt")
        
        prompt_context = args.context if args.context else args.subtopic
        print(f"Generating raw text for '{filename_safe}.txt'...")
        
        prompt = f"Generate a long, coherent, and grammatically rich block of text about: '{prompt_context}'. It should be approximately 600-800 words long. Do not use headers or markdown formatting, just plain text paragraphs. Return ONLY the text."
        
        response = get_completion("You are a creative writer. Output only the requested text.", prompt)
        
        if response:
            with open(filepath, 'w') as f:
                f.write(response)
            print(f"  Saved raw text to {filepath}")
        else:
            print("  Failed to generate raw text.")
        return

    subtopic = args.subtopic
    topic = CUSTOM_TOPIC_NAME
    
    # Create directories
    topic_safe = topic
    subtopic_safe = "".join(x for x in subtopic if x.isalnum() or x in " -_").strip()
    
    topic_dir = os.path.join(TRAINING_DATA_DIR, topic_safe)
    subtopic_dir = os.path.join(topic_dir, subtopic_safe)
    
    if args.casual and os.path.exists(subtopic_dir):
        print(f"Overwriting {subtopic_dir} for casual generation...")
        shutil.rmtree(subtopic_dir)

    os.makedirs(subtopic_dir, exist_ok=True)
    
    print(f"Generating examples for '{topic}' -> '{subtopic}'...")
    
    # Check existing files
    existing_files = [f for f in os.listdir(subtopic_dir) if f.endswith(".txt")]
    if len(existing_files) >= 100:
        print(f"Skipping {topic}/{subtopic} (already has {len(existing_files)} files).")
        return

    print(f"  Generating 100 examples...")
    
    length_instruction = "Keep the examples very concise (around 10 tokens)."
    
    context_str = f" specifically focusing on: '{args.context}'." if args.context else "."

    if args.casual:
        prompt = f"Generate 100 unique casual conversation pairs related to '{subtopic}'{context_str} The input should be a casual remark or greeting, and the output should be a natural, helpful response from an AI assistant. Ensure the AI does not claim to have personal experiences, holidays, or a physical life. Example: 'hi' -> 'hello, how can I assist you today?'. {length_instruction} Return ONLY valid JSON."
    else:
        prompt = f"Generate 100 unique question/response pairs for the topic '{subtopic}'{context_str} {length_instruction} The questions should vary in difficulty and style. Return ONLY valid JSON."
    
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
