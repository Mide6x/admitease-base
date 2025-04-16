import os
import shutil
import re
from transformers import T5Tokenizer
# import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# openai.api_key = os.getenv('OPEN_AI_KEY')
client = OpenAI()

def ensure_directories():
    """Create 'wrong' and 'corrected' directories if they don't exist"""
    for directory in ['wrong', 'corrected']:
        os.makedirs(directory, exist_ok=True)

def correct_essay(text):
    """Correct grammar, punctuation, and translate non-English text to proper English"""
    try:
        # response = openai.Completion.create(
        response = client.chat.completions.create(
             model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional essay editor. Correct grammar, "
                 "punctuation, and translate any non-English text (including pidgin) to proper English. "
                 "Maintain the original meaning and tone while making it more professional."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error correcting essay: {str(e)}")
        return None

def process_essays():
    """Process all text files in the 'wrong' folder"""
    wrong_dir = 'wrong'
    corrected_dir = 'corrected'
    
    # Ensure directories exist
    ensure_directories()
    
    # Process each file in the wrong directory
    for filename in os.listdir(wrong_dir):
        if filename.endswith(('.txt', '.doc', '.docx')):
            input_path = os.path.join(wrong_dir, filename)
            output_path = os.path.join(corrected_dir, f"corrected_{filename}")
            
            try:
                # Read the essay
                with open(input_path, 'r', encoding='utf-8') as file:
                    original_text = file.read()
                
                # Correct the essay
                corrected_text = correct_essay(original_text)
                
                if corrected_text:
                    # Save the corrected essay
                    with open(output_path, 'w', encoding='utf-8') as file:
                        file.write(corrected_text)
                    print(f"Successfully processed: {filename}")
                else:
                    print(f"Failed to process: {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    process_essays()

