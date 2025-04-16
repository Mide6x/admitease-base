import os
import shutil
from transformers import T5Tokenizer

# Config
MAX_WORDS = 350  # Hard limit for Nigerian English essays
MIN_WORDS = 50   # Minimum meaningful segment
TOKENIZER = T5Tokenizer.from_pretrained("t5-small")

def count_words(text):
    return len(text.split())

def split_essay(text):
    """Split essay into chunks at paragraph boundaries"""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = count_words(para)
        if current_length + para_length <= MAX_WORDS:
            current_chunk.append(para)
            current_length += para_length
        else:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def process_pair(wrong_path, correct_path, output_dir):
    with open(wrong_path, 'r', encoding='utf-8') as f:
        wrong = f.read().strip()
    with open(correct_path, 'r', encoding='utf-8') as f:
        correct = f.read().strip()
    
    # Case 1: Both within limits
    if count_words(wrong) <= MAX_WORDS and count_words(correct) <= MAX_WORDS:
        shutil.copy(wrong_path, os.path.join(output_dir, "wrong", os.path.basename(wrong_path)))
        shutil.copy(correct_path, os.path.join(output_dir, "correct", os.path.basename(correct_path)))
        return 1
    
    # Case 2: Needs splitting
    wrong_chunks = split_essay(wrong)
    correct_chunks = split_essay(correct)
    
    # Ensure we have matching chunks
    if len(wrong_chunks) != len(correct_chunks):
        print(f"⚠️ Could not align chunks for {os.path.basename(wrong_path)} - "
              f"{len(wrong_chunks)} wrong vs {len(correct_chunks)} correct")
        return 0

    
    # Save chunks
    base_name = os.path.splitext(os.path.basename(wrong_path))[0]
    for i, (w_chunk, c_chunk) in enumerate(zip(wrong_chunks, correct_chunks)):
        # Skip chunks that are too short
        if count_words(w_chunk) < MIN_WORDS or count_words(c_chunk) < MIN_WORDS:
            continue
            
        with open(os.path.join(output_dir, "wrong", f"{base_name}_part{i+1}.txt"), 'w', encoding='utf-8') as f:
            f.write(w_chunk)
        with open(os.path.join(output_dir, "correct", f"{base_name}_part{i+1}.txt"), 'w', encoding='utf-8') as f:
            f.write(c_chunk)

    
    return len(wrong_chunks)

def main():
    raw_wrong_dir = "./essays/raw_wrong"
    raw_correct_dir = "./essays/raw_correct"
    processed_dir = "./essays/processed"
    
    os.makedirs(os.path.join(processed_dir, "wrong"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "correct"), exist_ok=True)
    
    wrong_files = sorted(os.listdir(raw_wrong_dir))
    correct_files = sorted(os.listdir(raw_correct_dir))
    
    assert len(wrong_files) == len(correct_files), "Mismatched files!"
    
    total_pairs = 0
    for w_file, c_file in zip(wrong_files, correct_files):
        total_pairs += process_pair(
            os.path.join(raw_wrong_dir, w_file),
            os.path.join(raw_correct_dir, c_file),
            processed_dir
        )
    
    print(f"\n✅ Pre-processing complete\n"
          f"- Original pairs: {len(wrong_files)}\n"
          f"- Processed pairs: {total_pairs}\n"
          f"- Output directory: {processed_dir}")

if __name__ == "__main__":
    main()
