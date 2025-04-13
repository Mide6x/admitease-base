import os
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import nltk
from transformers import DataCollatorForSeq2Seq
import numpy as np
import random

# Ensure nltk is downloaded for sentence tokenization
nltk.download('punkt')
nltk.download('punkt_tab')


# Configuration
MODEL_NAME = "vennify/t5-base-grammar-correction"
MODEL_CACHE_DIR = "./model_cache"
DATA_DIR = "./essays/processed"
WRONG_DIR = os.path.join(DATA_DIR, "wrong")      
CORRECT_DIR = os.path.join(DATA_DIR, "correct")  
OUTPUT_DIR = "./fine_tuned_model"
BATCH_SIZE = 8
MAX_LENGTH = 512


# Load and preprocess data
def load_essays():
    wrong_files = sorted([f for f in os.listdir(WRONG_DIR) if f.startswith("essay")])
    correct_files = sorted([f for f in os.listdir(CORRECT_DIR) if f.startswith("essay")])
    
    assert len(wrong_files) == len(correct_files), "Mismatched essay pairs!"
    
    samples = []
    for w_file, c_file in zip(wrong_files, correct_files):
        with open(os.path.join(WRONG_DIR, w_file), 'r', encoding='utf-8') as f:
            wrong = f.read().strip()
        with open(os.path.join(CORRECT_DIR, c_file), 'r', encoding='utf-8') as f:
            correct = f.read().strip()
        
        samples.append({
            "input": wrong,
            "target": correct
        })
    
    return samples

# 2. Split into train/test
essay_pairs = load_essays()
train_data, test_data = train_test_split(essay_pairs, test_size=0.2, random_state=42)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    "input": [x["input"] for x in train_data],
    "target": [x["target"] for x in train_data]
})

test_dataset = Dataset.from_dict({
    "input": [x["input"] for x in test_data],
    "target": [x["target"] for x in test_data]
})

# 3. Load model and tokenizer (with local caching)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)

# 4. Tokenize function
def preprocess_function(examples):
    inputs = ["<grammar_correction> " + text for text in examples["input"]]
    targets = examples["target"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    
    labels = tokenizer(
        text=targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"    
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# 5. Metrics
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bleu = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "bleu": bleu["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"]
    }

# 6. Training setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Adjust training parameters with consistent evaluation and save strategies
training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",  # Set this to match save_strategy
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    eval_steps=100,  # Make this match save_steps for simplicity
    save_strategy="steps",  # This is default but explicitly setting for clarity
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train the model
print("Starting training...")
trainer.train()

# 8. Save the fine-tuned model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")


# Sentence-level prediction and suggestions
def get_sentence_suggestions(input_text, model, tokenizer, num_suggestions=3):
    sentences = nltk.tokenize.sent_tokenize(input_text)
    suggestions = {}

    for i, sentence in enumerate(sentences):
        # Add task prefix
        prefixed_sentence = f"grammar_correction> {sentence}"
        
        # Tokenize and generate predictions
        input_ids = tokenizer.encode(prefixed_sentence, return_tensors="pt").to(model.device)
        
        # Generate with do_sample=True since you're using temperature
        output_ids = model.generate(
            input_ids, 
            num_beams=5, 
            num_return_sequences=num_suggestions,
            do_sample=True,  # Add this since you're using temperature
            max_length=100,
            min_length=len(sentence.split()) // 2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=0.7,
        )

        # Decode suggestions and clean them
        suggestions_for_sentence = []
        for output in output_ids:
            suggestion = tokenizer.decode(output, skip_special_tokens=True)
            
            # Remove any prefixes that appear in the output
            prefixes_to_remove = [
                "grammar_correction>", "grammatic_correction>", 
                "gramar_correction>", "<grammar_correction>"
            ]
            
            for prefix in prefixes_to_remove:
                suggestion = suggestion.replace(prefix, "").strip()
            
            # Only add if different and not already in suggestions
            if (suggestion != sentence and 
                suggestion not in suggestions_for_sentence):
                suggestions_for_sentence.append(suggestion)
            
        # If we don't have enough valid suggestions, add the original
        while len(suggestions_for_sentence) < num_suggestions:
            suggestions_for_sentence.append(sentence)
        
        suggestions[f"sentence_{i+1}"] = {
            "wrong_sentence": sentence,
            "suggestions": suggestions_for_sentence[:num_suggestions]
        }

    return suggestions

def predict_on_random_test_data(test_dataset, model, tokenizer):
    # Randomly pick one example from the test dataset
    random_example = random.choice(test_dataset)
    input_text = random_example["input"]  # Get the input text from the random example
    
    # Get sentence-level suggestions for the selected input text
    suggestion = get_sentence_suggestions(input_text, model, tokenizer)
    
    return suggestion

# Then when calling this function:
random_prediction = predict_on_random_test_data(test_dataset, model, tokenizer)

# Print the prediction or save it to a file
with open("random_prediction.json", "w") as f:
    json.dump(random_prediction, f, indent=4)

print("Random prediction done! Check random_prediction.json")


# 9. Evaluate on test set
results = trainer.evaluate()
print("\nFinal Evaluation Results:")
print(f"BLEU Score: {results['eval_bleu']:.2f}")
print(f"ROUGE-1: {results['eval_rouge1']:.4f}")
print(f"ROUGE-2: {results['eval_rouge2']:.4f}")
print(f"ROUGE-L: {results['eval_rougeL']:.4f}")
