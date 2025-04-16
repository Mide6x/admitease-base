# AdmitEase Grammar Correction Model

A Python-based model for automatically correcting grammatical errors in English text using fine-tuned T5 models.

## Overview

This tool uses a fine-tuned T5 transformer model to identify and correct grammatical errors in English sentences. It can generate multiple correction suggestions for each sentence, making it useful for:

- Writing assistance applications
- Educational tools for language learners
- Content editing and proofreading automation
- Document quality improvement

## Installation

```bash
# Clone the repository
git clone https://github.com/Mide6x/admitease-base.git
cd admitease-base

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.18+
- NLTK
- tqdm
- numpy
- pandas
- rouge_score
- evaluate

## Using the Pre-trained Model

The tool comes with a pre-trained model based on `vennify/t5-base-grammar-correction`:

```python
from grammar_correction import load_model, correct_text

# Load the fine-tuned model
model, tokenizer = load_model("./fine_tuned_model")

# Correct a single sentence with multiple suggestions
text = "I face many problem in my life."
corrections = correct_text(text, model, tokenizer, num_suggestions=3)
print(corrections)
```

## Training Your Own Model

If you want to fine-tune the model on your own data:

```bash
python train.py --train_file your_train_data.csv --test_file your_test_data.csv --output_dir ./your_model_dir
```

Your training data should be a CSV file with two columns:

- `input`: The text with grammatical errors
- `target`: The corrected version of the **text**

## Command Line Interface

The tool includes a simple command-line interface:

```bash
# Correct a single sentence
python correct.py --text "I face many problem in my life."

# Correct a text file
python correct.py --file input.txt --output corrected.txt
```

## Model Details

The default model is a T5-base model fine-tuned on grammar correction tasks. The model:

- Handles common grammatical errors (subject-verb agreement, article usage, etc.)
- Preserves text meaning while improving grammar
- Works best on sentences rather than long paragraphs
- Returns multiple correction options with varying levels of modification

## Evaluation Metrics

The model's performance is evaluated using:

- BLEU score: Measures the similarity between model outputs and reference corrections
- ROUGE scores: Measures recall-oriented accuracy for generated text
- Custom grammar conformity metrics

Last fine-tuning results:

- BLEU Score: 1.17
- ROUGE-1: 0.2035
- ROUGE-2: 0.1055
- ROUGE-L: 0.2035

## Output Format

The tool returns a JSON object with suggestions for each sentence:

```json
{
  "sentence_1": {
    "wrong_sentence": "I face many problem in my life.",
    "suggestions": [
      "I face many problems in my life.",
      "I face many problems throughout my life.",
      "I face numerous problems in my life."
    ]
  }
}
```

## Customization

You can customize the correction behavior:

```python
# Adjust correction aggressiveness (higher values = more aggressive changes)
corrections = correct_text(text, model, tokenizer, temperature=0.9)

# Generate more diverse suggestions
corrections = correct_text(text, model, tokenizer, num_suggestions=5, do_sample=True)
```

## Limitations

- The model works best with English text
- Performance may vary for highly technical or domain-specific language
- Very long or complex sentences might be challenging to correct accurately
- The model focuses on grammatical correctness rather than stylistic improvements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
