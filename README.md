# MedicalQA-T5 Model

## Project Overview

This project involves fine-tuning the T5 transformer model for medical question-answering tasks. The model is trained on a domain-specific dataset, enabling it to generate accurate and contextually relevant medical responses.

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/abdulvahapmutlu/medicalqa-t5-model.git
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

Run the `preprocessing.py` script to prepare the data and train the SentencePiece model.

```
python scripts/preprocessing.py
```

### Model Training

Train the T5 model on the preprocessed dataset:

```
python scripts/training.py
```

### Text Generation

Generate medical answers using the fine-tuned model:

```
python scripts/generation.py
```

## Model Performance

- **Final Training Loss**: ~0.96
- **Final Validation Loss**: ~0.85

The model consistently improved across epochs, demonstrating effective learning and generalization.

## Future Work

- Expand the dataset for broader medical domains.
- Experiment with larger T5 models or alternative transformer architectures.
- Deploy the model in a web-based application for real-time medical Q&A.

## License

This project is licensed under the MIT License.
