from google.colab import drive
import pandas as pd
import sentencepiece as spm

# Mount Google Drive
drive.mount('/content/drive/', force_remount=True)

# Load the dataset
data_path = "/content/drive/My Drive/train.csv"
df = pd.read_csv(data_path)

# Display basic information about the dataset
print(df.info())
print(df.head())

# Check and drop duplicates
df = df.drop_duplicates()

# Train SentencePiece model
input_text = '/content/drive/My Drive/medicalqa.txt'  # Text data for SentencePiece
spm.SentencePieceTrainer.train(input=input_text, model_prefix='medicalqa', vocab_size=16000)

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('medicalqa.model')

# Tokenize a sample sentence
sample_text = "Who is at risk for Lymphocytic Choriomeningitis?"
tokenized_text = sp.encode_as_pieces(sample_text)
print(tokenized_text)
