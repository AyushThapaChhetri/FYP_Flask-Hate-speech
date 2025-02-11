from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re
from pathlib import Path
import os
import wget

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# CNN Model Definition
class CNN_NLP(nn.Module):
    def __init__(self, pretrained_embedding=None, freeze_embedding=False,
                 vocab_size=None, embed_dim=300, filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100], num_classes=3, dropout=0.5):
        super(CNN_NLP, self).__init__()
        
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                        freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=self.embed_dim,
                                        padding_idx=0,
                                        max_norm=5.0)
        
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                     out_channels=num_filters[i],
                     kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, input_ids):
        x_embed = self.embedding(input_ids).float()
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                      for x_conv in x_conv_list]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                        dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits

def clean_text(text, remove_repeat_text=True, is_lower=True):
    """Clean and preprocess text."""
    if is_lower:
        text = text.lower()
    
    if remove_repeat_text:
        text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = str(text).replace("\n", " ")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    text = re.sub(r' +', ' ', text).strip()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    
    return text

class HateSpeechDetector:
    def __init__(self, model_path='model.pth', word2idx_path='word2idx.npy'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 62
        
        # Load word2idx
        self.word2idx = np.load(word2idx_path, allow_pickle=True).item()
        
        # Initialize and load model
        self.model = CNN_NLP(vocab_size=len(self.word2idx), embed_dim=300)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # Clean and tokenize text
        cleaned_text = clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Pad tokens
        padded_tokens = tokens + ['<pad>'] * (self.max_len - len(tokens))
        
        # Convert tokens to indices
        input_id = [self.word2idx.get(token, self.word2idx['<unk>']) 
                   for token in padded_tokens[:self.max_len]]
        
        # Convert to tensor
        input_id = torch.tensor(input_id).unsqueeze(dim=0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(input_id)
            probs = F.softmax(logits, dim=1).squeeze(dim=0)
            predicted_class = torch.argmax(probs).item()
            
        return {
            "prediction": predicted_class,
            "probabilities": probs.tolist()
        }

def setup_requirements():
    """Download required files if they don't exist."""
    # Create a models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Download fastText embeddings if they don't exist
    fasttext_path = "models/crawl-300d-2M.vec"
    if not os.path.exists(fasttext_path):
        print("Downloading fastText embeddings...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        wget.download(url, "models/fasttext.zip")
        import zipfile
        with zipfile.ZipFile("models/fasttext.zip", 'r') as zip_ref:
            zip_ref.extractall("models")
        os.remove("models/fasttext.zip")

# Initialize the model
detector = None

# @app.before_first_request
def initialize():
    global detector
    setup_requirements()
    detector = HateSpeechDetector()

initialize()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = detector.predict(text)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    # initialize()
    app.run(debug=True)