"""
StackOverflow Tag Recommendation AI Model - Gradio App
This app provides a user interface for predicting StackOverflow tags from questions.
Deployable on HuggingFace Spaces with API support.
"""

import torch
import torch.nn as nn
import gradio as gr
import pickle
import os
from typing import List, Tuple, Dict
import numpy as np
from utils import text_preprocess, text_to_numerical_sequence


device = "cuda" if torch.cuda.is_available() else "cpu"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model", "ai_stackoverflow_model.pth")


# Define the model architecture (must match training)
class StackOverflowTagPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_layers=2, num_labels=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # Attention layer for better pooling
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Attention-based pooling
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        pooled = torch.sum(attention_weights * lstm_out, dim=1)
        dropped_out = self.dropout1(pooled)
        fc1_out = self.fc1(dropped_out)
        relu_out = self.relu(fc1_out)
        dropped2_out = self.dropout2(relu_out)
        logits = self.fc2(dropped2_out)
        return logits


# Load model and artifacts
# print(f"Loading model from: {model_path}")
# print(f"Current directory: {current_dir}")
# print(f"Model exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at {model_path}. "
        f"Please ensure the model file exists in the 'model' directory. "
        f"Current directory contents: {os.listdir(current_dir)}"
    )

ckpt = torch.load(model_path, map_location=device, weights_only=False)

vocab = ckpt['vocab']
label_cols = ckpt['labels']
max_len = ckpt['tokenizer_max_length']

model = StackOverflowTagPredictor(
    vocab_size=ckpt['vocab_size'],
    embed_dim=ckpt['embedding_dim'],
    hidden_dim=ckpt['hidden_dim'],
    num_layers=ckpt['num_layers'],
    num_labels=ckpt['num_labels'],
    dropout=ckpt['dropout']
)

model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

# print("Model loaded successfully!")
# print(f"Running on device: {device}")


def predict_tags_api(question: str, threshold: float = 0.3, top_k: int = 10) -> Dict:
    """
    Predict tags for a StackOverflow question (API format).
    Returns JSON-compatible dictionary for API consumption.
    
    Args:
        question: The question text
        threshold: Minimum probability threshold for predictions
        top_k: Maximum number of tags to return
        
    Returns:
        Dictionary with status, predictions, and metadata
    """
    if not question or len(question.strip()) == 0:
        return {
            "status": "error",
            "message": "Please provide a valid question.",
            "predictions": []
        }
    
    try:
        # Preprocess the question
        tokens = text_preprocess(question)
        if len(tokens) == 0:
            return {
                "status": "error",
                "message": "Question contains no valid tokens after preprocessing.",
                "predictions": []
            }
        
        # Convert to numerical sequence
        sequence = text_to_numerical_sequence(tokens, vocab, max_len)
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        # Filter by threshold and format results
        predictions = []
        for idx in top_indices:
            prob = float(probs[idx])
            if prob >= threshold:
                predictions.append({
                    "tag": label_cols[idx],
                    "confidence": round(prob, 4)
                })
        
        if len(predictions) == 0:
            return {
                "status": "success",
                "message": f"No tags found with confidence above {threshold:.1%}. Try lowering the threshold.",
                "predictions": []
            }
        
        return {
            "status": "success",
            "message": f"Found {len(predictions)} tags",
            "predictions": predictions,
            "metadata": {
                "threshold": threshold,
                "top_k": top_k,
                "device": device
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during prediction: {str(e)}",
            "predictions": []
        }


def predict_tags(question: str, threshold: float = 0.3, top_k: int = 10) -> str:
    """
    Predict tags for a StackOverflow question (UI format).
    
    Args:
        question: The question text
        threshold: Minimum probability threshold for predictions
        top_k: Maximum number of tags to return
        
    Returns:
        Formatted string with predicted tags and probabilities
    """
    result = predict_tags_api(question, threshold, top_k)
    
    if result["status"] == "error":
        return f"⚠️ {result['message']}"
    
    if len(result["predictions"]) == 0:
        return f"⚠️ {result['message']}"
    
    formatted_results = ""
    for pred in result["predictions"]:
        formatted_results += f"**{pred['tag']}**: {pred['confidence']:.2%}\n\n"
    
    return "### 🏷️ Predicted Tags:\n\n" + formatted_results

def create_examples():
    """Create example questions for users to try."""
    return [
        ["How do I read a CSV file in Python using pandas?", 0.3, 10],
        ["What's the difference between let and var in JavaScript?", 0.3, 10],
        ["How to reverse a string in Java?", 0.3, 10],
        ["How do I create a REST API with Flask?", 0.3, 10],
        ["What is the best way to handle async operations in React?", 0.3, 10],
        ["How to connect to MySQL database in PHP?", 0.3, 10],
        ["How do I sort a list in Python?", 0.3, 10],
        ["What's the difference between margin and padding in CSS?", 0.3, 10]
    ]


# Create Gradio interface (compatible with older versions)
with gr.Blocks(title="StackOverflow Tag Predictor") as app:
    gr.Markdown(
        """
        # 🤖 StackOverflow Tag Recommendation AI Model
        
        <div style="text-align: center; margin: 20px 0;">
            <p style="font-size: 18px; color: #666;">
                Built by <strong>Mominur Rahman</strong> | Aspiring AI & ML Engineer | Web Scraping & Automation Specialist
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 15px 0;">
                <a href="mailto:mominurr518@gmail.com" target="_blank" style="text-decoration: none;">
                    <span style="background: #EA4335; color: white; padding: 8px 16px; border-radius: 5px; font-weight: 500;">
                        📧 Email
                    </span>
                </a>
                <a href="https://github.com/mominurr" target="_blank" style="text-decoration: none;">
                    <span style="background: #333; color: white; padding: 8px 16px; border-radius: 5px; font-weight: 500;">
                        💻 GitHub
                    </span>
                </a>
                <a href="https://www.linkedin.com/in/mominur--rahman/" target="_blank" style="text-decoration: none;">
                    <span style="background: #0A66C2; color: white; padding: 8px 16px; border-radius: 5px; font-weight: 500;">
                        💼 LinkedIn
                    </span>
                </a>
                <a href="https://mominur.dev" target="_blank" style="text-decoration: none;">
                    <span style="background: #6366F1; color: white; padding: 8px 16px; border-radius: 5px; font-weight: 500;">
                        🌐 Portfolio
                    </span>
                </a>
                <a href="https://x.com/mominur_rahma_n" target="_blank" style="text-decoration: none;">
                    <span style="background: #1DA1F2; color: white; padding: 8px 16px; border-radius: 5px; font-weight: 500;">
                        🐦 Twitter
                    </span>
                </a>
            </div>
        </div>
        
        ---

        This AI model predicts relevant tags for StackOverflow questions using a trained neural network.
        Simply enter your programming question below and get tag recommendations!
        
        **How it works:** The model analyzes your question text and predicts the most relevant tags 
        based on patterns learned from thousands of StackOverflow questions.
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="📝 Enter your programming question",
                placeholder="e.g., How do I read a CSV file in Python using pandas?",
                lines=5
            )
            
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="🎚️ Confidence Threshold",
                )
                
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="🔢 Max Tags",
                )
            
            predict_btn = gr.Button("🔮 Predict Tags", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Markdown(label="Predictions")
    
    # Examples
    gr.Markdown("### 💡 Try these examples:")
    gr.Examples(
        examples=create_examples(),
        inputs=[question_input, threshold_slider, top_k_slider],
        outputs=output,
        fn=predict_tags,
        cache_examples=False
    )
    
    # Connect the button
    predict_btn.click(
        fn=predict_tags,
        inputs=[question_input, threshold_slider, top_k_slider],
        outputs=output
    )
    
    # Also trigger on Enter key
    question_input.submit(
        fn=predict_tags,
        inputs=[question_input, threshold_slider, top_k_slider],
        outputs=output
    )
    
    # ---------------------------------------------------------
    # API ENDPOINT EXPOSURE
    # This exposes the 'predict_tags_api' function as '/predict_tags_api'
    # ---------------------------------------------------------
    api_trigger = gr.Button(visible=False)
    api_trigger.click(
        fn=predict_tags_api,
        inputs=[question_input, threshold_slider, top_k_slider],
        outputs=gr.JSON(),
        api_name="predict_tags_api"  # <--- Updated Endpoint Name
    )


    gr.Markdown(
        """
        ---
        ### 📈 Model Performance
        | Metric | Train | Validation | Test |
        |--------|-------|------------|------|
        | **Micro F1** | 0.8242 | 0.8566 | 0.8566 |
        | **Macro F1** | 0.7530 | 0.8115 | 0.8115 |
        | **Precision** | 0.8783 | 0.8979 | 0.8979 |
        | **Recall** | 0.7765 | 0.8188 | 0.8188 |
        | **Exact Match Accuracy** | 61.46% | 68.05% | 68.05% |


        ### 📊 Model Information
        - **Architecture**: Bidirectional LSTM with Attention and Embedding layer
        - **Training Data**: StackOverflow questions with top 100 tags
        - **Task**: Multi-label classification
        - **Device**: Running on """ + device + """
        - **Model Parameters**: 11,877,885
        
        ### 📖 Tips
        - Try different confidence thresholds to see more or fewer tags
        - Higher thresholds give more confident predictions
        - Lower thresholds show more possible tags
        - Use the API endpoint to integrate with your web application

        ---

        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-top: 20px;
            max-width: 100%;
            box-sizing: border-box;
            text-align: center;
        ">
            <p style="font-size: 14px; color: #666; margin: 5px 0; word-break: break-word;">
                📝 <strong>Project Repository:</strong> 
                <a href="https://github.com/mominurr/StackOverflow-Tag-Recommendation-AI-Model" target="_blank" style="color: #0066cc; text-decoration: none;">
                    View on GitHub
                </a>
            </p>
            <p style="font-size: 14px; color: #666; margin: 5px 0; word-break: break-word;">
                🤗 <strong>Model Card:</strong> 
                <a href="https://huggingface.co/spaces/mominur-ai/StackOverflow-Tag-Recommendation-AI-Model" target="_blank" style="color: #0066cc; text-decoration: none;">
                    View on HuggingFace
                </a>
            </p>
            <p style="font-size: 14px; color: #666; margin: 5px 0;">
                ⭐ If you find this useful, please star the repo and follow me!
            </p>
            <p style="font-size: 12px; color: #999; margin-top: 15px;">
                Made with ❤️ by Mominur Rahman | © 2025
            </p>
        </div>

        """
    )


app.launch()  # Set share=False if not deploying publicly