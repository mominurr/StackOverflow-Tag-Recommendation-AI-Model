# 🤖 StackOverflow Tag Recommendation AI Model

An end-to-end machine learning project that automatically recommends relevant tags for StackOverflow questions using deep learning. The project includes data preprocessing, model training (both custom neural networks and transformers), deployment via Gradio on HuggingFace Spaces, and a web interface for GitHub Pages.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.50%2B-yellow)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project tackles the multi-label classification problem of predicting relevant tags for StackOverflow questions. Given a question text, the model predicts the top-k most relevant tags from a set of 100 popular StackOverflow tags.

**Key Features:**
- 📊 Comprehensive data preprocessing with stratified splitting
- 🧠 Two model approaches: Custom LSTM-based NN and BERT Transformer
- 🎨 Interactive Gradio web interface
- 🚀 Deployable on HuggingFace Spaces
- 🌐 Static web interface for GitHub Pages
- 📈 Detailed evaluation metrics and visualization

**Use Cases & Benefits**
1. **Automated Tagging System:** Reduces the manual effort required to tag questions by automatically suggesting the most relevant tags.
2. **Enhanced Search & Discovery:** Ensures accurate categorization, leading to improved search results and question recommendations.
3. **Duplicate Question Detection:** Identifies similar or duplicate questions based on tag similarity, helping users find existing answers more efficiently.
4. **Tag Optimization for New Users:** Assists new users in selecting the most appropriate tags when posting questions, reducing the risk of misclassification.
5. **Content Moderation & Filtering:** Provides Stack Overflow moderators with an efficient tool for filtering, categorizing, and managing content more effectively.

## 📁 Project Structure

```
StackOverflow-Tag-Recommendation-AI-Model/
│
├── data/                          # Dataset files
│   ├── train.csv                  # Training set (~85%)
│   ├── val.csv                    # Validation set (~10%)
│   └── test.csv                   # Test set (~5%)
│
├── model/                         # Trained model files
│   ├── ai_stackoverflow_model.pth # Custom NN model
│
├── notebooks/                     # Google Colab notebooks
│   ├── data_preparation.ipynb     # Data preprocessing
│   ├── train_custom_nn.ipynb      # Custom NN training
│   └── train_transformer.ipynb    # Transformer training
│
├── docs/                          # GitHub Pages website
│   ├── index.html                 # Main webpage
│
├── app.py                         # Gradio app for HuggingFace
├── utils.py                       # Helper functions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📊 Dataset

The dataset consists of StackOverflow questions with the following structure:

**Dataset Statistics:**
- Total samples: 210,656
- Number of tags: 100 (most popular)
- `data/train.csv` (179,903 samples)
- `data/val.csv` (20,015 samples)
- `data/test.csv` (10,738 samples)
- Stratified split to preserve tag distribution
- File size is big that's why not uploaded here. You can get the dataset from [this link](https://www.kaggle.com/datasets/mominurr518/data-files)

**Top Tags**: `python`, `javascript`, `java`, `c#`, `php`, `android`, `html`, `jquery`, `c++`, `css`, etc.


## 🚀 Usage

1. **Clone the repository**

```bash
git clone https://github.com/mominurr/StackOverflow-Tag-Recommendation-AI-Model.git
cd StackOverflow-Tag-Recommendation-AI-Model
```

2. **Create a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Local Deployment**

Run the Gradio app locally:

```bash
python app.py
```

The app will start at `http://localhost:7860`

**Features:**
- Interactive text input for questions
- Adjustable confidence threshold
- Adjustable top-k predictions
- Real-time tag predictions with confidence scores


## 🏗️ Model Architecture

### Custom Neural Network

```
Input Text
    ↓
Text Preprocessing (tokenization, lemmatization)
    ↓
Embedding Layer (vocab_size × 200)
    ↓
Bidirectional LSTM (2 layers, hidden=256)
    ↓
Mean Pooling
    ↓
Dropout (0.3)
    ↓
Fully Connected (512)
    ↓
ReLU + Dropout
    ↓
Fully Connected (100 tags)
    ↓
Sigmoid → Tag Probabilities
```

**Model Parameters**: 11,877,885

**Input**: Tokenized text (max length: 128)

**Output**: 100-dimensional probability vector

### Transformer Model

```
Input Text
    ↓
BERT Tokenizer
    ↓
BERT Base (bert-base-uncased)
    ↓
[CLS] Token Representation
    ↓
Dropout (0.3)
    ↓
Linear Classifier (768 → 100)
    ↓
Sigmoid → Tag Probabilities
```

**Model Parameters**: ~110M

**Input**: BERT tokens (max length: 128)

**Output**: 100-dimensional probability vector

## 📈 Results

### Custom Neural Network Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Micro F1** | 0.8242 | 0.8566 | 0.8566 |
| **Macro F1** | 0.7530 | 0.8115 | 0.8115 |
| **Precision** | 0.8783 | 0.8979 | 0.8979 |
| **Recall** | 0.7765 | 0.8188 | 0.8188 |
| **Exact Match Accuracy** | 61.46% | 68.05% | 68.05% |

### Transformer Model (BERT) Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Micro F1** | 0.7891 | 0.8144 | 0.8162 |
| **Macro F1** | 0.7007 | 0.7344 | 0.7378 |
| **Precision** | 0.8631 | 0.8732 | 0.8790 |
| **Recall** | 0.7266 | 0.7630 | 0.7617 |
| **Exact Match Accuracy** | 55.13% | 59.22% | 59.15% |

**Observations:**
- Custom NN performs better than Transformer on this task
- Custom NN provides good balance of speed and accuracy
- Both models handle rare tags reasonably well, but Custom NN has an edge
- Multi-label F1 scores are competitive with state-of-the-art
- Exact match accuracy indicates room for improvement in predicting all tags correctly
- Further hyperparameter tuning and data augmentation could enhance performance
- **I chose to deploy the Custom Neural Network model for the Gradio app due to its superior performance and efficiency.**

## 🌍 Deployment
The trained model is deployed using **Gradio** on **Hugging Face** for easy access and real-time testing.

🔗 **[HuggingFace Spaces App Live URL](https://huggingface.co/spaces/mominur-ai/StackOverflow-Tag-Recommendation-AI-Model)**

### Deployed Model Testing Image Result
<p align="center">
  <img src="test_result_img\result.png" width="100%">
</p>

### API-Based Webpage
A **webpage** is being developed where users can **interact with the deployed model** through an **API**, allowing them to upload questions in text form and receive real-time prediction results.

🔗 **[Webpage Live URL](https://mominurr.github.io/StackOverflow-Tag-Recommendation-AI-Model/)**



## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

**Areas for improvement:**
- Add more training data
- Experiment with different architectures
- Implement attention mechanisms
- Add model interpretability features
- Add multilingual support

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- StackOverflow for the dataset
- HuggingFace for hosting infrastructure
- PyTorch team for the amazing framework
- Gradio team for the easy-to-use interface library

## 📧 Contact

For any inquiries or collaborations:
- **Portfolio:** [mominur.dev](https://mominur.dev)
- **GitHub:** [github.com/mominurr](https://github.com/mominurr)
- **LinkedIn:** [linkedin.com/in/mominur--rahman](https://www.linkedin.com/in/mominur--rahman/)
- **Email:** mominurr518@gmail.com

---

**⭐ Star this repository if you find it helpful!**
