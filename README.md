**General Needs Classification Project
HAST-LNC: Infant Cry Classification using Transformer-Based Audio Embeddings**

**1. Project Description**
   
This project presents an automated Infant Cry Classification System based on the HAST-LNC (Hybrid Audio Spectrogram Transformer Embedding with Lightweight Neural Classification) framework. The system analyzes infant cry audio recordings and predicts the possible reason behind the cry such as hunger, discomfort, tiredness, belly pain, or burping. The approach combines transformer-based audio feature extraction with lightweight machine learning classifiers to achieve accurate and computationally efficient cry detection.

**2. Overview**
   
Infant crying is the primary way babies communicate their needs. Understanding the reason behind a baby's cry can help caregivers respond quickly and appropriately. Traditional methods rely on manual observation or handcrafted acoustic features. In this project, we leverage Audio Spectrogram Transformer (AST) embeddings to automatically learn high-level representations from cry audio signals. These embeddings are then used with Logistic Regression and Multi-Layer Perceptron (MLP) classifiers to predict the cry category.

**3. Installation**

Clone the repository:
git clone https://github.com/yourusername/infant-cry-classification.git
cd infant-cry-classification
Install required dependencies:
pip install -r requirements.txt

**4. Dataset**
The project uses the Donate-a-Cry Infant Cry Dataset.
Attribute	Details
Dataset Name	Donate-a-Cry Corpus
Audio Format	WAV
Total Files	457
Duration	6–7 seconds
Categories	Belly Pain, Burping, Discomfort, Hungry, Tired

**Dataset Source:**
https://www.kaggle.com/datasets/warcoder/donateacry-corpus

**Project Workflow**
The system follows the HAST-LNC workflow pipeline:
Infant Cry Audio
        ↓
Audio Preprocessing
        ↓
Spectrogram Conversion
        ↓
AST Feature Extraction
        ↓
Classification (Logistic Regression / MLP)
        ↓
Cry Category Prediction

**Audio Preprocessing**

Audio signals are preprocessed before feature extraction.
Steps include:
1.Noise reduction
2.udio normalization
3.Resampling to 16 kHz
4.Removing corrupted audio samples

Example preprocessing code:
import librosa
import numpy as np

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio = audio / np.max(np.abs(audio))
    return audio

**Feature Extraction using AST**
The Audio Spectrogram Transformer (AST) extracts deep audio embeddings from spectrogram representations.
Example feature extraction pipeline:
X_clean = []
y_clean = []

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            features = extract_ast_features(path)

            if features.shape == (768,):
                X_clean.append(features)
                y_clean.append(label2id[label])

X = np.array(X_clean)
y = np.array(y_clean)
Output feature vector size:
768-dimensional AST embeddings

**Logistic Regression Classifier**

Logistic Regression is used as a baseline classifier.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

**MLP Classifier**

The Multi-Layer Perceptron (MLP) classifier learns nonlinear relationships between audio features.
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation='relu',
    max_iter=200
)

mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

**MLP Architecture**
Layer	Units	Activation
Input Layer	768	—
Hidden Layer 1	128	ReLU
Hidden Layer 2	64	ReLU
Output Layer	5	Softmax

**Evaluation Metrics**
The model performance is evaluated using the following metrics:
1.Accuracy
2.Precision
3.Recall
4.F1 Score
5.Confusion Matrix
6.ROC Curve
7.Matthews Correlation Coefficient

**Example confusion matrix code:**
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

**Results**
The experimental results show that AST embeddings combined with lightweight classifiers provide reliable infant cry classification.
**Key observations:**
1.AST embeddings capture subtle acoustic variations in infant cries.
2.Logistic Regression provides baseline performance.
3.MLP improves classification by learning nonlinear decision boundaries.
4.The hybrid approach achieves strong accuracy with lower computational cost.

**Applications**
The system can be applied in:
1.Smart baby monitoring systems
2.Healthcare support applications
3.Parenting assistance tools
4.Infant behavior analysis

**Technologies Used**
1.Python
2.PyTorch
3.Scikit-learn
4.Librosa
5.NumPy
6.Matplotlib
7.Seaborn
8.HuggingFace Transformers

**Future Work**
Possible improvements include:
1.Real-time infant cry detection
2.Mobile or edge-device deployment
3.Larger datasets for improved model generalization
4.Integration with IoT-based baby monitoring devices

**Author**
Deekshita Gudla, Anusha Kalluru,Nandu Attili,Bodala Pallavi
