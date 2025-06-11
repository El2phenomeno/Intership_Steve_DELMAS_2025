import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- 1. Chargement et préparation du dataset UCI Adult ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)

# Encodage des colonnes catégorielles
for col in ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]:
    df[col] = LabelEncoder().fit_transform(df[col])

df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

# Séparation des features
bank_features = ["education-num", "capital-gain", "hours-per-week"]
assurance_features = ["age", "sex", "race"]
Xb = df[bank_features].values
Xa = df[assurance_features].values
y = df["income"].values

# Normalisation
Xb = StandardScaler().fit_transform(Xb)
Xa = StandardScaler().fit_transform(Xa)

# Train/test split
Xb_train, Xb_test, Xa_train, Xa_test, y_train, y_test = train_test_split(Xb, Xa, y, test_size=0.2, random_state=42)

# --- 2. PyTorch - Préparation des tenseurs ---
Xb_train = torch.tensor(Xb_train, dtype=torch.float32)
Xa_train = torch.tensor(Xa_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# --- 3. Définition des modèles banque + assurance ---
class BankEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class LabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5 + 3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, emb, x_a):
        return self.model(torch.cat([emb, x_a], dim=1))

encoder = BankEncoder()
label_model = LabelModel()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(label_model.parameters()), lr=0.01)
loss_fn = nn.BCELoss()

# --- 4. Entraînement ---
for epoch in range(150):
    emb = encoder(Xb_train)
    preds = label_model(emb, Xa_train)
    loss = loss_fn(preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- 5. Attaque spectrale ---
encoder.eval()
with torch.no_grad():
    embeddings = encoder(Xb_train).numpy()
    true_labels = y_train.squeeze().numpy()

centered = embeddings - embeddings.mean(axis=0)
svd = TruncatedSVD(n_components=1)
scores = svd.fit_transform(centered)[:, 0]

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(scores.reshape(-1, 1))
auc0 = roc_auc_score(true_labels, clusters)
auc1 = roc_auc_score(true_labels, 1 - clusters)
best_auc = max(auc0, auc1)
print(f"\n🎯 AUC Spectral Attack: {best_auc:.4f}")

# --- 6. VISUALISATIONS ---

# 1. Projection sur 1ère composante SVD
plt.figure(figsize=(8, 3))
plt.scatter(scores, np.zeros_like(scores), c=true_labels, cmap="coolwarm", alpha=0.6)
plt.title("Spectral Attack on Embeddings (UCI Adult Dataset)")
plt.xlabel("Projection onto 1st SVD component")
plt.yticks([])
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Courbe ROC
fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle='--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Spectral Attack")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Histogramme des scores SVD par label
plt.figure(figsize=(7, 4))
plt.hist(scores[true_labels == 0], bins=30, alpha=0.6, label='Label 0 (≤50K)', color='blue')
plt.hist(scores[true_labels == 1], bins=30, alpha=0.6, label='Label 1 (>50K)', color='red')
plt.xlabel("SVD Projection Score")
plt.ylabel("Nombre d'exemples")
plt.title("Distribution des Scores Spectral Attack par Label")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
