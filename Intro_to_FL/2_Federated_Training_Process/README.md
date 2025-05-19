# Federated Learning with Flower 🌼 – MNIST Example

Ce projet met en œuvre un processus d'entraînement fédéré utilisant [Flower](https://flower.dev/) et [PyTorch](https://pytorch.org/) sur le dataset MNIST.  
Il a été réalisé dans le cadre d’un stage de recherche sur l'intégration du Federated Learning dans des architectures sécurisées et distribuées.

---

## 🚀 Objectifs du projet

- Comprendre et tester le fonctionnement de l’apprentissage fédéré.
- Utiliser le framework **Flower** pour simuler plusieurs clients.
- Entraîner un modèle CNN en FL sur **MNIST** avec des données réparties.
- Préparer un socle pour de futures expériences intégrant la **blockchain**.

---

## 🧰 Technologies utilisées

- Python 3.11.9
- [Flower](https://flower.dev/) 1.10.0
- PyTorch 2.2.1
- Torchvision
- Jupyter Notebook (exécution dans VS Code)
- Environnement virtuel `venv`

---

## ⚙️ Installation

1. **Cloner le dépôt**  
```bash
git clone https://github.com/ton-pseudo/mon-projet-flower.git
cd mon-projet-flower
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
```

3. **Activer l’environnement virtuel**

- Sous Windows :
```bash
.env\Scriptsctivate
```
- Sous Mac/Linux :
```bash
source venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

---

## 📦 Contenu

| Fichier                         | Description                            |
|----------------------------------|----------------------------------------|
| `L2_Federated_Training_Process.ipynb` | Notebook principal avec le processus FL |
| `requirements.txt`             | Dépendances à installer                |
| `.gitignore`                   | Fichiers exclus du versionnement       |

---

## 🧪 Exécution du notebook

1. Ouvrir le notebook dans VS Code ou Jupyter.
2. Sélectionner le kernel lié au venv.
3. Lancer cellule par cellule :
   - Téléchargement de MNIST
   - Préparation des datasets pour les clients
   - Définition du modèle
   - Entraînement fédéré sur 3 rounds

---

## 📝 Résultat

- Le modèle est entraîné via **3 clients simulés**.
- Chacun reçoit une **partie du dataset** et entraîne un modèle localement.
- Les modèles sont **agrégés** côté serveur avec **FedAvg**.

---

## 📌 À venir

- Intégration d’une couche **blockchain** pour la traçabilité.
- Test sur des datasets plus complexes (CIFAR-10).
- Simulation d’environnements edge distribués.

---

## 👤 Auteur

Brandon Delmas  
Étudiant en 4e année à ESIEE Paris – Stage de recherche à l’UMONS  
2025
