# 🧠 TITAN AGENT: Autonomous OS Controller

> **"We aren't downloading a brain. We are building one."**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20%2F%20Alpha-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Titan est une initiative de recherche expérimentale visant à créer un agent d'Intelligence Artificielle Générale (**AGI**) capable d'utiliser un système d'exploitation (Windows/Linux) comme un humain.

Contrairement aux approches basées sur les LLM (comme GPT-4), Titan ne "lit" pas le HTML. Il **voit** l'écran, **comprend** l'interface graphique via la vision par ordinateur, et **agit** en déplaçant la souris et en tapant au clavier.

Le projet est inspiré par le papier *"World Models"* (Ha & Schmidhuber, 2018), mais appliqué à un environnement de bureau complexe.

---

## 🏗 Architecture : "The Sensory-Motor Loop"

Le projet est divisé en trois composants neuronaux distincts :

### 1. 👁 The Vision Encoder (Actuellement en Phase 1)
*   **Rôle :** Compresser des captures d'écran 1080p en un vecteur latent compact.
*   **Technologie :** **VAE (Variational Autoencoder)** avec Perceptual Loss (VGG19) et détection de contours (Laplacian).
*   **Pourquoi ?** Pour permettre à l'agent de "voir" sans saturer la VRAM. Il apprend à reconnaître les fenêtres, le texte et les boutons.

### 2. 🧠 The World Model (Prochaine étape)
*   **Rôle :** Prédire le futur.
*   **Concept :** "Si je clique ici, que va-t-il se passer ?"
*   **Technologie :** Transformer / LSTM. Il simule l'environnement mentalement avant d'agir.

### 3. 🎮 The Policy Network (Contrôleur)
*   **Rôle :** Prendre des décisions.
*   **Technologie :** Reinforcement Learning (PPO/Dreamer).

---

## 🚀 Fonctionnalités Clés

*   **Intelligence Multi-Moniteurs :** Le système de collecte détecte automatiquement sur quel écran se trouve votre souris et n'enregistre que l'écran actif (supporte les setups horizontaux complexes).
*   **High-Fidelity Vision :** Entraînement sur des images natives 1080p via une stratégie de "Random Cropping" (256x256) pour préserver la lisibilité du code et du texte.
*   **Optimisation Hardware :** Conçu pour tourner sur des GPU grand public (12GB VRAM - RTX 3060 / RX 6700 XT).
*   **Titan GUI :** Interface graphique complète (Tkinter) pour piloter l'entraînement, visualiser les pertes en temps réel et gérer les checkpoints.

---

## 📦 Installation

### Prérequis
*   Python 3.9 ou supérieur.
*   Un GPU avec au moins **8 Go de VRAM** (12 Go recommandés).
*   **Pour les utilisateurs AMD (RX 6000/7000)** : Linux recommandé avec ROCm installé.
*   **Pour les utilisateurs NVIDIA** : CUDA Toolkit installé.

### Setup

1.  Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-user/titan-agent.git
    cd titan-agent
    ```

2.  Installez les dépendances :
    ```bash
    pip install torch torchvision numpy opencv-python mss pynput matplotlib
    ```
    *(Note : Pour PyTorch, visitez [pytorch.org](https://pytorch.org/) pour la commande exacte selon votre OS/GPU).*

---

## 🕹 Utilisation

### Phase 1 : Collecte de Données (Behavioral Cloning)
Avant que l'IA ne puisse agir, elle doit observer.

1.  Lancez l'enregistreur :
    ```bash
    python data_collector/recorder.py
    ```
2.  Travaillez normalement sur votre PC (Codez, naviguez, utilisez le terminal).
3.  L'enregistreur capture l'écran actif (là où est la souris) et sauvegarde les données compressées dans `/data`.
4.  Appuyez sur `Ctrl+C` dans le terminal pour arrêter et sauvegarder.

### Phase 2 : Entraînement de la Vision (VAE)
Apprenez à l'IA à comprendre ce qu'elle voit.

1.  Lancez l'interface d'entraînement :
    ```bash
    python titan_gui.py
    ```
2.  Sélectionnez votre dossier de données.
3.  Réglez les hyperparamètres (ou laissez par défaut).
4.  Cliquez sur **START NEW RUN**.
5.  Observez la courbe de "Loss" descendre et les reconstructions s'améliorer.

---

## 📂 Structure du Projet

```text
/titan-agent
│
├── /data_collector       # Outils d'enregistrement
│   ├── recorder.py       # Capture intelligente (Multi-screen)
│   ├── config.py         # Paramètres (FPS, Résolution)
│   └── io_utils.py       # Gestion stockage efficace (.npz)
│
├── /models               # Architectures neuronales
│   └── vae.py            # Le réseau de vision (Perceptual VAE)
│
├── /checkpoints          # Sauvegardes des poids du modèle
├── /results              # Images générées pendant l'entraînement
│
├── titan_gui.py          # Dashboard de contrôle (Tkinter)
└── train_vision_robust.py # Moteur d'entraînement (Backend)
```

---

## ⚠️ Avertissement de Sécurité

Ce projet a pour but ultime de donner le contrôle de la souris et du clavier à un réseau de neurones.
*   Lors des phases futures (RL), il est impératif d'exécuter l'agent dans une **Machine Virtuelle (VM)** ou un environnement sandboxé.
*   L'auteur n'est pas responsable si l'agent supprime vos fichiers ou envoie des messages aléatoires sur Slack.

---

## 🤝 Contribuer
