# Curso de Inteligência Artificial 🤖

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvarosamp/Curso-IA)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Licença](https://img.shields.io/badge/licença-MIT-green)

Repositório do curso prático de **Inteligência Artificial**, cobrindo desde os fundamentos até aplicações reais com visão computacional e deep learning.

---

## � Sobre o Curso

**Carga horária total:** 64 horas  
**Nível:** Graduação / Pós-graduação  
**Pré-requisitos:** Lógica de programação básica. Python é um diferencial, mas não obrigatório.

O curso é **hands-on**: cada módulo combina teoria, exemplos comentados em Jupyter Notebook e uma atividade prática com dataset real.

---

## 🗂️ Estrutura do Repositório

```
Curso-IA/
├── Aula 1/   # Fundamentos de IA + Ambiente + Git
├── Aula 2/   # Python & Bibliotecas de Data Science
├── Aula 3/   # Fundamentos de Machine Learning
├── Aula 4/   # Deep Learning (MLP, TensorFlow, PyTorch)
├── Aula 5/   # Visão Computacional (OpenCV, CNN, YOLOv8)
└── runs/     # Resultados de treinos YOLO (gerado automaticamente)
```

---

## 📚 Módulos

### Aula 1 — Fundamentos de IA + Ambiente + Git `(8h)`

- O que é IA, ML e Deep Learning
- Aprendizado supervisionado vs. não supervisionado
- Configuração de ambiente virtual (`venv`)
- Git e GitHub: commits semânticos, boas práticas
- **Exemplo prático:** Iris dataset — Decision Tree e K-Means

### Aula 2 — Python Aplicado a Data Science `(8h)`

- Revisão de Python: listas, dicionários, list comprehensions
- **NumPy:** arrays, reshape, operações vetorizadas
- **Pandas:** DataFrames, filtragem, groupby, tratamento de NaN
- **Matplotlib:** histogramas, barras, scatter, boxplot
- **Atividade:** EDA completa no Iris dataset

### Aula 3 — Fundamentos de Machine Learning `(16h)`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvarosamp/Curso-IA/blob/main/Aula%203/Aula_3.ipynb)

- Tipos de aprendizado e trade-off viés-variância
- Pré-processamento: `SimpleImputer`, `OneHotEncoder`, scalers (`MinMax`, `Standard`, `Robust`)
- Seleção de features: `SelectKBest`, RFE, importância em árvores
- Modelos: Random Forest, Regressão Linear, KNN, Árvore de Decisão
- Validação cruzada (K-Fold, StratifiedKFold) e métricas
- Otimização de hiperparâmetros: Grid Search e Random Search
- Pipelines com `ColumnTransformer`

### Aula 4 — Deep Learning `(12h)`

- Perceptron: modelo matemático e limitação (XOR)
- MLP: forward/backpropagation, inicialização de pesos
- Funções de ativação: Sigmoid, tanh, ReLU, Softmax
- Funções de perda: MSE, Cross-Entropy
- Otimizadores: SGD, Momentum, Adam
- **TensorFlow/Keras:** compilação e treinamento de modelos
- **PyTorch:** visão geral e comparação com TensorFlow

### Aula 5 — Visão Computacional `(12h)`

- Processamento de imagens com **OpenCV**: leitura, redimensionamento, normalização, filtros, Canny
- **CNNs:** camadas convolucionais, pooling, arquiteturas clássicas (LeNet, AlexNet, VGG, ResNet)
- **Transfer Learning** com MobileNetV2 (feature extraction e fine-tuning)
- Métricas de detecção: IoU, mAP, curva Precision–Recall
- **Roboflow:** gerenciamento e exportação de datasets
- **YOLOv8 (Ultralytics):** treinamento, validação e inferência
- **Atividade prática:** classificação de dígitos no MNIST (≥3 arquiteturas, comparação de otimizadores)

### Aulas 6 e 7 — Quantização, Otimização & Projeto Final `(8h)`

- Otimização de modelos para dispositivos embarcados (em breve)
- Projeto final: pipeline completo de IA do início ao fim

---

## 🚀 Como Começar

### 1. Clonar o repositório

```bash
git clone https://github.com/alvarosamp/Curso-IA.git
cd Curso-IA
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv ia_env

# macOS/Linux
source ia_env/bin/activate

# Windows
ia_env\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow ultralytics opencv-python
```

### Alternativa: Google Colab

Clique no badge **Open in Colab** no topo de cada módulo. Não é necessário instalar nada localmente.

---

## 🛠️ Tecnologias

| Biblioteca | Uso principal |
|---|---|
| `numpy` | Operações numéricas e arrays |
| `pandas` | Manipulação de dados tabulares |
| `matplotlib` / `seaborn` | Visualização de dados |
| `scikit-learn` | ML clássico, pré-processamento, métricas |
| `tensorflow` / `keras` | Redes neurais e deep learning |
| `pytorch` | Deep learning (visão geral) |
| `opencv-python` | Processamento de imagens |
| `ultralytics` | YOLOv8 — detecção de objetos |

---

## 🔗 Links Úteis

- 📘 Repositório de referência Python: [github.com/alvarosamp/Treinamento_python](https://github.com/alvarosamp/Treinamento_python)
- 🏷️ Dataset management: [roboflow.com](https://roboflow.com)
- 🔬 YOLOv8 docs: [docs.ultralytics.com](https://docs.ultralytics.com)

---

*Última atualização: Março 2026*
