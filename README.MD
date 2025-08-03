# 📸 Curadoria Automática de Fotos com CLIP + MLP

Este projeto realiza **curadoria automática de imagens** utilizando **modelos de visão computacional** para selecionar apenas as melhores fotos, com suporte a arquivos **HEIC, JPEG, PNG e WebP**.  

Ele é ideal para criar coleções de **wallpapers verticais** ou selecionar automaticamente fotos com maior **qualidade estética**.

---

## 🚀 Como funciona

1. **Leitura de imagens** em uma pasta de origem (`fotos/`), incluindo HEIC (via `pillow-heif`).  
2. **Filtragem opcional por orientação** – processa apenas imagens **em pé (altura > largura)**.  
3. **Extração de embeddings visuais** com [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14).  
4. **Avaliação de qualidade estética** com uma MLP treinada  
   (modelo de Christoph Schuhmann – `sac+logos+ava1-l14-linearMSE.pth`).  
5. **Seleção das melhores fotos** com duas estratégias:  
   - **Por threshold**: copia imagens com nota acima de um valor definido.  
   - **Top X%**: copia apenas uma porcentagem das fotos com melhor pontuação.  
6. **Cópia para a pasta `wallpapers/`**, preservando metadados.  

---

## 📂 Estrutura esperada do projeto

.
├── fotos/                       # Pasta de origem com imagens
├── wallpapers/                  # Pasta destino para as melhores imagens
├── models/
│   └── sac+logos+ava1-l14-linearMSE.pth
├── mlp.py                       # Definição da MLP
├── curadoria.py                 # Script principal
└── requirements.txt             # Dependências (opcional)

---

## 🛠️ Instalação

1. Clone o repositório:
git clone https://github.com/seu-usuario/curadoria-de-fotos.git
cd curadoria-de-fotos


2. Crie um ambiente virtual (opcional, mas recomendado):
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows


3. Instale as dependências:
pip install torch pillow pillow-heif tqdm transformers

---

## ▶️ Uso

1. Coloque suas imagens na pasta `fotos/`.  
2. Ajuste no script:
   - `THRESHOLD` (modo por nota mínima) **ou**
   - `TOP_PERCENT` (modo por top X%).  
3. Execute:
python curadoria.py

4. As imagens selecionadas serão copiadas para `wallpapers/`.

---

## 🔹 Tecnologias utilizadas

- [Python 3](https://www.python.org/)  
- [PyTorch](https://pytorch.org/)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
- [Pillow](https://pillow.readthedocs.io/)  
- [pillow-heif](https://pypi.org/project/pillow-heif/)  
- [tqdm](https://tqdm.github.io/)  

---

## 📜 Licença

Este projeto é distribuído sob a licença MIT.  
Sinta-se livre para usar e modificar.

---

### ✨ Exemplo de resultado

- Entrada: 500 fotos diversas (HEIC, JPEG, PNG)  
- Saída: 25 fotos verticais com maior qualidade estética copiadas para `wallpapers/`  