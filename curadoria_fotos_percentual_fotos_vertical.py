import os
import torch
from PIL import Image
import pillow_heif
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from mlp import MLP
import shutil

# Registrar suporte HEIC no PIL
pillow_heif.register_heif_opener()

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_FOTOS = os.path.join(BASE_DIR, "fotos")
PASTA_DESTINO = os.path.join(BASE_DIR, "wallpapers")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sac+logos+ava1-l14-linearMSE.pth")
os.makedirs(PASTA_DESTINO, exist_ok=True)

# Percentual de imagens que serão copiadas (ex.: 0.05 = 5%)
TOP_PERCENT = 0.05

# Carregar CLIP ViT-L/14 para extrair embeddings
clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name)
processor = CLIPProcessor.from_pretrained(clip_model_name)

# Carregar MLP treinada do Christoph
mlp = MLP(768)
mlp.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
mlp.eval()

def avaliar_imagem(path):
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            score = mlp(features).item()
        return score
    except Exception as e:
        print(f"Erro ao avaliar {path}: {e}")
        return 0

# Extensões suportadas
extensoes = (".jpg", ".jpeg", ".png", ".heic", ".webp")

# Avaliar todas as imagens e salvar notas apenas se forem em pé
resultados = []
for arquivo in tqdm(os.listdir(PASTA_FOTOS)):
    if arquivo.lower().endswith(extensoes):
        caminho = os.path.join(PASTA_FOTOS, arquivo)
        try:
            # Verifica orientação antes de avaliar
            with Image.open(caminho) as img:
                largura, altura = img.size
                if altura <= largura:  # ignora paisagem ou quadradas
                    continue

            nota = avaliar_imagem(caminho)
            resultados.append((caminho, nota))

        except Exception as e:
            print(f"Erro ao abrir {arquivo} para checar orientação: {e}")

# Ordenar por nota (decrescente)
resultados.sort(key=lambda x: x[1], reverse=True)

# Calcular top X%
top_n = int(len(resultados) * TOP_PERCENT)
print(f"Total de imagens em pé: {len(resultados)}, copiando top {top_n} ({TOP_PERCENT*100:.1f}%)")

# Copiar apenas top X%
for caminho, nota in resultados[:top_n]:
    nome_arquivo = os.path.basename(caminho)
    destino = os.path.join(PASTA_DESTINO, nome_arquivo)
    try:
        shutil.copy2(caminho, destino)  # copia com metadados
        print(f"Copiada: {nome_arquivo} (nota {nota:.2f})")
    except Exception as e:
        print(f"Erro ao copiar {nome_arquivo}: {e}")

print(f"Curadoria concluída! {top_n} imagens copiadas para {PASTA_DESTINO}")
