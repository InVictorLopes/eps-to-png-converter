import os
import cv2
import numpy as np
from tqdm import tqdm
import subprocess

# --- CONFIGURAÇÕES ---
input_folder = r"C:\ConversorEPS\Entrada"
output_folder = r"C:\ConversorEPS\Saida"
TARGET_SIZE = 1080
RESOLUTION_PPI = 300

os.makedirs(output_folder, exist_ok=True)

def carregar_eps_com_transparencia(eps_path, temp_png):
    """Converte EPS para PNG com transparência mantendo qualidade."""
    ghostscript_path = r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe"
    subprocess.run([
        ghostscript_path, "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha", "-dEPSCrop",
        f"-r{RESOLUTION_PPI}", f"-sOutputFile={temp_png}", eps_path
    ], check=True, capture_output=True, text=True)

def centralizar_e_redimensionar(img):
    """Redimensiona imagem para 1080x1080 centralizada, com fundo transparente e margem extra."""
    if img is None:
        return None

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]

    margem = 0.85  # Reduz a imagem para 85% do espaço disponível
    escala = min((TARGET_SIZE * margem) / w, (TARGET_SIZE * margem) / h)
    novo_w, novo_h = int(w * escala), int(h * escala)

    redimensionado = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 4), dtype=np.uint8)
    px = (TARGET_SIZE - novo_w) // 2
    py = (TARGET_SIZE - novo_h) // 2
    canvas[py:py + novo_h, px:px + novo_w] = redimensionado

    return canvas


# --- PROCESSAMENTO ---
eps_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".eps")]

for filename in tqdm(eps_files, desc="Convertendo e processando arquivos", unit="arquivo"):
    eps_path = os.path.join(input_folder, filename)
    temp_png = os.path.join(output_folder, "temp.png")
    final_png = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

    try:
        carregar_eps_com_transparencia(eps_path, temp_png)
        imagem = cv2.imread(temp_png, cv2.IMREAD_UNCHANGED)
        imagem_final = centralizar_e_redimensionar(imagem)

        if imagem_final is not None:
            cv2.imwrite(final_png, imagem_final)
        else:
            print(f"   - ERRO: Imagem vazia em {filename}")

        os.remove(temp_png)

    except Exception as e:
        print(f"Erro ao processar {filename}: {e}")

print("\n✅ Imagens processadas e salvas como PNG 1080x1080 @ 300ppi com centralização e qualidade melhorada.")
