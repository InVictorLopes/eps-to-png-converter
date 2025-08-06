import os
import cv2
import numpy as np
from tqdm import tqdm
import subprocess

# --- CONFIGURAÇÕES GERAIS ---
input_folder = r"C:\ConversorEPS\Entrada"  # Caminho da pasta contendo os arquivos EPS de entrada
output_folder = r"C:\ConversorEPS\Saida"   # Caminho da pasta onde os arquivos PNG convertidos serão salvos
TARGET_SIZE = 1080                         # Tamanho final da imagem em pixels (largura e altura)
RESOLUTION_PPI = 300                       # Resolução da imagem exportada em pontos por polegada (DPI)

os.makedirs(output_folder, exist_ok=True)  # Cria a pasta de saída caso ela não exista

def carregar_eps_com_transparencia(eps_path, temp_png):
    """Converte um arquivo EPS em PNG com fundo transparente e alta qualidade usando Ghostscript."""
    ghostscript_path = r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe"  # Caminho para o executável do Ghostscript
    subprocess.run([
        ghostscript_path, "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha", "-dEPSCrop",
        f"-r{RESOLUTION_PPI}", f"-sOutputFile={temp_png}", eps_path
    ], check=True, capture_output=True, text=True)

def centralizar_e_redimensionar(img):
    """
    Redimensiona a imagem proporcionalmente para caber em uma tela 1080x1080 com fundo transparente.
    A imagem é centralizada e ajustada para ocupar até 85% da área total.
    """
    if img is None:
        return None

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Adiciona canal alfa caso a imagem não tenha

    h, w = img.shape[:2]

    margem = 0.85  # Percentual máximo da área que a imagem ocupará dentro do canvas
    escala = min((TARGET_SIZE * margem) / w, (TARGET_SIZE * margem) / h)  # Escala proporcional baseada na menor dimensão
    novo_w, novo_h = int(w * escala), int(h * escala)  # Novas dimensões redimensionadas

    redimensionado = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_LANCZOS4)  # Redimensiona com interpolação de alta qualidade

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 4), dtype=np.uint8)  # Cria imagem base (canvas) com fundo transparente
    px = (TARGET_SIZE - novo_w) // 2  # Posição horizontal para centralizar
    py = (TARGET_SIZE - novo_h) // 2  # Posição vertical para centralizar
    canvas[py:py + novo_h, px:px + novo_w] = redimensionado  # Coloca a imagem redimensionada no centro do canvas

    return canvas

# --- PROCESSAMENTO DOS ARQUIVOS EPS ---
eps_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".eps")]  # Lista todos os arquivos EPS na pasta de entrada

for filename in tqdm(eps_files, desc="Convertendo e processando arquivos", unit="arquivo"):
    eps_path = os.path.join(input_folder, filename)  # Caminho completo do arquivo EPS
    temp_png = os.path.join(output_folder, "temp.png")  # Caminho do arquivo PNG temporário
    final_png = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")  # Caminho final do arquivo PNG de saída

    try:
        carregar_eps_com_transparencia(eps_path, temp_png)  # Converte EPS em PNG temporário com transparência
        imagem = cv2.imread(temp_png, cv2.IMREAD_UNCHANGED)  # Lê a imagem mantendo o canal alfa
        imagem_final = centralizar_e_redimensionar(imagem)  # Redimensiona e centraliza a imagem no canvas

        if imagem_final is not None:
            cv2.imwrite(final_png, imagem_final)  # Salva a imagem final no formato PNG
        else:
            print(f"   - ERRO: Imagem vazia em {filename}")  # Aviso caso a imagem lida esteja vazia

        os.remove(temp_png)  # Remove o arquivo temporário após o uso

    except Exception as e:
        print(f"Erro ao processar {filename}: {e}")  # Exibe erro se houver falha no processamento do arquivo

print("\n Imagens processadas e salvas como PNG 1080x1080 @ 300ppi com centralização e qualidade melhorada.")

if imagem_final is not None:
    cv2.imwrite(final_png, imagem_final)  # Salva a imagem final no formato PNG

    # Remove bordas transparentes ou de cor uniforme usando ImageMagick
    subprocess.run(["mogrify", "-trim", final_png], check=True)

else:
    print(f"   - ERRO: Imagem vazia em {filename}")
