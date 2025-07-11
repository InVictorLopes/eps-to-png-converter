import os
from tqdm import tqdm
import subprocess
import cv2
import numpy as np

# --- PARÂMETRO CONFIGURÁVEL ---
# Define o preenchimento (em pixels) a ser adicionado em volta do conteúdo na imagem quadrada final.
FINAL_PADDING_PIXELS = 40

# --- CAMINHOS ---
ghostscript_path = r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe"
input_folder = r"C:\ConversorEPS\Entrada"
output_folder = r"C:\ConversorEPS\Saida"

os.makedirs(output_folder, exist_ok=True)

# --- FUNÇÕES DE PROCESSAMENTO ---

def recenter_content(img_data):
    """Recebe dados de imagem, centraliza o conteúdo e retorna os novos dados."""
    try:
        if img_data is None or img_data.shape[2] < 4: return img_data
        alpha_channel = img_data[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)
        if coords is None: return img_data
        
        x, y, w, h = cv2.boundingRect(coords)
        content_crop = img_data[y:y+h, x:x+w]
        orig_h, orig_w = img_data.shape[:2]
        pasted_image = np.zeros_like(img_data)
        paste_x = (orig_w - w) // 2
        paste_y = (orig_h - h) // 2
        pasted_image[paste_y:paste_y+h, paste_x:paste_x+w] = content_crop
        return pasted_image
    except Exception as e:
        print(f"   - ERRO ao centralizar conteúdo: {e}")
        return None

# --- A FUNÇÃO "isolate_central_object" FOI REMOVIDA ---

def resize_to_square_canvas(img_data, padding):
    """
    Recebe dados de uma imagem e a coloca em um novo canvas quadrado (1:1) com padding.
    """
    try:
        if img_data is None: return None
        alpha_channel = img_data[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)
        if coords is None: return None

        x, y, w, h = cv2.boundingRect(coords)
        content_crop = img_data[y:y+h, x:x+w]

        # Determina o tamanho do novo canvas quadrado
        side_len = max(w, h) + padding * 2
        
        # Cria o canvas quadrado e transparente
        square_canvas = np.zeros((side_len, side_len, 4), dtype=np.uint8)

        # Calcula onde colar o logo para que fique centralizado
        paste_x = (side_len - w) // 2
        paste_y = (side_len - h) // 2

        square_canvas[paste_y:paste_y+h, paste_x:paste_x+w] = content_crop
        return square_canvas

    except Exception as e:
        print(f"   - ERRO ao redimensionar para 1:1: {e}")
        return None


# --- LOOP PRINCIPAL SIMPLIFICADO ---
eps_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".eps")]

for filename in tqdm(eps_files, desc="Convertendo e processando arquivos", unit="arquivo"):
    input_path_eps = os.path.join(input_folder, filename)
    output_filename_png = os.path.splitext(filename)[0] + ".png"
    output_path_png = os.path.join(output_folder, output_filename_png)

    print(f"\n--- Processando: {filename} ---")
    try:
        # Etapa 1: Converter para PNG
        print("-> Etapa 1: Convertendo para PNG...")
        subprocess.run([
            ghostscript_path, "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
            "-r300", f"-sOutputFile={output_path_png}", input_path_eps
        ], check=True, capture_output=True, text=True)
        
        # Carrega a imagem convertida UMA VEZ
        img_data = cv2.imread(output_path_png, cv2.IMREAD_UNCHANGED)
        if img_data is None: 
            print("   - ERRO: Falha ao ler o PNG convertido.")
            continue
            
        print("-> Processamento de imagem iniciado...")
        
        # Etapa 2: Centralizar o conteúdo (logo + barras)
        centered_data = recenter_content(img_data)
        
        # --- ETAPA DE ISOLAMENTO REMOVIDA ---

        # Etapa 3: Redimensionar para um canvas quadrado 1:1
        # A função agora recebe os dados diretamente da etapa de centralização
        final_square_data = resize_to_square_canvas(centered_data, FINAL_PADDING_PIXELS)
        
        # Etapa 4: Salvar o resultado final
        if final_square_data is not None:
            cv2.imwrite(output_path_png, final_square_data)
            print(f"-> Sucesso! Imagem final salva em '{output_path_png}'")
        else:
            print("   - ERRO: Não foi possível gerar a imagem final. O arquivo pode estar vazio ou corrompido.")

    except subprocess.CalledProcessError as e:
        print(f"ERRO DE CONVERSÃO com Ghostscript: {e.stderr}")
    except FileNotFoundError:
        print(f"ERRO: Ghostscript não encontrado em: {ghostscript_path}")
        break

print("\nProcessamento concluído!")