import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import json
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy, kurtosis
import pandas as pd
import time
import cv2
from sklearn.cluster import KMeans

# Configuração da página Streamlit
st.set_page_config(
    page_title="Mirror Glass - Detector de Fraudes em Imagens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e introdução
st.title("📊 Mirror Glass: Sistema de Detecção de Fraudes em Imagens")
st.markdown("""
Este sistema utiliza técnicas avançadas de visão computacional para:
1. **Detectar imagens duplicadas** ou altamente semelhantes, mesmo com alterações como cortes ou ajustes
2. **Identificar manipulações por IA** que criam texturas artificialmente uniformes em áreas danificadas

### Como funciona?
1. Faça upload das imagens para análise
2. O sistema analisa duplicidade usando SIFT/SSIM e manipulações de textura usando LBP
3. Resultados são exibidos com detalhamento visual e score de naturalidade
""")

# Classe para análise de texturas melhorada
class TextureAnalyzer:
    """
    Classe para análise de texturas usando Local Binary Pattern (LBP).
    Detecta manipulações em imagens automotivas, principalmente restaurações por IA.
    """
    
    def __init__(self, P=8, R=1, block_size=8, threshold=0.10):
        self.P = P  # Número de pontos vizinhos
        self.R = R  # Raio
        self.block_size = block_size  # Tamanho dos blocos para análise
        self.threshold = threshold  # Limiar para textura suspeita
        self.scales = [0.5, 1.0, 2.0]  # Múltiplas escalas para análise
    
    def calculate_lbp(self, image):
        # Converter para escala de cinza e array numpy
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
            
        # Calcular LBP
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        # Calcular histograma de padrões
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalização
        
        return lbp, hist, img_gray
    
    def analyze_texture_variance(self, image):
        """
        Versão especializada para detecção de manipulações por IA em imagens de veículos
        """
        # Converter para formato numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Converter para escala de cinza
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        # 1. Detecção de bordas usando Sobel e Canny
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude do gradiente
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Detecção de bordas com Canny
        edges = cv2.Canny(img_gray, 50, 150)
        
        # 2. Aplicar filtro de mediana para reduzir ruído
        img_filtered = cv2.medianBlur(img_gray, 5)
        
        # 3. Calcular LBP em múltiplas escalas
        lbp_maps = []
        blurred_maps = []
        
        for scale in self.scales:
            # Redimensionar para a escala atual
            if scale != 1.0:
                height, width = img_gray.shape
                new_height, new_width = int(height * scale), int(width * scale)
                img_scaled = cv2.resize(img_gray, (new_width, new_height))
                # Aplicar blurring para simular diferentes níveis de detalhes
                blurred = cv2.GaussianBlur(img_scaled, (5, 5), 0)
                lbp_scaled, _, _ = self.calculate_lbp(blurred)
                # Redimensionar de volta para tamanho original
                lbp_map = cv2.resize(lbp_scaled, (width, height))
            else:
                lbp_map, _, _ = self.calculate_lbp(img_gray)
            
            lbp_maps.append(lbp_map)
            
            # Filtro Gaussiano em diferentes escalas para detectar áreas suspeitas
            for sigma in [1, 3, 5]:
                blurred = cv2.GaussianBlur(img_gray, (sigma*2+1, sigma*2+1), sigma)
                blurred_maps.append(blurred)
        
        # 4. Análise de textura em blocos
        height, width = img_gray.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        gradient_consistency_map = np.zeros((rows, cols))
        edge_density_map = np.zeros((rows, cols))
        blur_consistency_map = np.zeros((rows, cols))
        
        # 5. Analisar em blocos
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                # Extrair blocos
                block_gray = img_gray[i:i+self.block_size, j:j+self.block_size]
                if i < lbp_maps[1].shape[0] - self.block_size and j < lbp_maps[1].shape[1] - self.block_size:
                    block_lbp = lbp_maps[1][i:i+self.block_size, j:j+self.block_size]  # Escala padrão
                else:
                    continue  # Pular blocos que estão fora dos limites
                
                block_gradient = gradient_magnitude[i:i+self.block_size, j:j+self.block_size]
                block_edges = edges[i:i+self.block_size, j:j+self.block_size]
                
                # Calcular entropia LBP (mede aleatoriedade da textura)
                hist, _ = np.histogram(block_lbp, bins=10, range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                max_entropy = np.log(10)
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                
                # Variância da textura (texturas naturais são mais variadas)
                block_variance = np.var(block_lbp) / 255.0
                
                # Consistência do gradiente (gradientes naturais são menos regulares)
                grad_hist, _ = np.histogram(block_gradient, bins=8)
                grad_hist = grad_hist.astype("float")
                grad_hist /= (grad_hist.sum() + 1e-7)
                grad_entropy = entropy(grad_hist)
                grad_consistency = 1.0 - (grad_entropy / np.log(8))  # Normalizado e invertido
                
                # Densidade de bordas (áreas restauradas têm menos bordas naturais)
                edge_density = np.sum(block_edges > 0) / (self.block_size * self.block_size)
                
                # Consistência do blur (resposta a diferentes níveis de borramento)
                blur_responses = []
                for blurred in blurred_maps:
                    blur_block = blurred[i:i+self.block_size, j:j+self.block_size]
                    # Diferença entre original e borrado
                    diff = np.abs(block_gray.astype(float) - blur_block.astype(float)).mean()
                    blur_responses.append(diff)
                
                # O desvio padrão das respostas mede a naturalidade
                # Texturas reais têm resposta mais variada ao blurring
                blur_consistency = 1.0 - min(np.std(blur_responses) / 10.0, 1.0)  # Normalizado e invertido
                
                # Armazenar nos mapas
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
                    gradient_consistency_map[row_idx, col_idx] = grad_consistency
                    edge_density_map[row_idx, col_idx] = edge_density
                    blur_consistency_map[row_idx, col_idx] = blur_consistency
        
        # 6. Análise específica para carros (grandes áreas planas com textura uniforme)
        # Converter para espaço de cor LAB para análise mais perceptual
        if len(image.shape) > 2:
            try:
                lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel = lab_image[:,:,0]  # Canal de luminância
                
                # Detectar áreas de luminância semelhante
                luminance_variance = np.zeros((rows, cols))
                for i in range(0, height - self.block_size + 1, self.block_size):
                    for j in range(0, width - self.block_size + 1, self.block_size):
                        if i < l_channel.shape[0] - self.block_size and j < l_channel.shape[1] - self.block_size:
                            block_l = l_channel[i:i+self.block_size, j:j+self.block_size]
                            row_idx = i // self.block_size
                            col_idx = j // self.block_size
                            if row_idx < rows and col_idx < cols:
                                luminance_variance[row_idx, col_idx] = np.var(block_l)
                
                # Normalizar variância de luminância
                if np.max(luminance_variance) > 0:
                    luminance_variance = cv2.normalize(luminance_variance, None, 0, 1, cv2.NORM_MINMAX)
                
                # Áreas com baixa variância de luminância e baixa entropia de textura
                # são candidatas fortes para manipulação por IA
                flat_surface_map = (1.0 - luminance_variance) * (1.0 - entropy_map)
            except Exception as e:
                # Em caso de erro, criar um mapa vazio
                flat_surface_map = np.zeros_like(entropy_map)
        else:
            flat_surface_map = np.zeros_like(entropy_map)
        
        # 7. Detecção de padrões repetitivos (característico de IA)
        # Implementação corrigida que evita o erro de matchTemplate
        lbp_main = lbp_maps[1]  # Escala padrão
        repetitive_pattern_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if i >= lbp_main.shape[0] - self.block_size or j >= lbp_main.shape[1] - self.block_size:
                    continue  # Pular se fora dos limites
                    
                block = lbp_main[i:i+self.block_size, j:j+self.block_size].copy()
                
                # Verificar se o bloco tem valores válidos
                if np.isfinite(block).all() and np.any(block != 0):
                    try:
                        # Garantir que seja float32 para matchTemplate
                        block_float = block.astype(np.float32)
                        
                        # Calcular a autocorrelação de maneira simplificada e robusta
                        # Criar versão suavizada para análise de textura
                        block_smooth = cv2.GaussianBlur(block_float, (3, 3), 0)
                        
                        # Calcular a variação da textura de forma mais robusta
                        texel_variation = np.std(block_smooth) / np.mean(block_smooth) if np.mean(block_smooth) > 0 else 0
                        
                        # Texturas artificiais têm variação mais baixa (fator invertido)
                        repetitive_score = max(0, 1.0 - min(texel_variation * 2, 1.0))
                    except Exception as e:
                        # Em caso de erro, atribuir valor médio neutro
                        repetitive_score = 0.5
                else:
                    repetitive_score = 0.5
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                if row_idx < rows and col_idx < cols:
                    repetitive_pattern_map[row_idx, col_idx] = repetitive_score
        
        # 8. Combinar todas as métricas para score final
        # Pesos de cada métrica (ajustados especificamente para carros)
        weights = {
            'entropy': 0.15,            # Aleatoriedade da textura
            'variance': 0.10,           # Variação da textura
            'gradient': 0.10,           # Regularidade do gradiente
            'edge_density': 0.15,       # Densidade de bordas naturais
            'blur_consistency': 0.20,   # Resposta a diferentes níveis de blur
            'flat_surface': 0.20,       # Áreas planas com textura uniforme
            'repetitive': 0.10          # Padrões repetitivos
        }
        
        # Invertemos algumas métricas para que valores maiores indiquem manipulação
        naturalness_map = (
            (1.0 - weights['entropy'] * (1.0 - entropy_map)) *          # Entropia (maior é melhor)
            (1.0 - weights['variance'] * (1.0 - variance_map)) *        # Variância (maior é melhor)
            (1.0 - weights['gradient'] * gradient_consistency_map) *    # Consistência do gradiente (menor é melhor)
            (1.0 - weights['edge_density'] * (1.0 - edge_density_map)) * # Densidade de bordas (maior é melhor)
            (1.0 - weights['blur_consistency'] * blur_consistency_map) * # Consistência do blur (menor é melhor)
            (1.0 - weights['flat_surface'] * flat_surface_map) *        # Superfícies planas artificiais (menor é melhor)
            (1.0 - weights['repetitive'] * repetitive_pattern_map)      # Padrões repetitivos (menor é melhor)
        )
        
        # Normalizar mapa para visualização
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # 9. Aplicar threshold mais baixo em áreas de veículos (carroceria)
        # As áreas planas precisam de mais sensibilidade
        vehicle_regions = flat_surface_map > 0.5  # Áreas prováveis de carroceria
        adjusted_threshold_map = np.ones_like(norm_naturalness_map) * self.threshold
        adjusted_threshold_map[vehicle_regions] = self.threshold * 0.7  # 30% mais sensível
        
        # Cria máscara de áreas suspeitas usando threshold adaptativo
        suspicious_mask = norm_naturalness_map < adjusted_threshold_map
        
        # 10. Calcular score de naturalidade (0-100)
        naturalness_score = int(np.mean(norm_naturalness_map) * 100)
        
        # Penalizar scores para imagens de veículos com grandes áreas suspeitas
        if np.sum(vehicle_regions) > 0.2 * rows * cols:  # Se mais de 20% da imagem for veículo
            if np.mean(suspicious_mask[vehicle_regions]) > 0.3:  # Se mais de 30% das áreas de veículo forem suspeitas
                naturalness_score = max(10, naturalness_score - 30)  # Reduzir score em 30 pontos (mínimo 10)
        
        # 11. Converte para mapa de calor para visualização
        heatmap = cv2.applyColorMap(
            (norm_naturalness_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Criar mapas individuais para visualização
        # Normalizar e converter para mapa de calor
        def create_heatmap(data):
            norm_data = cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)
            return cv2.applyColorMap((norm_data * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Criar heatmaps com tratamento de erro
        try:
            entropy_heatmap = create_heatmap(entropy_map)
            variance_heatmap = create_heatmap(variance_map)
            gradient_heatmap = create_heatmap(gradient_consistency_map)
            edge_heatmap = create_heatmap(edge_density_map)
            blur_heatmap = create_heatmap(blur_consistency_map)
            flat_surface_heatmap = create_heatmap(flat_surface_map)
            repetitive_heatmap = create_heatmap(repetitive_pattern_map)
        except Exception as e:
            # Em caso de erro, criar mapas de calor vazios
            empty_map = np.zeros((10, 10, 3), dtype=np.uint8)
            entropy_heatmap = variance_heatmap = gradient_heatmap = edge_heatmap = empty_map
            blur_heatmap = flat_surface_heatmap = repetitive_heatmap = empty_map
        
        return {
            "naturalness_map": norm_naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "entropy_map": entropy_map,
            "variance_map": variance_map,
            "gradient_map": gradient_consistency_map,
            "edge_map": edge_density_map,
            "blur_map": blur_consistency_map,
            "flat_surface_map": flat_surface_map,
            "repetitive_map": repetitive_pattern_map,
            "entropy_heatmap": entropy_heatmap,
            "variance_heatmap": variance_heatmap,
            "gradient_heatmap": gradient_heatmap,
            "edge_heatmap": edge_heatmap,
            "blur_heatmap": blur_heatmap,
            "flat_surface_heatmap": flat_surface_heatmap,
            "repetitive_heatmap": repetitive_heatmap
        }
    
    def classify_naturalness(self, score):
        """
        Classificação ajustada para maior sensibilidade
        """
        if score <= 45:  # Limiar mais alto para manipulação
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:  # Faixa mais ampla para suspeita
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, analysis_results):
        # Converter para numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Extrair resultados
        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]
        
        # Redimensionar para o tamanho da imagem original
        height, width = image.shape[:2]
        
        # Redimensionar naturalness_map e suspicious_mask
        naturalness_map_resized = cv2.resize(naturalness_map, 
                                           (width, height), 
                                           interpolation=cv2.INTER_LINEAR)
        
        mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), 
                                 (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Criar mapa de calor
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        
        # Criar overlay com 40% de transparência
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Destacar áreas suspeitas com contorno
        highlighted = overlay.copy()
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Desenhar retângulos em áreas suspeitas maiores
        for contour in contours:
            # Filtrar contornos muito pequenos (ruído)
            area = cv2.contourArea(contour)
            if area > 50:  # Reduzido para detectar áreas menores
                x, y, w, h = cv2.boundingRect(contour)
                # Desenhar retângulo roxo
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (128, 0, 128), 2)
        
        # Classificar resultado
        category, description = self.classify_naturalness(score)
        
        # Adicionar informações na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        # Criar visualização de mapas individuais
        detailed_maps = {}
        map_names = [
            "entropy_heatmap", "variance_heatmap", "gradient_heatmap", 
            "edge_heatmap", "blur_heatmap", "flat_surface_heatmap", "repetitive_heatmap"
        ]
        
        for map_name in map_names:
            if map_name in analysis_results:
                map_data = analysis_results[map_name]
                if map_data is not None and map_data.size > 0:
                    try:
                        map_resized = cv2.resize(map_data, (width, height), 
                                               interpolation=cv2.INTER_LINEAR)
                        detailed_maps[map_name] = map_resized
                    except Exception as e:
                        pass  # Ignora mapas com problemas
        
        return highlighted, heatmap, detailed_maps
    
    def analyze_image(self, image):
        # Inicializa um relatório padrão com valores seguros
        report = {
            "score": 0,
            "category": "Erro",
            "description": "Falha na análise inicial",
            "percentual_suspeito": 0,
            "visual_report": None,
            "heatmap": None,
            "detailed_maps": {},
            "analysis_results": {}
        }
        
        try:
            # Analisar textura
            analysis_results = self.analyze_texture_variance(image)
            if analysis_results is None:
                raise ValueError("analyze_texture_variance retornou None")
            report["analysis_results"] = analysis_results

            # Gerar visualização
            visual_report, heatmap, detailed_maps = self.generate_visual_report(image, analysis_results)
            report["visual_report"] = visual_report
            report["heatmap"] = heatmap
            report["detailed_maps"] = detailed_maps if detailed_maps is not None else {}

            # Classificar o resultado
            score = analysis_results.get("naturalness_score", 0)
            report["score"] = score
            category, description = self.classify_naturalness(score)
            report["category"] = category
            report["description"] = description

            # Calcular percentual de áreas suspeitas
            suspicious_mask = analysis_results.get("suspicious_mask")
            if suspicious_mask is not None:
                report["percentual_suspeito"] = float(np.mean(suspicious_mask) * 100)
            else:
                report["percentual_suspeito"] = 0.0

            return report
        except Exception as e:
            # Atualiza a descrição do erro no report padrão
            report["description"] = f"Erro na análise de imagem: {str(e)}"
            # Retorna o dicionário de erro padronizado
            return report

# Barra lateral com controles
st.sidebar.header("⚙️ Configurações")

# Seleção de modo
modo_analise = st.sidebar.radio(
   "Modo de Análise",
   ["Duplicidade", "Manipulação por IA", "Análise Completa"],
   help="Escolha o tipo de análise a ser realizada"
)

# Configurações para detecção de duplicidade
if modo_analise in ["Duplicidade", "Análise Completa"]:
   st.sidebar.subheader("Configurações de Duplicidade")
   limiar_similaridade = st.sidebar.slider(
       "Limiar de Similaridade (%)", 
       min_value=30, 
       max_value=100, 
       value=50, 
       help="Imagens com similaridade acima deste valor serão consideradas possíveis duplicatas"
   )
   limiar_similaridade = limiar_similaridade / 100  # Converter para decimal

   metodo_deteccao = st.sidebar.selectbox(
       "Método de Detecção",
       ["SIFT (melhor para recortes)", "SSIM + SIFT", "SSIM"],
       help="Escolha o método para detectar imagens similares"
   )

# Configurações para detecção de manipulação por IA
if modo_analise in ["Manipulação por IA", "Análise Completa"]:
   st.sidebar.subheader("Configurações de Análise de Textura")
   limiar_naturalidade = st.sidebar.slider(
       "Limiar de Naturalidade", 
       min_value=30, 
       max_value=80, 
       value=50, 
       help="Score abaixo deste valor indica possível manipulação por IA"
   )
   
   tamanho_bloco = st.sidebar.slider(
       "Tamanho do Bloco", 
       min_value=8, 
       max_value=32, 
       value=20, 
       step=4,
       help="Tamanho do bloco para análise de textura (menor = mais sensível)"
   )
   
   threshold_lbp = st.sidebar.slider(
       "Sensibilidade LBP", 
       min_value=0.1, 
       max_value=0.5, 
       value=0.50, 
       step=0.05,
       help="Limiar para detecção de áreas suspeitas (menor = mais sensível)"
   )

# Funções para processamento de imagens - DUPLICIDADE
def preprocessar_imagem(img, tamanho=(300, 300)):
   try:
       # Redimensionar
       img_resize = img.resize(tamanho)
       # Converter para escala de cinza para SSIM
       img_gray = img_resize.convert('L')
       # Converter para array numpy
       img_array = np.array(img_gray)
       # Normalizar valores para [0, 1]
       img_array = img_array / 255.0
       # Converter para CV2 formato (para SIFT)
       img_cv = np.array(img_resize)
       img_cv = img_cv[:, :, ::-1].copy()  # RGB para BGR
       return img_array, img_cv
   except Exception as e:
       st.error(f"Erro ao processar imagem: {e}")
       return None, None

def calcular_similaridade_ssim(img1, img2):
   try:
       # Garantir que as imagens tenham o mesmo tamanho
       if img1.shape != img2.shape:
           img2 = resize(img2, img1.shape)
       
       # Calcular SSIM com data_range especificado
       score = ssim(img1, img2, data_range=1.0)
       return score
   except Exception as e:
       st.error(f"Erro ao calcular similaridade SSIM: {e}")
       return 0

def calcular_similaridade_sift(img1_cv, img2_cv):
   try:
       # Converter para escala de cinza
       img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
       img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
       
       # Inicializar o detector SIFT
       sift = cv2.SIFT_create()
       
       # Detectar keypoints e descritores
       kp1, des1 = sift.detectAndCompute(img1_gray, None)
       kp2, des2 = sift.detectAndCompute(img2_gray, None)
       
       # Se não houver descritores suficientes, retorna 0
       if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
           return 0
           
       # Usar o matcher FLANN
       FLANN_INDEX_KDTREE = 1
       index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
       search_params = dict(checks=50)
       flann = cv2.FlannBasedMatcher(index_params, search_params)
       
       # Encontrar os 2 melhores matches para cada descritor
       matches = flann.knnMatch(des1, des2, k=2)
       
       # Filtrar bons matches usando o teste de proporção de Lowe
       good_matches = []
       for m, n in matches:
           if m.distance < 0.7 * n.distance:
               good_matches.append(m)
       
       # Calcular a similaridade baseada no número de bons matches
       max_matches = min(len(kp1), len(kp2))
       if max_matches == 0:
           return 0
           
       similarity = len(good_matches) / max_matches
       
       # Normalizar para evitar valores muito baixos
       if similarity < 0.05:
           adjusted_similarity = 0
       else:
           # Expandir valores pequenos para uma escala mais ampla
           adjusted_similarity = min(1.0, similarity * 2)
       
       return adjusted_similarity
       
   except Exception as e:
       st.error(f"Erro ao calcular similaridade SIFT: {e}")
       return 0

def calcular_similaridade_combinada(img1_gray, img2_gray, img1_cv, img2_cv):
   try:
       # Calcular similaridade usando ambos os métodos
       sim_ssim = calcular_similaridade_ssim(img1_gray, img2_gray)
       sim_sift = calcular_similaridade_sift(img1_cv, img2_cv)
       
       # A similaridade combinada é a média ponderada dos dois valores
       # SIFT tem mais peso para detectar recortes
       return (sim_ssim * 0.3) + (sim_sift * 0.7)
   except Exception as e:
       st.error(f"Erro ao calcular similaridade combinada: {e}")
       return 0

def get_csv_download_link(df, filename, text):
   csv = df.to_csv(index=False)
   b64 = base64.b64encode(csv.encode()).decode()
   href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
   return href

def get_image_download_link(img, filename, text):
   # Converter para PIL Image se for numpy array
   if isinstance(img, np.ndarray):
       img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
   else:
       img_pil = img
       
   # Salvar em buffer
   buf = io.BytesIO()
   img_pil.save(buf, format='JPEG')
   buf.seek(0)
   
   # Codificar para base64
   img_str = base64.b64encode(buf.read()).decode()
   href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
   
   return href

def visualizar_duplicatas(imagens, nomes, duplicatas, limiar):
   if not duplicatas:
       st.info("Nenhuma duplicata encontrada com o limiar de similaridade atual.")
       return None
   
   # Criar DataFrame para relatório
   relatorio_dados = []
   
   # Para cada grupo de duplicatas
   for idx, (img_orig_idx, similares) in enumerate(duplicatas.items()):
       st.write("---")
       st.subheader(f"Grupo de Duplicatas #{idx+1}")
       
       # Layout para imagem original e suas duplicatas
       cols = st.columns(min(len(similares) + 1, 4))  # Limita a 4 colunas por linha
       
       # Mostrar imagem original
       with cols[0]:
           st.image(imagens[img_orig_idx], caption=f"Original: {nomes[img_orig_idx]}", width=200)
       
       # Mostrar duplicatas
       for i, (similar_idx, similaridade) in enumerate(similares):
           col_index = (i + 1) % len(cols)
           
           # Se precisar de uma nova linha
           if col_index == 0 and i > 0:
               st.write("")  # Linha em branco
               cols = st.columns(min(len(similares) - i + 1, 4))
           
           with cols[col_index]:
               st.image(imagens[similar_idx], width=200)
               caption = f"{nomes[similar_idx]}\nSimilaridade: {similaridade:.2f}"
               st.caption(caption)
               
               # Destacar em verde se acima do limiar
               if similaridade >= limiar:
                   st.success("DUPLICATA DETECTADA")
               
               # Adicionar ao relatório
               relatorio_dados.append({
                   "Arquivo Original": nomes[img_orig_idx],
                   "Arquivo Duplicado": nomes[similar_idx],
                   "Similaridade (%)": round(similaridade * 100, 2)
               })
   
   # Criar DataFrame do relatório
   if relatorio_dados:
       df_relatorio = pd.DataFrame(relatorio_dados)
       return df_relatorio
   return None

# Função principal para detectar duplicatas
def detectar_duplicatas(imagens, nomes, limiar=0.5, metodo="SIFT (melhor para recortes)"):
   # Mostrar progresso
   progress_bar = st.progress(0)
   status_text = st.empty()
   
   # Processar imagens
   status_text.text("Extraindo características das imagens...")
   arrays_processados_gray = []  # Para SSIM
   arrays_processados_cv = []    # Para SIFT
   indices_validos = []
   
   for i, img in enumerate(imagens):
       # Atualizar barra de progresso
       progress = (i + 1) / len(imagens)
       progress_bar.progress(progress)
       status_text.text(f"Processando imagem {i+1} de {len(imagens)}: {nomes[i]}")
       
       # Preprocessar imagem
       img_array_gray, img_array_cv = preprocessar_imagem(img)
       if img_array_gray is not None:
           arrays_processados_gray.append(img_array_gray)
           arrays_processados_cv.append(img_array_cv)
           indices_validos.append(i)
   
   if not arrays_processados_gray:
       status_text.error("Nenhuma imagem válida para processamento.")
       progress_bar.empty()
       return None
   
   # Calcular similaridades
   status_text.text("Comparando imagens e buscando duplicatas...")
   duplicatas = {}  # {índice_original: [(índice_similar, similaridade), ...]}
   
   total_comparacoes = len(arrays_processados_gray) * (len(arrays_processados_gray) - 1) // 2
   comparacao_atual = 0
   
   for i in range(len(arrays_processados_gray)):
       similares = []
       for j in range(len(arrays_processados_gray)):
           # Não comparar uma imagem com ela mesma
           if i != j:
               comparacao_atual += 1
               
               # Atualizar progresso de maneira mais segura
               if total_comparacoes > 0:
                   # Certificar que o progresso sempre está entre 0 e 1
                   progress = min(max(comparacao_atual / total_comparacoes, 0.0), 1.0)
                   progress_bar.progress(progress)
               
               # Calcular similaridade com base no método selecionado
               if metodo == "SSIM":
                   similaridade = calcular_similaridade_ssim(
                       arrays_processados_gray[i], 
                       arrays_processados_gray[j]
                   )
               elif metodo == "SIFT (melhor para recortes)":
                   similaridade = calcular_similaridade_sift(
                       arrays_processados_cv[i], 
                       arrays_processados_cv[j]
                   )
               else:  # SSIM + SIFT
                   similaridade = calcular_similaridade_combinada(
                       arrays_processados_gray[i], 
                       arrays_processados_gray[j],
                       arrays_processados_cv[i], 
                       arrays_processados_cv[j]
                   )
               
               # Se acima do limiar, adicionar como duplicata
               if similaridade >= limiar:
                   similares.append((indices_validos[j], similaridade))
       
       # Se encontrou duplicatas, adicionar à lista
       if similares:
           duplicatas[indices_validos[i]] = similares
   
   progress_bar.empty()
   status_text.text("Processamento concluído!")
   
   return duplicatas

# Funções para análise de manipulação por IA
def analisar_manipulacao_ia(imagens, nomes, limiar_naturalidade=50, tamanho_bloco=16, threshold=0.35):
    # Inicializar analisador de textura com parâmetros atualizados
    analyzer = TextureAnalyzer(P=8, R=1, block_size=tamanho_bloco, threshold=threshold)
    
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Resultados
    resultados = []
    
    # Processar cada imagem individualmente
    for i, img in enumerate(imagens):
        # Atualizar barra de progresso
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Analisando textura da imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        try:
            # Analisar imagem individualmente
            report = analyzer.analyze_image(img)
            
            # Validação adicional para garantir que report não é None
            if report is None:
                st.error(f"Erro crítico: analyze_image retornou None para {nomes[i]}")
                resultados.append({
                    "indice": i, 
                    "nome": nomes[i], 
                    "score": 0,
                    "categoria": "Erro Crítico", 
                    "descricao": "Falha interna na análise",
                    "percentual_suspeito": 0,
                    "visual_report": None, 
                    "heatmap": None, 
                    "detailed_maps": {}
                })
                continue  # Pula para a próxima imagem
            
            # Adicionar informações ao relatório (agora com acesso mais seguro)
            resultados.append({
                "indice": i,
                "nome": nomes[i],
                "score": report.get("score", 0),
                "categoria": report.get("category", "Erro"),
                "descricao": report.get("description", "N/A"),
                "percentual_suspeito": report.get("percentual_suspeito", 0),
                "visual_report": report.get("visual_report"),
                "heatmap": report.get("heatmap"),
                "detailed_maps": report.get("detailed_maps", {})
            })
        except Exception as e:
            st.error(f"Erro ao analisar imagem {nomes[i]}: {str(e)}")
            # Adicionar um relatório vazio para manter a consistência
            resultados.append({
                "indice": i,
                "nome": nomes[i],
                "score": 0,
                "categoria": "Erro na análise",
                "descricao": f"Erro: {str(e)}",
                "percentual_suspeito": 0,
                "visual_report": None,
                "heatmap": None,
                "detailed_maps": {}
            })
    
    progress_bar.empty()
    status_text.text("Análise de textura concluída!")
    
    return resultados

# Função para exibir resultados da análise de textura
def exibir_resultados_textura(resultados):
    if not resultados:
        st.info("Nenhum resultado de análise de textura disponível.")
        return None
    
    # Criar DataFrame para relatório
    relatorio_dados = []
    
    # Para cada imagem analisada
    for res in resultados:
        # Adicionar cabeçalho
        st.write("---")
        st.subheader(f"Análise de Textura: {res['nome']}")
        
        # Verificar se tivemos erro na análise
        if res["visual_report"] is None:
            st.error(f"❌ Erro na análise: {res['descricao']}")
            continue
        
        # Layout para exibir resultados padrão
        col1, col2 = st.columns(2)
        
        # Coluna 1: Imagem original e informações
        with col1:
            st.image(res["visual_report"], caption=f"Análise de Textura - {res['nome']}", use_column_width=True)
            
            # Adicionar métricas
            st.metric("Score de Naturalidade", res["score"])
            
            # Status baseado no score
            if res["score"] <= 45:
                st.error(f"⚠️ {res['categoria']}: {res['descricao']}")
            elif res["score"] <= 70:
                st.warning(f"⚠️ {res['categoria']}: {res['descricao']}")
            else:
                st.success(f"✅ {res['categoria']}: {res['descricao']}")
                
            # Download da imagem analisada
            st.markdown(
                get_image_download_link(
                    res["visual_report"], 
                    f"analise_{res['nome'].replace(' ', '_')}.jpg",
                    "📥 Baixar Imagem Analisada"
                ),
                unsafe_allow_html=True
            )
        
        # Coluna 2: Mapa de calor e detalhes
        with col2:
            st.image(res["heatmap"], caption="Mapa de Calor LBP", use_column_width=True)
            
            st.write("### Detalhes da Análise")
            percentual = res['percentual_suspeito']
            if percentual > 60:
                st.error(f"🚨 **ÁREAS SUSPEITAS: {percentual:.2f}% da imagem** - ALTO RISCO!")
            elif percentual > 30:
                st.warning(f"⚠️ **ÁREAS SUSPEITAS: {percentual:.2f}% da imagem** - ATENÇÃO!")
            else:
                st.write(f"- **Áreas suspeitas:** {percentual:.2f}% da imagem")
            st.write(f"- **Interpretação:** {res['descricao']}")
            st.write("- **Legenda do Mapa de Calor:**")
            st.write("  - Azul: Texturas naturais (alta variabilidade)")
            st.write("  - Vermelho: Texturas artificiais (baixa variabilidade)")
            st.write("  - Retângulos roxos: Áreas com maior probabilidade de manipulação")
        
        # Mostrar mapas detalhados se disponíveis
        if "detailed_maps" in res and res["detailed_maps"] is not None and len(res["detailed_maps"]) > 0:
            with st.expander("Ver Análise Detalhada por Métrica"):
                st.write("Cada mapa destaca um aspecto diferente da análise de textura:")
                
                # Mostrar mapas em pares (2 colunas)
                map_titles = {
                    "entropy_heatmap": "Entropia (aleatoriedade)",
                    "variance_heatmap": "Variância (uniformidade)",
                    "gradient_heatmap": "Gradiente (bordas)",
                    "edge_heatmap": "Densidade de Bordas",
                    "blur_heatmap": "Resposta ao Blur",
                    "flat_surface_heatmap": "Superfícies Planas",
                    "repetitive_heatmap": "Padrões Repetitivos"
                }
                
                # Dividir em várias linhas de 2 colunas
                maps_to_show = []
                for map_name, title in map_titles.items():
                    if map_name in res["detailed_maps"] and res["detailed_maps"][map_name] is not None:
                        maps_to_show.append((map_name, title))
                
                # Mostrar em pares
                for i in range(0, len(maps_to_show), 2):
                    map_cols = st.columns(2)
                    
                    # Primeiro mapa do par
                    with map_cols[0]:
                        map_name, title = maps_to_show[i]
                        if res["detailed_maps"][map_name] is not None:
                            st.image(res["detailed_maps"][map_name], caption=title, use_column_width=True)
                        else:
                            st.warning(f"Mapa de {title} não disponível")

                    # Segundo mapa do par (se houver)
                    if i + 1 < len(maps_to_show):
                        with map_cols[1]:
                            map_name, title = maps_to_show[i + 1]
                            if res["detailed_maps"][map_name] is not None:
                                st.image(res["detailed_maps"][map_name], caption=title, use_column_width=True)
                            else:
                                st.warning(f"Mapa de {title} não disponível")
        
        # Adicionar ao relatório
        relatorio_dados.append({
            "Arquivo": res["nome"],
            "Score de Naturalidade": res["score"],
            "Categoria": res["categoria"],
            "Percentual Suspeito (%)": round(res["percentual_suspeito"], 2)
        })
    
    # Criar DataFrame do relatório
    if relatorio_dados:
        st.write("---")
        st.write("### Resumo da Análise de Textura")
        df_relatorio = pd.DataFrame(relatorio_dados)
        st.dataframe(df_relatorio)
        
        # Opção para download do relatório
        nome_arquivo = f"relatorio_texturas_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        st.markdown(
            get_csv_download_link(df_relatorio, nome_arquivo, "📥 Baixar Relatório CSV"),
            unsafe_allow_html=True
        )
        
        return df_relatorio
    return None

# Função para converter numpy arrays para listas (para JSON)
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj.items()]
    else:
        return obj

# Função para gerar JSON resumido
def gerar_json_resumido(dados, tipo_analise):
    if tipo_analise == "Duplicidade":
        if dados:
            resumo = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tipo_analise": "Duplicidade",
                "metodo_usado": "SSIM + SIFT",
                "total_grupos_duplicatas": len(dados),
                "total_duplicatas_encontradas": sum(len(similares) for similares in dados.values()),
                "resumo_grupos": []
            }
            
            for img_orig_idx, similares in dados.items():
                grupo_resumo = {
                    "imagem_original_indice": img_orig_idx,
                    "quantidade_duplicatas": len(similares),
                    "maior_similaridade": max([sim for _, sim in similares]) if similares else 0,
                    "menor_similaridade": min([sim for _, sim in similares]) if similares else 0
                }
                resumo["resumo_grupos"].append(grupo_resumo)
            
            return resumo
        else:
            return {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tipo_analise": "Duplicidade",
                "total_grupos_duplicatas": 0,
                "total_duplicatas_encontradas": 0,
                "resultado": "Nenhuma duplicata encontrada"
            }
    
    elif tipo_analise == "Manipulação por IA":
        if dados:
            # Contar categorias
            manipuladas = sum(1 for item in dados if item["score"] <= 45)
            suspeitas = sum(1 for item in dados if 45 < item["score"] <= 70)
            naturais = sum(1 for item in dados if item["score"] > 70)
            
            resumo = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tipo_analise": "Manipulação por IA",
                "total_imagens_analisadas": len(dados),
                "estatisticas": {
                    "manipuladas": manipuladas,
                    "suspeitas": suspeitas,
                    "naturais": naturais
                },
                "score_medio": round(np.mean([item["score"] for item in dados]), 2),
                "resumo_por_imagem": [
                    {
                        "nome": item["nome"],
                        "score": item["score"],
                        "categoria": item["categoria"],
                        "percentual_suspeito": round(item["percentual_suspeito"], 2)
                    }
                    for item in dados
                ]
            }
            return resumo
        else:
            return {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tipo_analise": "Manipulação por IA",
                "total_imagens_analisadas": 0,
                "resultado": "Nenhuma imagem analisada"
            }

# Interface principal
st.markdown("### 🔹 Passo 1: Carregar Imagens")
uploaded_files = st.file_uploader(
    "Faça upload das imagens para análise", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagens carregadas")
    
    # Criar botão para iniciar processamento
    if st.button("🚀 Iniciar Análise", key="iniciar_analise"):
        # Carregar imagens
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir a imagem {arquivo.name}: {e}")
        
        # Processar de acordo com o modo selecionado
        if modo_analise in ["Duplicidade", "Análise Completa"]:
            try:
                st.markdown("## 🔍 Análise de Duplicidade")
                duplicatas = detectar_duplicatas(imagens, nomes, limiar_similaridade, metodo_deteccao)
                
                # Visualizar resultados de duplicidade
                if duplicatas:
                    # Estatísticas
                    total_duplicatas = sum(len(similares) for similares in duplicatas.values())
                    st.metric("Total de possíveis duplicatas encontradas", total_duplicatas)
                    
                    # Visualizar duplicatas
                    df_relatorio = visualizar_duplicatas(imagens, nomes, duplicatas, limiar_similaridade)
                    
                    # Gerar relatório
                    if df_relatorio is not None:
                        st.markdown("### 🔹 Relatório de Duplicatas")
                        st.dataframe(df_relatorio)
                        
                        # Opção para download do relatório
                        nome_arquivo = f"relatorio_duplicatas_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                        st.markdown(get_csv_download_link(df_relatorio, nome_arquivo, 
                                                     "📥 Baixar Relatório CSV"), unsafe_allow_html=True)
                    
                    # JSON Resumido
                    with st.expander("📄 Ver JSON Resumido - Duplicidade"):
                        json_resumido = gerar_json_resumido(duplicatas, "Duplicidade")
                        json_str = json.dumps(json_resumido, indent=2, ensure_ascii=False)
                        st.code(json_str, language='json')
                        
                        st.download_button(
                            label="📥 Baixar JSON Resumido",
                            data=json_str,
                            file_name=f"resumo_duplicatas_{time.strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.warning("Nenhuma duplicata encontrada com o limiar atual. Tente reduzir o limiar de similaridade.")
                    
                    # JSON vazio para nenhuma duplicata
                    with st.expander("📄 Ver JSON Resumido - Duplicidade"):
                        json_resumido = gerar_json_resumido(None, "Duplicidade")
                        json_str = json.dumps(json_resumido, indent=2, ensure_ascii=False)
                        st.code(json_str, language='json')
                        
            except Exception as e:
                st.error(f"Erro durante a detecção de duplicatas: {str(e)}")
        
        # Análise de manipulação por IA
        if modo_analise in ["Manipulação por IA", "Análise Completa"]:
            try:
                st.markdown("## 🤖 Análise de Manipulação por IA")
                resultados_textura = analisar_manipulacao_ia(
                    imagens, 
                    nomes, 
                    limiar_naturalidade,
                    tamanho_bloco,
                    threshold_lbp
                )
                
                # Exibir resultados
                exibir_resultados_textura(resultados_textura)
                
                # JSON Resumido
                with st.expander("📄 Ver JSON Resumido - Manipulação por IA"):
                    json_resumido = gerar_json_resumido(resultados_textura, "Manipulação por IA")
                    json_str = json.dumps(json_resumido, indent=2, ensure_ascii=False)
                    st.code(json_str, language='json')
                    
                    st.download_button(
                        label="📥 Baixar JSON Resumido",
                        data=json_str,
                        file_name=f"resumo_manipulacao_ia_{time.strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"Erro durante a análise de textura: {str(e)}")

else:
    # Mostrar exemplo quando não há imagens carregadas
    st.info("Faça upload de imagens para começar a detecção de fraudes.")
    
    # Adicionar imagens de exemplo
    if st.button("🔍 Ver exemplos de detecção", key="ver_exemplos"):
        st.write("### Exemplos de Análise de Textura")
        
        # Criar colunas para exibir os exemplos
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/400x300?text=Original", caption="Imagem Original")
            st.write("Score de Naturalidade: 85")
            st.success("✅ Textura Natural")
            
        with col2:
            st.image("https://via.placeholder.com/400x300?text=Manipulada+por+IA", caption="Imagem Manipulada por IA")
            st.write("Score de Naturalidade: 25")
            st.error("⚠️ Alta chance de manipulação")
            
        st.write("### Exemplo de Detecção de Duplicidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/400x300?text=Original", caption="Imagem Original")
            
        with col2:
            st.image("https://via.placeholder.com/400x300?text=Duplicata+Recortada", caption="Duplicata (Recortada)")
            st.write("Similaridade: 0.78")
            st.success("DUPLICATA DETECTADA")
        
        # Exemplo de JSON Resumido
        with st.expander("📄 Exemplo de JSON Resumido"):
            exemplo_json = {
                "timestamp": "2025-06-06 14:30:00",
                "tipo_analise": "Análise Completa",
                "duplicidade": {
                    "total_grupos_duplicatas": 1,
                    "total_duplicatas_encontradas": 1,
                    "resumo_grupos": [
                        {
                            "imagem_original_indice": 0,
                            "quantidade_duplicatas": 1,
                            "maior_similaridade": 0.85,
                            "menor_similaridade": 0.85
                        }
                    ]
                },
                "manipulacao_ia": {
                    "total_imagens_analisadas": 2,
                    "estatisticas": {
                        "manipuladas": 1,
                        "suspeitas": 0,
                        "naturais": 1
                    },
                    "score_medio": 55.0,
                    "resumo_por_imagem": [
                        {
                            "nome": "exemplo1.jpg",
                            "score": 75,
                            "categoria": "Textura natural",
                            "percentual_suspeito": 5.2
                        },
                        {
                            "nome": "exemplo2.jpg",
                            "score": 35,
                            "categoria": "Alta chance de manipulação",
                            "percentual_suspeito": 45.8
                        }
                    ]
                }
            }
            
            json_str = json.dumps(exemplo_json, indent=2, ensure_ascii=False)
            st.code(json_str, language='json')

# Rodapé
st.markdown("---")
st.markdown("### Como interpretar os resultados")

# Explicação sobre duplicidade
if modo_analise in ["Duplicidade", "Análise Completa"]:
    st.write("""
    **Análise de Duplicidade:**
    - **Similaridade 100%**: Imagens idênticas
    - **Similaridade >90%**: Praticamente idênticas (possivelmente recortadas ou com filtros)
    - **Similaridade 70-90%**: Muito semelhantes (potenciais duplicatas)
    - **Similaridade 50-70%**: Semelhantes (verificar manualmente)
    - **Similaridade 30-50%**: Possivelmente relacionadas (verificar com atenção)
    - **Similaridade <30%**: Provavelmente não são duplicatas
    """)

# Explicação sobre análise de textura
if modo_analise in ["Manipulação por IA", "Análise Completa"]:
    st.write("""
    **Análise de Manipulação por IA:**
    - **Score 0-45**: Alta probabilidade de manipulação por IA  
    - **Score 46-70**: Textura suspeita, requer verificação manual
    - **Score 71-100**: Textura natural, baixa probabilidade de manipulação
    
    **Como funciona:**
    - **Análise multiescala**: Examina a imagem em diferentes níveis de zoom
    - **Entropia**: Detecta falta de aleatoriedade natural em texturas
    - **Variância**: Identifica uniformidade excessiva (típica de IA)
    - **Densidade de bordas**: Áreas manipuladas têm menos bordas naturais
    - **Resposta ao blur**: Texturas reais respondem de forma diferente ao borramento
    - **Superfícies planas**: Detecta áreas grandes com textura artificial uniforme

    O mapa de calor mostra áreas com baixa variância de textura (vermelho) típicas 
    de restaurações por IA, onde a textura é artificialmente uniforme.
    Retângulos roxos destacam as áreas com maior probabilidade de manipulação.
    """)

# Contato e informações
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido para:** Mirror Glass
**Projeto:** Detecção de Fraudes em Imagens Automotivas
**Versão:** 1.2.0 (Junho/2025)
**Método Duplicidade:** SIFT + SSIM
""")

# Adicionar explicação sobre JSON resumido na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Sobre o JSON Resumido")
st.sidebar.write("""
O JSON resumido contém apenas as informações essenciais:
- **Estatísticas gerais** de duplicatas e manipulações
- **Scores e categorias** por imagem
- **Métricas consolidadas** sem dados técnicos extensos
- **Formato otimizado** para integração com outros sistemas
""")
