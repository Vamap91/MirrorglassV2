# texture_analyzer.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import base64

class TextureAnalyzer:
    """
    Classe para análise de texturas usando Local Binary Pattern (LBP).
    Detecta manipulações em imagens automotivas, principalmente restaurações por IA.
    """
    
    def __init__(self, P=8, R=1, block_size=16, threshold=0.3):
        """
        Inicializa o analisador de textura.
        
        Args:
            P: Número de pontos vizinhos para o LBP
            R: Raio para o LBP
            block_size: Tamanho dos blocos para análise local (pixels)
            threshold: Limiar para considerar baixa variância (0-1)
        """
        self.P = P  # Número de pontos vizinhos
        self.R = R  # Raio
        self.block_size = block_size  # Tamanho dos blocos para análise
        self.threshold = threshold  # Limiar para textura suspeita
    
    def calculate_lbp(self, image):
        """
        Calcula o padrão binário local (LBP) da imagem.
        
        Args:
            image: Imagem (array numpy ou imagem PIL)
            
        Returns:
            Imagem LBP e histograma de padrões
        """
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
        
        return lbp, hist
    
    def analyze_texture_variance(self, image):
        """
        Analisa a variância local das texturas e identifica áreas com baixa entropia.
        
        Args:
            image: Imagem (PIL ou array numpy)
            
        Returns:
            Dict contendo:
                - variance_map: Mapa de variância normalizado
                - suspicious_mask: Máscara de áreas suspeitas
                - naturalness_score: Score de naturalidade (0-100)
                - heatmap: Mapa de calor para visualização
        """
        # Converter para formato numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Cálculo do LBP
        lbp_image, _ = self.calculate_lbp(image)
        
        # Inicializa matriz de variância e entropia
        height, width = lbp_image.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        
        # Analisa blocos da imagem
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = lbp_image[i:i+self.block_size, j:j+self.block_size]
                
                # Calcula a entropia do bloco (medida de aleatoriedade)
                hist, _ = np.histogram(block, bins=10, range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                
                # Normaliza a entropia para um intervalo de 0 a 1
                max_entropy = np.log(10)  # Entropia máxima para 10 bins
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                
                # Calcula variância normalizada
                block_variance = np.var(block) / 255.0
                
                # Armazena nos mapas
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
        
        # Combina entropia e variância para pontuação de naturalidade (70% entropia, 30% variância)
        naturalness_map = entropy_map * 0.7 + variance_map * 0.3
        
        # Normaliza o mapa para visualização
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Cria máscara de áreas suspeitas (baixa naturalidade)
        suspicious_mask = norm_naturalness_map < self.threshold
        
        # Calcula score de naturalidade (0-100)
        naturalness_score = int(np.mean(norm_naturalness_map) * 100)
        
        # Converte para mapa de calor para visualização
        heatmap = cv2.applyColorMap((norm_naturalness_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "variance_map": variance_map,
            "naturalness_map": norm_naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap
        }
    
    def classify_naturalness(self, score):
        """
        Classifica o score de naturalidade em categorias.
        
        Args:
            score: Score de naturalidade (0-100)
            
        Returns:
            Categoria e descrição
        """
        if score <= 30:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, analysis_results):
        """
        Gera relatório visual com áreas suspeitas destacadas.
        
        Args:
            image: Imagem original (PIL ou numpy)
            analysis_results: Resultados da análise de textura
            
        Returns:
            Imagem com áreas suspeitas destacadas e mapa de calor
        """
        # Converter para numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Extrair resultados
        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]
        
        # Redimensionar para o tamanho da imagem original
        height, width = image.shape[:2]
        mask_height, mask_width = suspicious_mask.shape
        
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
        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
        
        # Classificar resultado
        category, description = self.classify_naturalness(score)
        
        # Adicionar informações na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        return highlighted, heatmap
    
    def analyze_image(self, image):
        """
        Função completa para analisar a textura de uma imagem.
        
        Args:
            image: Imagem a ser analisada (PIL ou numpy)
            
        Returns:
            Dict com resultados da análise e imagens de visualização
        """
        # Analisar textura
        analysis_results = self.analyze_texture_variance(image)
        
        # Gerar visualização
        visual_report, heatmap = self.generate_visual_report(image, analysis_results)
        
        # Classificar o resultado
        score = analysis_results["naturalness_score"]
        category, description = self.classify_naturalness(score)
        
        # Calcular percentual de áreas suspeitas
        percent_suspicious = float(np.mean(analysis_results["suspicious_mask"]) * 100)
        
        # Criar relatório final
        report = {
            "score": score,
            "category": category,
            "description": description,
            "percent_suspicious": percent_suspicious,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "analysis_results": analysis_results
        }
        
        return report

# Função auxiliar para integrar com o Mirror Glass
def get_image_download_link(img, filename, text):
    """
    Gera link para download de imagem.
    
    Args:
        img: Imagem como array numpy
        filename: Nome do arquivo para download
        text: Texto do link
        
    Returns:
        HTML para link de download
    """
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
