# ✅ TextureAnalyzer (LBP) ✅ EdgeAnalyzer (Bordas) ✅ NoiseAnalyzer (Ruído) ✅ LightingAnalyzer (Iluminação)
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
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
    
    def __init__(self, P=8, R=1, block_size=16, threshold=0.50):
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
        naturalness_map = entropy_map * 0.5 + variance_map * 0.5
        
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
        if score <= 45:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 65:
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


class EdgeAnalyzer:
    """
    Classe para análise de bordas e detecção de transições artificiais.
    Complementa o TextureAnalyzer detectando áreas manipuladas através de 
    inconsistências nas bordas e gradientes.
    """
    
    def __init__(self, block_size=16, edge_threshold_low=50, edge_threshold_high=150):
        """
        Inicializa o analisador de bordas.
        
        Args:
            block_size: Tamanho dos blocos para análise local (pixels)
            edge_threshold_low: Limiar inferior para detecção Canny
            edge_threshold_high: Limiar superior para detecção Canny
        """
        self.block_size = block_size
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
    
    def _convert_to_gray(self, image):
        """Converte imagem para escala de cinza."""
        if isinstance(image, Image.Image):
            return np.array(image.convert('L'))
        elif len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image
    
    def detect_edges(self, image):
        """
        Detecta bordas usando algoritmo Canny.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Mapa de bordas binário
        """
        gray = self._convert_to_gray(image)
        
        # Aplicar desfoque gaussiano para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Detecção de bordas com Canny
        edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        
        return edges
    
    def compute_gradients(self, image):
        """
        Calcula magnitude e direção dos gradientes.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com magnitude e direção dos gradientes
        """
        gray = self._convert_to_gray(image)
        
        # Calcular gradientes usando Sobel
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude do gradiente
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Direção do gradiente (em radianos)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # Normalizar magnitude para visualização
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return {
            "magnitude": magnitude,
            "magnitude_normalized": magnitude_normalized,
            "direction": direction,
            "gradient_x": gradient_x,
            "gradient_y": gradient_y
        }
    
    def analyze_edge_coherence(self, image):
        """
        Analisa a coerência espacial das bordas.
        Bordas naturais tendem a ser mais coerentes que bordas artificiais.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com mapa de coerência e score
        """
        # Converter para numpy se necessário
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = self._convert_to_gray(image)
        height, width = gray.shape
        
        # Calcular gradientes
        gradients = self.compute_gradients(image)
        magnitude = gradients["magnitude"]
        direction = gradients["direction"]
        
        # Dividir em blocos
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        coherence_map = np.zeros((rows, cols))
        edge_density_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                # Extrair bloco
                block_mag = magnitude[i:i+self.block_size, j:j+self.block_size]
                block_dir = direction[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx >= rows or col_idx >= cols:
                    continue
                
                # Calcular densidade de bordas (magnitude média)
                edge_density = np.mean(block_mag) / 255.0
                edge_density_map[row_idx, col_idx] = edge_density
                
                # Calcular coerência de direção
                # Bordas coerentes têm direções similares
                if np.sum(block_mag > 10) > 10:  # Se houver bordas significativas
                    # Usar apenas pixels com magnitude significativa
                    significant_pixels = block_mag > np.percentile(block_mag, 70)
                    if np.any(significant_pixels):
                        directions_sig = block_dir[significant_pixels]
                        
                        # Calcular desvio padrão circular das direções
                        # Menor desvio = maior coerência
                        mean_cos = np.mean(np.cos(directions_sig))
                        mean_sin = np.mean(np.sin(directions_sig))
                        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
                        
                        coherence_map[row_idx, col_idx] = 1 - circular_variance
                    else:
                        coherence_map[row_idx, col_idx] = 0.5
                else:
                    coherence_map[row_idx, col_idx] = 0.5  # Neutro para áreas sem bordas
        
        # Normalizar mapas
        coherence_normalized = cv2.normalize(coherence_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_density_normalized = cv2.normalize(edge_density_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Calcular score geral de naturalidade das bordas
        # Combinação: 60% coerência + 40% densidade
        edge_naturalness = coherence_normalized * 0.6 + edge_density_normalized * 0.4
        
        # Score final (0-100)
        edge_score = int(np.mean(edge_naturalness) * 100)
        
        return {
            "coherence_map": coherence_normalized,
            "edge_density_map": edge_density_normalized,
            "edge_naturalness_map": edge_naturalness,
            "edge_score": edge_score
        }
    
    def detect_artificial_transitions(self, image):
        """
        Detecta transições artificiais que podem indicar manipulação.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Máscara de áreas com transições suspeitas
        """
        gray = self._convert_to_gray(image)
        
        # Detectar bordas
        edges = self.detect_edges(image)
        
        # Calcular gradientes
        gradients = self.compute_gradients(image)
        magnitude = gradients["magnitude_normalized"]
        
        # Detectar transições abruptas
        # Usar filtro Laplaciano para detectar mudanças bruscas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)
        
        # Normalizar
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold para detectar transições muito abruptas (suspeitas)
        _, suspicious_transitions = cv2.threshold(laplacian_norm, 180, 255, cv2.THRESH_BINARY)
        
        # Aplicar morfologia para limpar ruído
        kernel = np.ones((3, 3), np.uint8)
        suspicious_transitions = cv2.morphologyEx(suspicious_transitions, cv2.MORPH_CLOSE, kernel)
        suspicious_transitions = cv2.morphologyEx(suspicious_transitions, cv2.MORPH_OPEN, kernel)
        
        return suspicious_transitions
    
    def generate_edge_visualization(self, image, analysis_results):
        """
        Gera visualização das análises de borda.
        
        Args:
            image: Imagem original
            analysis_results: Resultados da análise de bordas
            
        Returns:
            Imagem com visualização das bordas e áreas suspeitas
        """
        # Converter para numpy RGB
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        height, width = image.shape[:2]
        
        # Redimensionar mapas para tamanho original
        edge_naturalness = analysis_results["edge_naturalness_map"]
        edge_naturalness_resized = cv2.resize(edge_naturalness, (width, height), 
                                             interpolation=cv2.INTER_LINEAR)
        
        # Criar mapa de calor
        heatmap = cv2.applyColorMap((edge_naturalness_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Adicionar informações
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = analysis_results["edge_score"]
        cv2.putText(overlay, f"Edge Score: {score}/100", (10, 90), font, 0.7, (255, 255, 255), 2)
        
        return overlay, heatmap
    
    def analyze_image(self, image):
        """
        Análise completa de bordas da imagem.
        
        Args:
            image: Imagem a ser analisada
            
        Returns:
            Dict com todos os resultados da análise de bordas
        """
        # Análise de coerência
        coherence_results = self.analyze_edge_coherence(image)
        
        # Detecção de transições artificiais
        suspicious_transitions = self.detect_artificial_transitions(image)
        
        # Calcular percentual de transições suspeitas
        percent_suspicious_transitions = (np.sum(suspicious_transitions > 0) / 
                                         suspicious_transitions.size * 100)
        
        # Visualização
        visual_report, heatmap = self.generate_edge_visualization(image, coherence_results)
        
        # Score e classificação
        edge_score = coherence_results["edge_score"]
        
        if edge_score <= 40:
            category = "Bordas artificiais detectadas"
            description = "Alta probabilidade de manipulação"
        elif edge_score <= 65:
            category = "Bordas suspeitas"
            description = "Requer verificação manual"
        else:
            category = "Bordas naturais"
            description = "Baixa probabilidade de manipulação"
        
        return {
            "edge_score": edge_score,
            "category": category,
            "description": description,
            "percent_suspicious_transitions": percent_suspicious_transitions,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "coherence_map": coherence_results["coherence_map"],
            "edge_density_map": coherence_results["edge_density_map"],
            "suspicious_transitions": suspicious_transitions
        }


class NoiseAnalyzer:
    """
    Classe para análise de padrões de ruído e detecção de inconsistências.
    IAs generativas alteram ou removem o ruído natural do sensor da câmera,
    deixando padrões inconsistentes detectáveis.
    """
    
    def __init__(self, block_size=32, sigma_threshold=0.15):
        """
        Inicializa o analisador de ruído.
        
        Args:
            block_size: Tamanho dos blocos para análise local (pixels)
            sigma_threshold: Limiar para detectar variação suspeita de ruído (0-1)
        """
        self.block_size = block_size
        self.sigma_threshold = sigma_threshold
    
    def _convert_to_gray(self, image):
        """Converte imagem para escala de cinza."""
        if isinstance(image, Image.Image):
            return np.array(image.convert('L'))
        elif len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image
    
    def estimate_noise_level(self, image):
        """
        Estima o nível de ruído da imagem completa.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Sigma estimado (desvio padrão do ruído)
        """
        gray = self._convert_to_gray(image)
        
        # Usar estimate_sigma do scikit-image
        # average_sigmas=True para obter média dos canais
        sigma = estimate_sigma(gray, average_sigmas=True, channel_axis=None)
        
        return sigma
    
    def analyze_local_noise(self, image):
        """
        Analisa o ruído localmente por blocos.
        Áreas manipuladas terão ruído diferente das áreas originais.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com mapa de ruído local e estatísticas
        """
        # Converter para numpy se necessário
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = self._convert_to_gray(image)
        height, width = gray.shape
        
        # Dividir em blocos
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        noise_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = gray[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx >= rows or col_idx >= cols:
                    continue
                
                # Estimar ruído do bloco
                try:
                    block_sigma = estimate_sigma(block, average_sigmas=True, channel_axis=None)
                    noise_map[row_idx, col_idx] = block_sigma
                except:
                    # Em caso de erro, usar desvio padrão simples
                    noise_map[row_idx, col_idx] = np.std(block)
        
        # Normalizar mapa
        noise_map_normalized = cv2.normalize(noise_map, None, 0, 1, cv2.NORM_MINMAX)
        
        return {
            "noise_map": noise_map,
            "noise_map_normalized": noise_map_normalized
        }
    
    def detect_noise_inconsistencies(self, image):
        """
        Detecta inconsistências no padrão de ruído.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com análise de inconsistências
        """
        # Estimar ruído global
        global_noise = self.estimate_noise_level(image)
        
        # Analisar ruído local
        local_analysis = self.analyze_local_noise(image)
        noise_map = local_analysis["noise_map"]
        
        # Calcular média e desvio padrão do ruído local
        noise_mean = np.mean(noise_map)
        noise_std = np.std(noise_map)
        
        # Calcular coeficiente de variação (CV)
        # CV alto = inconsistência suspeita
        if noise_mean > 0:
            noise_cv = noise_std / noise_mean
        else:
            noise_cv = 0
        
        # Detectar blocos com ruído muito diferente da média
        # Blocos "limpos demais" ou "ruidosos demais"
        noise_deviation = np.abs(noise_map - noise_mean) / (noise_std + 1e-7)
        
        # Máscara de áreas suspeitas (desvio > 2 sigmas)
        suspicious_noise_mask = noise_deviation > 2.0
        
        # Calcular score de consistência de ruído (0-100)
        # Menor CV = mais consistente = mais natural
        # Normalizar CV para score
        noise_consistency_score = int(max(0, min(100, 100 - (noise_cv * 200))))
        
        return {
            "global_noise": global_noise,
            "noise_mean": noise_mean,
            "noise_std": noise_std,
            "noise_cv": noise_cv,
            "noise_deviation_map": noise_deviation,
            "suspicious_noise_mask": suspicious_noise_mask,
            "noise_consistency_score": noise_consistency_score
        }
    
    def analyze_high_frequency_noise(self, image):
        """
        Analisa componentes de alta frequência (ruído fino).
        IAs tendem a suavizar ou criar ruído artificial.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com análise de alta frequência
        """
        gray = self._convert_to_gray(image)
        
        # Aplicar filtro passa-alta para isolar ruído
        # Usar Laplaciano como filtro passa-alta
        laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F, ksize=3)
        
        # Calcular energia de alta frequência por bloco
        height, width = gray.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        hf_energy_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block_hf = laplacian[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx >= rows or col_idx >= cols:
                    continue
                
                # Energia = soma dos quadrados
                hf_energy = np.sum(block_hf ** 2)
                hf_energy_map[row_idx, col_idx] = hf_energy
        
        # Normalizar
        hf_energy_normalized = cv2.normalize(hf_energy_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Calcular uniformidade da energia
        hf_uniformity = 1.0 - np.std(hf_energy_normalized)
        
        return {
            "hf_energy_map": hf_energy_map,
            "hf_energy_normalized": hf_energy_normalized,
            "hf_uniformity": hf_uniformity
        }
    
    def generate_noise_visualization(self, image, analysis_results):
        """
        Gera visualização da análise de ruído.
        
        Args:
            image: Imagem original
            analysis_results: Resultados da análise de ruído
            
        Returns:
            Imagem com visualização do ruído
        """
        # Converter para numpy RGB
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        height, width = image.shape[:2]
        
        # Redimensionar mapa de ruído
        noise_map_normalized = analysis_results["noise_map_normalized"]
        noise_map_resized = cv2.resize(noise_map_normalized, (width, height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Criar mapa de calor
        heatmap = cv2.applyColorMap((noise_map_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Adicionar informações
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = analysis_results["noise_consistency_score"]
        cv2.putText(overlay, f"Noise Score: {score}/100", (10, 120), font, 0.7, (255, 255, 255), 2)
        
        return overlay, heatmap
    
    def analyze_image(self, image):
        """
        Análise completa de ruído da imagem.
        
        Args:
            image: Imagem a ser analisada
            
        Returns:
            Dict com todos os resultados da análise de ruído
        """
        # Análise de ruído local
        local_analysis = self.analyze_local_noise(image)
        
        # Detecção de inconsistências
        inconsistency_results = self.detect_noise_inconsistencies(image)
        
        # Análise de alta frequência
        hf_analysis = self.analyze_high_frequency_noise(image)
        
        # Combinar resultados
        noise_map_normalized = local_analysis["noise_map_normalized"]
        
        # Calcular percentual de áreas suspeitas
        suspicious_mask = inconsistency_results["suspicious_noise_mask"]
        percent_suspicious_noise = float(np.mean(suspicious_mask) * 100)
        
        # Preparar para visualização
        visualization_data = {
            "noise_map_normalized": noise_map_normalized,
            "noise_consistency_score": inconsistency_results["noise_consistency_score"]
        }
        
        # Visualização
        visual_report, heatmap = self.generate_noise_visualization(image, visualization_data)
        
        # Score e classificação
        noise_score = inconsistency_results["noise_consistency_score"]
        
        if noise_score <= 40:
            category = "Padrão de ruído artificial"
            description = "Alta probabilidade de manipulação"
        elif noise_score <= 65:
            category = "Ruído inconsistente"
            description = "Requer verificação manual"
        else:
            category = "Ruído natural e consistente"
            description = "Baixa probabilidade de manipulação"
        
        return {
            "noise_score": noise_score,
            "category": category,
            "description": description,
            "percent_suspicious_noise": percent_suspicious_noise,
            "global_noise": inconsistency_results["global_noise"],
            "noise_cv": inconsistency_results["noise_cv"],
            "visual_report": visual_report,
            "heatmap": heatmap,
            "noise_map": local_analysis["noise_map"],
            "suspicious_noise_mask": suspicious_mask,
            "hf_energy_map": hf_analysis["hf_energy_map"]
        }


class LightingAnalyzer:
    """
    Analisador de iluminação baseado em princípios físicos.
    Detecta manipulações por IA através de inconsistências na física da luz:
    reflexos, gradientes, sombras e consistência global.
    """
    
    def __init__(self, reflection_weight=0.30, gradient_weight=0.30, 
                 shadow_weight=0.20, global_weight=0.20):
        """
        Inicializa o analisador de iluminação.
        
        Args:
            reflection_weight: Peso para análise de reflexos (0-1)
            gradient_weight: Peso para análise de gradientes (0-1)
            shadow_weight: Peso para análise de sombras (0-1)
            global_weight: Peso para análise de consistência global (0-1)
        """
        self.reflection_weight = reflection_weight
        self.gradient_weight = gradient_weight
        self.shadow_weight = shadow_weight
        self.global_weight = global_weight
    
    def _convert_to_gray(self, image):
        """Converte imagem para escala de cinza."""
        if isinstance(image, Image.Image):
            return np.array(image.convert('L'))
        elif len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image
    
    def analyze_specular_reflections(self, image):
        """
        Analisa reflexos especulares (highlights).
        Reflexos naturais têm forma e distribuição específicas.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com análise de reflexos e score
        """
        # Converter para numpy se necessário
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Converter para HSV (melhor para análise de brilho)
        if len(image.shape) == 2:
            value_channel = image
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value_channel = hsv[:, :, 2]
        
        # Detectar highlights (pixels muito brilhantes)
        _, highlights = cv2.threshold(value_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos dos reflexos
        contours, _ = cv2.findContours(highlights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analisar forma dos reflexos
        natural_reflections = 0
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Ignorar reflexos muito pequenos (ruído) ou muito grandes
            if 50 < area < 5000:
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Calcular circularidade (0 = linha, 1 = círculo perfeito)
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Reflexos naturais tendem a ser elípticos/circulares
                    if 0.4 < circularity < 1.0:
                        natural_reflections += 1
                        total_area += area
        
        # Score baseado em número e qualidade dos reflexos
        reflection_score = min(natural_reflections * 5, 30)
        
        # Calcular percentual de área com reflexos
        image_area = value_channel.shape[0] * value_channel.shape[1]
        reflection_ratio = (total_area / image_area) * 100
        
        return {
            "num_highlights": len(contours),
            "natural_reflections": natural_reflections,
            "reflection_ratio": reflection_ratio,
            "score_adjustment": reflection_score
        }
    
    def analyze_lighting_gradients(self, image):
        """
        Analisa gradientes de iluminação.
        Iluminação natural tem gradientes suaves e direcionais.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com análise de gradientes e score
        """
        gray = self._convert_to_gray(image)
        
        # Calcular gradientes usando Sobel (kernel maior para captar iluminação)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # Analisar suavidade dos gradientes
        # Segunda derivada (Laplaciano) para detectar mudanças bruscas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        smoothness = 1.0 / (np.std(laplacian) + 1)
        
        # Detectar direção dominante da luz
        # Histograma circular de direções (36 bins = 10 graus cada)
        direction_hist, _ = np.histogram(direction, bins=36, range=(-np.pi, np.pi))
        
        # Normalizar histograma
        if np.sum(direction_hist) > 0:
            direction_hist = direction_hist / np.sum(direction_hist)
            direction_consistency = np.max(direction_hist)
        else:
            direction_consistency = 0
        
        # Validar transições
        # Transições muito abruptas são suspeitas
        abrupt_transitions = np.sum(magnitude > np.percentile(magnitude, 95))
        abrupt_ratio = abrupt_transitions / magnitude.size
        
        # Score combinado
        gradient_score = 0
        
        if smoothness > 0.05:  # Gradientes suaves
            gradient_score += 10
        
        if direction_consistency > 0.15:  # Direção consistente
            gradient_score += 10
        
        if abrupt_ratio < 0.05:  # Poucas transições abruptas
            gradient_score += 10
        
        return {
            "smoothness": smoothness,
            "direction_consistency": direction_consistency,
            "abrupt_ratio": abrupt_ratio,
            "score_adjustment": gradient_score
        }
    
    def analyze_shadows(self, image):
        """
        Analisa sombras para validar iluminação natural.
        Sombras devem ter direção e intensidade consistentes.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com análise de sombras e score
        """
        # Converter para numpy se necessário
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Converter para LAB (melhor para luminância)
        if len(image.shape) == 2:
            l_channel = image
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
        
        # Detectar áreas escuras (possíveis sombras)
        _, shadow_mask = cv2.threshold(l_channel, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Analisar conectividade das sombras
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_mask)
        
        # Validar sombras
        valid_shadows = 0
        shadow_directions = []
        
        for i in range(1, num_labels):  # Ignorar background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filtrar sombras por tamanho (muito pequenas = ruído, muito grandes = objeto escuro)
            if 100 < area < 50000:
                valid_shadows += 1
                
                # Calcular orientação da sombra
                shadow_region = (labels == i).astype(np.uint8)
                moments = cv2.moments(shadow_region)
                
                # Evitar divisão por zero
                if moments['mu20'] != 0 and moments['mu02'] != 0:
                    # Calcular ângulo principal
                    angle = 0.5 * np.arctan2(2 * moments['mu11'], 
                                            moments['mu20'] - moments['mu02'])
                    shadow_directions.append(angle)
        
        # Verificar consistência de direção das sombras
        if len(shadow_directions) > 1:
            # Calcular variância circular
            mean_cos = np.mean(np.cos(shadow_directions))
            mean_sin = np.mean(np.sin(shadow_directions))
            shadow_consistency = np.sqrt(mean_cos**2 + mean_sin**2)
        else:
            shadow_consistency = 0.5  # Neutro se poucas sombras
        
        # Score
        shadow_score = 0
        
        if valid_shadows > 0:  # Tem sombras detectáveis
            shadow_score += 10
        
        if shadow_consistency > 0.6:  # Direção consistente
            shadow_score += 15
        
        return {
            "num_shadows": valid_shadows,
            "shadow_consistency": shadow_consistency,
            "score_adjustment": shadow_score
        }
    
    def analyze_global_consistency(self, image):
        """
        Analisa consistência global da iluminação.
        Verifica se iluminação é uniforme em toda a imagem.
        
        Args:
            image: Imagem (PIL ou numpy)
            
        Returns:
            Dict com análise de consistência global e score
        """
        # Converter para numpy se necessário
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Converter para LAB
        if len(image.shape) == 2:
            l_channel = image
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
        
        # Dividir imagem em quadrantes
        h, w = l_channel.shape
        quadrants = [
            l_channel[0:h//2, 0:w//2],      # Superior esquerdo
            l_channel[0:h//2, w//2:w],      # Superior direito
            l_channel[h//2:h, 0:w//2],      # Inferior esquerdo
            l_channel[h//2:h, w//2:w]       # Inferior direito
        ]
        
        # Calcular luminância média de cada quadrante
        quad_means = [np.mean(q) for q in quadrants]
        
        # Verificar consistência entre quadrantes
        mean_luminance = np.mean(quad_means)
        
        if mean_luminance > 0:
            max_deviation = np.max(np.abs(quad_means - mean_luminance))
            consistency_ratio = 1.0 - (max_deviation / mean_luminance)
        else:
            consistency_ratio = 0.5
        
        # Analisar histograma de luminância
        hist, _ = np.histogram(l_channel, bins=64, range=(0, 255))
        hist = hist.astype(float) / (hist.sum() + 1e-7)
        
        # Entropia do histograma (distribuição de intensidades)
        hist_entropy = entropy(hist + 1e-7)  # Evitar log(0)
        
        # Suavidade do histograma
        hist_diff = np.diff(hist)
        hist_smoothness = 1.0 - min(np.std(hist_diff), 1.0)
        
        # Score
        global_score = 0
        
        if consistency_ratio > 0.75:  # Boa consistência entre quadrantes
            global_score += 10
        
        if hist_entropy > 3.5:  # Boa distribuição de intensidades
            global_score += 10
        
        if hist_smoothness > 0.7:  # Histograma suave
            global_score += 10
        
        return {
            "consistency_ratio": consistency_ratio,
            "hist_entropy": hist_entropy,
            "hist_smoothness": hist_smoothness,
            "score_adjustment": global_score
        }
    
    def generate_lighting_visualization(self, image, analysis_results):
        """
        Gera visualização da análise de iluminação.
        
        Args:
            image: Imagem original
            analysis_results: Resultados da análise de iluminação
            
        Returns:
            Imagem com visualização da iluminação
        """
        # Converter para numpy RGB
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Criar visualização simples com overlay de informações
        visual = image.copy()
        
        # Adicionar informações na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = analysis_results["lighting_score"]
        
        cv2.putText(visual, f"Lighting Score: {score}/100", (10, 150), 
                   font, 0.7, (255, 255, 255), 2)
        
        # Criar mapa de calor baseado em luminância
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        return visual, overlay
    
    def analyze_image(self, image):
        """
        Análise completa de iluminação da imagem.
        
        Args:
            image: Imagem a ser analisada
            
        Returns:
            Dict com todos os resultados da análise de iluminação
        """
        # Executar os 4 componentes de análise
        reflections = self.analyze_specular_reflections(image)
        gradients = self.analyze_lighting_gradients(image)
        shadows = self.analyze_shadows(image)
        global_consistency = self.analyze_global_consistency(image)
        
        # Calcular score ponderado
        lighting_score = (
            reflections["score_adjustment"] * self.reflection_weight +
            gradients["score_adjustment"] * self.gradient_weight +
            shadows["score_adjustment"] * self.shadow_weight +
            global_consistency["score_adjustment"] * self.global_weight
        )
        
        # Garantir que score está entre 0-100
        lighting_score = int(min(max(lighting_score, 0), 100))
        
        # Classificação
        if lighting_score >= 20:
            category = "Iluminação natural"
            description = "Física da luz consistente"
        elif lighting_score >= 10:
            category = "Iluminação aceitável"
            description = "Algumas inconsistências menores"
        else:
            category = "Iluminação suspeita"
            description = "Inconsistências físicas detectadas"
        
        # Preparar dados para visualização
        visualization_data = {
            "lighting_score": lighting_score
        }
        
        # Gerar visualização
        visual_report, heatmap = self.generate_lighting_visualization(image, visualization_data)
        
        return {
            "lighting_score": lighting_score,
            "category": category,
            "description": description,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "components": {
                "reflections": reflections,
                "gradients": gradients,
                "shadows": shadows,
                "global": global_consistency
            }
        }


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
