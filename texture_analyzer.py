# texture_analyzer.py
# Sistema de Análise Sequencial com Validação em Cadeia
# Versão: 4.1.0 - Janeiro 2025 (Fixed Detection + No Normalization)

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import io
import base64


class TextureAnalyzer:
    """Análise de texturas usando LBP - DETECTOR PRIMÁRIO (SEM CLAHE)."""
    
    def __init__(self, P=8, R=1, block_size=16, threshold=0.50):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
    
    def calculate_lbp(self, image):
        """Calcula LBP SEM CLAHE - textura pura"""
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()
        
        # CRITICAL: SEM CLAHE para detectar uniformidade de IA!
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        
        return lbp, hist
    
    def analyze_texture_variance(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        lbp_image, _ = self.calculate_lbp(image)
        height, width = lbp_image.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        uniformity_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = lbp_image[i:i+self.block_size, j:j+self.block_size]
                
                hist, _ = np.histogram(block, bins=10, range=(0, 10))
                hist = hist.astype("float") / (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                
                max_entropy = np.log(10)
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                block_variance = np.var(block) / 255.0
                
                max_hist_value = np.max(hist)
                uniformity_penalty = 1.0 - max_hist_value
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
                    uniformity_map[row_idx, col_idx] = uniformity_penalty
        
        # CRITICAL FIX: Pesos ajustados para detectar IA
        # Variância e uniformidade são mais importantes que entropia!
        naturalness_map = (entropy_map * 0.30 +  # Reduzido de 0.60
                          variance_map * 0.40 +   # Aumentado de 0.20
                          uniformity_map * 0.30)  # Aumentado de 0.20
        
        # NÃO NORMALIZAR! Manter valores absolutos 0-1
        
        suspicious_mask = naturalness_map < self.threshold
        
        mean_naturalness = np.mean(naturalness_map)
        suspicious_ratio = np.mean(suspicious_mask)
        
        # CRITICAL FIX: Penalização MUITO mais agressiva
        # Qualquer sinal de uniformidade deve reduzir drasticamente o score
        if suspicious_ratio > 0.05:  # Se > 5% da imagem é suspeita
            penalty_factor = 1.0 - (suspicious_ratio * 3.0)  # Aumentado de 1.5 para 3.0!
        else:
            penalty_factor = 1.0 - (suspicious_ratio * 2.0)  # Mesmo pequenas áreas penalizam
        
        penalty_factor = max(0.2, penalty_factor)  # Mínimo 0.2 (era 0.3)
        
        naturalness_score = int(mean_naturalness * penalty_factor * 100)
        naturalness_score = max(0, min(100, naturalness_score))
        
        # Heatmap para visualização (aqui SIM normalizar só para cores)
        norm_for_display = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "variance_map": variance_map,
            "naturalness_map": naturalness_map,  # Valores absolutos!
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio,
            "mean_naturalness_raw": mean_naturalness  # Para debug
        }
    
    def classify_naturalness(self, score):
        # CRITICAL FIX: Thresholds mais rigorosos
        if score <= 50:  # Aumentado de 45 para 50
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 68:  # Aumentado de 65 para 68
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]
        
        height, width = image.shape[:2]
        naturalness_map_resized = cv2.resize(naturalness_map, (width, height), 
                                           interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Para visualização, normalizar
        norm_for_display = cv2.normalize(naturalness_map_resized, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((norm_for_display * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        highlighted = overlay.copy()
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
        
        category, description = self.classify_naturalness(score)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        return highlighted, heatmap
    
    def analyze_image(self, image):
        analysis_results = self.analyze_texture_variance(image)
        visual_report, heatmap = self.generate_visual_report(image, analysis_results)
        
        score = analysis_results["naturalness_score"]
        category, description = self.classify_naturalness(score)
        percent_suspicious = float(np.mean(analysis_results["suspicious_mask"]) * 100)
        
        return {
            "score": score,
            "category": category,
            "description": description,
            "percent_suspicious": percent_suspicious,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "analysis_results": analysis_results,
            "clahe_enabled": False
        }


class EdgeAnalyzer:
    """Análise de bordas COM CLAHE - útil para revelar transições."""
    
    def __init__(self, block_size=16, edge_threshold_low=50, edge_threshold_high=150,
                 use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = block_size
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def _convert_to_gray(self, image):
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)
    
    def compute_gradients(self, image):
        gray = self._convert_to_gray(image)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        return {
            "magnitude": magnitude,
            "direction": direction
        }
    
    def analyze_edge_coherence(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = self._convert_to_gray(image)
        height, width = gray.shape
        
        gradients = self.compute_gradients(image)
        magnitude = gradients["magnitude"]
        direction = gradients["direction"]
        
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        coherence_map = np.zeros((rows, cols))
        edge_density_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block_mag = magnitude[i:i+self.block_size, j:j+self.block_size]
                block_dir = direction[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx >= rows or col_idx >= cols:
                    continue
                
                edge_density = np.mean(block_mag) / 255.0
                edge_density_map[row_idx, col_idx] = edge_density
                
                if np.sum(block_mag > 10) > 10:
                    significant_pixels = block_mag > np.percentile(block_mag, 70)
                    if np.any(significant_pixels):
                        directions_sig = block_dir[significant_pixels]
                        mean_cos = np.mean(np.cos(directions_sig))
                        mean_sin = np.mean(np.sin(directions_sig))
                        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
                        coherence_map[row_idx, col_idx] = 1 - circular_variance
                    else:
                        coherence_map[row_idx, col_idx] = 0.5
                else:
                    coherence_map[row_idx, col_idx] = 0.5
        
        coherence_normalized = cv2.normalize(coherence_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_density_normalized = cv2.normalize(edge_density_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_naturalness = coherence_normalized * 0.6 + edge_density_normalized * 0.4
        edge_score = int(np.mean(edge_naturalness) * 100)
        
        return {
            "edge_score": edge_score
        }
    
    def analyze_image(self, image):
        coherence_results = self.analyze_edge_coherence(image)
        edge_score = coherence_results["edge_score"]
        
        if edge_score <= 40:
            category = "Bordas artificiais"
            description = "Alta probabilidade de manipulação"
        elif edge_score <= 65:
            category = "Bordas suspeitas"
            description = "Requer verificação"
        else:
            category = "Bordas naturais"
            description = "Baixa probabilidade de manipulação"
        
        return {
            "edge_score": edge_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class NoiseAnalyzer:
    """Análise de ruído COM CLAHE."""
    
    def __init__(self, block_size=32, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = block_size
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def _convert_to_gray(self, image):
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)
    
    def analyze_local_noise(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = self._convert_to_gray(image)
        height, width = gray.shape
        
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
                
                try:
                    block_sigma = estimate_sigma(block, average_sigmas=True, channel_axis=None)
                    noise_map[row_idx, col_idx] = block_sigma
                except:
                    noise_map[row_idx, col_idx] = np.std(block)
        
        noise_mean = np.mean(noise_map)
        noise_std = np.std(noise_map)
        
        noise_cv = noise_std / noise_mean if noise_mean > 0 else 0
        noise_consistency_score = int(max(0, min(100, 100 - (noise_cv * 200))))
        
        return noise_consistency_score
    
    def analyze_image(self, image):
        noise_score = self.analyze_local_noise(image)
        
        if noise_score <= 40:
            category = "Ruído artificial"
            description = "Alta probabilidade de manipulação"
        elif noise_score <= 65:
            category = "Ruído inconsistente"
            description = "Requer verificação"
        else:
            category = "Ruído natural"
            description = "Baixa probabilidade de manipulação"
        
        return {
            "noise_score": noise_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class LightingAnalyzer:
    """Analisador de iluminação COM CLAHE."""
    
    def __init__(self, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def _convert_to_gray(self, image):
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)
    
    def analyze_image(self, image):
        gray = self._convert_to_gray(image)
        
        # Análise simplificada de iluminação
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        smoothness = 1.0 / (np.std(magnitude) + 1)
        lighting_score = int(min(smoothness * 50, 30))  # Score simplificado
        
        if lighting_score >= 20:
            category = "Iluminação natural"
            description = "Física consistente"
        elif lighting_score >= 10:
            category = "Iluminação aceitável"
            description = "Pequenas inconsistências"
        else:
            category = "Iluminação suspeita"
            description = "Inconsistências detectadas"
        
        return {
            "lighting_score": lighting_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class SequentialAnalyzer:
    """Sistema de Análise Sequencial - Validação em Cadeia"""
    
    def __init__(self):
        self.texture_analyzer = TextureAnalyzer()
        self.edge_analyzer = EdgeAnalyzer(use_clahe=True)
        self.noise_analyzer = NoiseAnalyzer(use_clahe=True)
        self.lighting_analyzer = LightingAnalyzer(use_clahe=True)
    
    def analyze_sequential(self, image):
        """Análise sequencial com validação em cadeia."""
        validation_chain = []
        all_scores = {}
        
        # ========================================
        # FASE 1: DETECTOR PRIMÁRIO (Textura)
        # ========================================
        texture_result = self.texture_analyzer.analyze_image(image)
        texture_score = texture_result['score']
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        # CRITICAL FIX: Threshold mais rigoroso (50 ao invés de 45)
        if texture_score < 50:
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial detectada",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score {texture_score}/100 indica textura artificial típica de IA."
            }
        
        if texture_score > 75:  # Aumentado de 70 para 75
            return {
                "verdict": "NATURAL",
                "confidence": 85,
                "reason": "Textura natural com alta variabilidade",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score {texture_score}/100 indica textura natural."
            }
        
        # ========================================
        # FASE 2: VALIDADOR DE BORDAS
        # ========================================
        edge_result = self.edge_analyzer.analyze_image(image)
        edge_score = edge_result['edge_score']
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        if edge_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 90,
                "reason": "Textura duvidosa + bordas artificiais",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 2,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Textura suspeita ({texture_score}/100) confirmada por bordas artificiais ({edge_score}/100)."
            }
        
        # ========================================
        # FASE 3: VALIDADOR DE RUÍDO
        # ========================================
        noise_result = self.noise_analyzer.analyze_image(image)
        noise_score = noise_result['noise_score']
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        if noise_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Múltiplos indicadores artificiais",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 3,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Textura suspeita ({texture_score}/100) + ruído artificial ({noise_score}/100)."
            }
        
        # ========================================
        # FASE 4: VALIDADOR DE FÍSICA
        # ========================================
        lighting_result = self.lighting_analyzer.analyze_image(image)
        lighting_score = lighting_result['lighting_score']
        all_scores['lighting'] = lighting_score
        validation_chain.append('lighting')
        
        if lighting_score < 10:
            return {
                "verdict": "MANIPULADA",
                "confidence": 80,
                "reason": "Física da iluminação impossível",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 4,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Iluminação inconsistente ({lighting_score}/100)."
            }
        
        # ========================================
        # CASO FINAL: ANÁLISE INCONCLUSIVA
        # ========================================
        weighted_score = (
            texture_score * 0.50 +
            edge_score * 0.25 +
            noise_score * 0.15 +
            lighting_score * 0.10
        )
        
        if weighted_score < 55:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Múltiplos indicadores ambíguos"
        else:
            verdict = "INCONCLUSIVA"
            confidence = 60
            reason = "Revisão manual necessária"
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": int(weighted_score),
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": 4,
            "visual_report": texture_result['visual_report'],
            "heatmap": texture_result['heatmap'],
            "percent_suspicious": texture_result['percent_suspicious'],
            "detailed_reason": f"Score ponderado: {int(weighted_score)}/100."
        }


def get_image_download_link(img, filename, text):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(img)
    else:
        img_pil = img
    
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href
