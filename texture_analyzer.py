# texture_analyzer.py
# Sistema de Análise Sequencial com Validação em Cadeia
# Versão: 4.0.0 - Janeiro 2025 (Sequential Validation Logic)

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
        
        naturalness_map = (entropy_map * 0.60 + 
                          variance_map * 0.20 + 
                          uniformity_map * 0.20)
        
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        suspicious_mask = norm_naturalness_map < self.threshold
        
        mean_naturalness = np.mean(norm_naturalness_map)
        suspicious_ratio = np.mean(suspicious_mask)
        
        # Penalização mais agressiva
        if suspicious_ratio > 0.10:  # Reduzido de 0.20 para 0.10
            penalty_factor = 1.0 - (suspicious_ratio * 0.8)  # Aumentado de 0.5 para 0.8
            naturalness_score = int(mean_naturalness * penalty_factor * 100)
        else:
            naturalness_score = int(mean_naturalness * 100)
        
        naturalness_score = max(0, min(100, naturalness_score))
        
        heatmap = cv2.applyColorMap((norm_naturalness_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "variance_map": variance_map,
            "naturalness_map": norm_naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio
        }
    
    def classify_naturalness(self, score):
        if score <= 35:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 55:
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
        
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), 
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
        cv2.putText(highlighted, "CLAHE: OFF (Pure Texture)", (10, 90), font, 0.5, (255, 0, 0), 1)
        
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
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return {
            "magnitude": magnitude,
            "magnitude_normalized": magnitude_normalized,
            "direction": direction,
            "gradient_x": gradient_x,
            "gradient_y": gradient_y
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
            "coherence_map": coherence_normalized,
            "edge_density_map": edge_density_normalized,
            "edge_naturalness_map": edge_naturalness,
            "edge_score": edge_score
        }
    
    def analyze_image(self, image):
        coherence_results = self.analyze_edge_coherence(image)
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
            "clahe_enabled": self.use_clahe
        }


class NoiseAnalyzer:
    """Análise de ruído COM CLAHE - detecta inconsistências."""
    
    def __init__(self, block_size=32, sigma_threshold=0.15,
                 use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = block_size
        self.sigma_threshold = sigma_threshold
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
    
    def estimate_noise_level(self, image):
        gray = self._convert_to_gray(image)
        sigma = estimate_sigma(gray, average_sigmas=True, channel_axis=None)
        return sigma
    
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
        
        noise_map_normalized = cv2.normalize(noise_map, None, 0, 1, cv2.NORM_MINMAX)
        
        return {
            "noise_map": noise_map,
            "noise_map_normalized": noise_map_normalized
        }
    
    def detect_noise_inconsistencies(self, image):
        global_noise = self.estimate_noise_level(image)
        local_analysis = self.analyze_local_noise(image)
        noise_map = local_analysis["noise_map"]
        
        noise_mean = np.mean(noise_map)
        noise_std = np.std(noise_map)
        
        if noise_mean > 0:
            noise_cv = noise_std / noise_mean
        else:
            noise_cv = 0
        
        noise_deviation = np.abs(noise_map - noise_mean) / (noise_std + 1e-7)
        suspicious_noise_mask = noise_deviation > 2.0
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
    
    def analyze_image(self, image):
        inconsistency_results = self.detect_noise_inconsistencies(image)
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
            "clahe_enabled": self.use_clahe
        }


class LightingAnalyzer:
    """Analisador de iluminação COM CLAHE - valida física da luz."""
    
    def __init__(self, reflection_weight=0.30, gradient_weight=0.30, 
                 shadow_weight=0.20, global_weight=0.20,
                 use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.reflection_weight = reflection_weight
        self.gradient_weight = gradient_weight
        self.shadow_weight = shadow_weight
        self.global_weight = global_weight
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
    
    def analyze_specular_reflections(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            value_channel = self.apply_clahe(image)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value_channel = hsv[:, :, 2]
            value_channel = self.apply_clahe(value_channel)
        
        _, highlights = cv2.threshold(value_channel, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(highlights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        natural_reflections = 0
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if 0.4 < circularity < 1.0:
                        natural_reflections += 1
                        total_area += area
        
        reflection_score = min(natural_reflections * 5, 30)
        image_area = value_channel.shape[0] * value_channel.shape[1]
        reflection_ratio = (total_area / image_area) * 100
        
        return {
            "num_highlights": len(contours),
            "natural_reflections": natural_reflections,
            "reflection_ratio": reflection_ratio,
            "score_adjustment": reflection_score
        }
    
    def analyze_lighting_gradients(self, image):
        gray = self._convert_to_gray(image)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        smoothness = 1.0 / (np.std(laplacian) + 1)
        
        direction_hist, _ = np.histogram(direction, bins=36, range=(-np.pi, np.pi))
        if np.sum(direction_hist) > 0:
            direction_hist = direction_hist / np.sum(direction_hist)
            direction_consistency = np.max(direction_hist)
        else:
            direction_consistency = 0
        
        abrupt_transitions = np.sum(magnitude > np.percentile(magnitude, 95))
        abrupt_ratio = abrupt_transitions / magnitude.size
        
        gradient_score = 0
        if smoothness > 0.05:
            gradient_score += 10
        if direction_consistency > 0.15:
            gradient_score += 10
        if abrupt_ratio < 0.05:
            gradient_score += 10
        
        return {
            "smoothness": smoothness,
            "direction_consistency": direction_consistency,
            "abrupt_ratio": abrupt_ratio,
            "score_adjustment": gradient_score
        }
    
    def analyze_image(self, image):
        reflections = self.analyze_specular_reflections(image)
        gradients = self.analyze_lighting_gradients(image)
        
        lighting_score = (
            reflections["score_adjustment"] * self.reflection_weight +
            gradients["score_adjustment"] * self.gradient_weight +
            15 * self.shadow_weight +  # Score médio para sombras
            20 * self.global_weight     # Score médio para consistência global
        )
        
        lighting_score = int(min(max(lighting_score, 0), 100))
        
        if lighting_score >= 20:
            category = "Iluminação natural"
            description = "Física da luz consistente"
        elif lighting_score >= 10:
            category = "Iluminação aceitável"
            description = "Algumas inconsistências menores"
        else:
            category = "Iluminação suspeita"
            description = "Inconsistências físicas detectadas"
        
        return {
            "lighting_score": lighting_score,
            "category": category,
            "description": description,
            "clahe_enabled": self.use_clahe
        }


class SequentialAnalyzer:
    """Sistema de Análise Sequencial - Validação em Cadeia"""
    
    def __init__(self):
        self.texture_analyzer = TextureAnalyzer()  # SEM CLAHE
        self.edge_analyzer = EdgeAnalyzer(use_clahe=True)
        self.noise_analyzer = NoiseAnalyzer(use_clahe=True)
        self.lighting_analyzer = LightingAnalyzer(use_clahe=True)
    
    def analyze_sequential(self, image):
        """
        Análise sequencial com validação em cadeia.
        Retorna assim que houver certeza suficiente.
        """
        validation_chain = []
        all_scores = {}
        
        # ========================================
        # FASE 1: DETECTOR PRIMÁRIO (Textura SEM CLAHE)
        # ========================================
        texture_result = self.texture_analyzer.analyze_image(image)
        texture_score = texture_result['score']
        all_scores['texture'] = texture_score
        validation_chain.append('texture')
        
        # CASO 1: Certeza de manipulação
        if texture_score < 35:
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial detectada (LBP puro)",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Score de textura muito baixo ({texture_score}/100). Padrão típico de IA generativa."
            }
        
        # CASO 2: Certeza de natural
        if texture_score > 70:
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
                "detailed_reason": f"Score de textura alto ({texture_score}/100). Textura com variabilidade natural."
            }
        
        # CASO 3: ZONA CINZA (35-70) → Ir para FASE 2
        
        # ========================================
        # FASE 2: VALIDADOR DE BORDAS
        # ========================================
        edge_result = self.edge_analyzer.analyze_image(image)
        edge_score = edge_result['edge_score']
        all_scores['edge'] = edge_score
        validation_chain.append('edge')
        
        # CASO 4: Bordas confirmam manipulação
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
        
        # CASO 5: Bordas naturais mas textura ainda duvidosa → Ir para FASE 3
        
        # ========================================
        # FASE 3: VALIDADOR DE RUÍDO
        # ========================================
        noise_result = self.noise_analyzer.analyze_image(image)
        noise_score = noise_result['noise_score']
        all_scores['noise'] = noise_score
        validation_chain.append('noise')
        
        # CASO 6: Ruído confirma manipulação
        if noise_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Múltiplos indicadores artificiais (textura + ruído)",
                "main_score": texture_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 3,
                "visual_report": texture_result['visual_report'],
                "heatmap": texture_result['heatmap'],
                "percent_suspicious": texture_result['percent_suspicious'],
                "detailed_reason": f"Textura suspeita ({texture_score}/100) + ruído artificial ({noise_score}/100)."
            }
        
        # CASO 7: Ainda inconclusivo → Ir para FASE 4
        
        # ========================================
        # FASE 4: VALIDADOR DE FÍSICA (Desempate)
        # ========================================
        lighting_result = self.lighting_analyzer.analyze_image(image)
        lighting_score = lighting_result['lighting_score']
        all_scores['lighting'] = lighting_score
        validation_chain.append('lighting')
        
        # CASO 8: Física impossível
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
                "detailed_reason": f"Iluminação inconsistente ({lighting_score}/100) indica manipulação."
            }
        
        # ========================================
        # CASO 9: TODOS INCONCLUSIVOS - Calcular ponderado
        # ========================================
        weighted_score = (
            texture_score * 0.50 +
            edge_score * 0.25 +
            noise_score * 0.15 +
            lighting_score * 0.10
        )
        
        if weighted_score < 50:
            verdict = "SUSPEITA"
            confidence = 70
            reason = "Múltiplos indicadores ambíguos - provável manipulação"
        else:
            verdict = "INCONCLUSIVA"
            confidence = 60
            reason = "Análise inconclusiva - revisão manual necessária"
        
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
            "detailed_reason": f"Score ponderado: {int(weighted_score)}/100. Todos os testes foram inconclusivos."
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
