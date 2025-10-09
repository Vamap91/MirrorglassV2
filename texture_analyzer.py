# texture_analyzer.py
# Sistema Unificado de Análise de Imagens com CLAHE
# Versão: 3.0.1 - Janeiro 2025 (FIX: OpenCV type compatibility)

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import io
import base64


class TextureAnalyzer:
    """Análise de texturas usando LBP com suporte a CLAHE."""
    
    def __init__(self, P=8, R=1, block_size=16, threshold=0.50, 
                 use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def apply_clahe(self, img_gray):
        if not self.use_clahe:
            return img_gray
        # Garantir que a imagem seja uint8
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))
        return clahe.apply(img_gray)
    
    def calculate_lbp(self, image):
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()
        
        img_gray = self.apply_clahe(img_gray)
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
                
                # Calcular histograma
                hist, _ = np.histogram(block, bins=10, range=(0, 10))
                hist = hist.astype("float") / (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                
                max_entropy = np.log(10)
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                block_variance = np.var(block) / 255.0
                
                # CRITICAL FIX: Detectar uniformidade artificial (IA)
                # Se um único bin domina o histograma = textura artificial
                max_hist_value = np.max(hist)
                uniformity_penalty = 1.0 - max_hist_value  # 0 = muito uniforme, 1 = variado
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
                    uniformity_map[row_idx, col_idx] = uniformity_penalty
        
        # CRITICAL FIX: Combinar entropia + variância + penalidade de uniformidade
        # Peso maior para entropia (70%) e uniformity (20%)
        naturalness_map = (entropy_map * 0.60 + 
                          variance_map * 0.20 + 
                          uniformity_map * 0.20)
        
        # Normalizar para 0-1
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Áreas suspeitas = baixa naturalness (< threshold)
        suspicious_mask = norm_naturalness_map < self.threshold
        
        # Score = média da naturalness (0-100)
        # AJUSTE: Penalizar fortemente se houver muitas áreas uniformes
        mean_naturalness = np.mean(norm_naturalness_map)
        suspicious_ratio = np.mean(suspicious_mask)
        
        # Se > 20% da imagem é suspeita, reduzir score drasticamente
        if suspicious_ratio > 0.20:
            penalty_factor = 1.0 - (suspicious_ratio * 0.5)  # Até -50%
            naturalness_score = int(mean_naturalness * penalty_factor * 100)
        else:
            naturalness_score = int(mean_naturalness * 100)
        
        # Garantir limites
        naturalness_score = max(0, min(100, naturalness_score))
        
        # Heatmap: Azul (natural) → Vermelho (artificial)
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
        
        if self.use_clahe:
            cv2.putText(highlighted, "CLAHE: ON", (10, 90), font, 0.5, (0, 255, 0), 1)
        
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
            "clahe_enabled": self.use_clahe
        }


class EdgeAnalyzer:
    """Análise de bordas com suporte a CLAHE."""
    
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
    
    def detect_edges(self, image):
        gray = self._convert_to_gray(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        return edges
    
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
    
    def detect_artificial_transitions(self, image):
        gray = self._convert_to_gray(image)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, suspicious_transitions = cv2.threshold(laplacian_norm, 180, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        suspicious_transitions = cv2.morphologyEx(suspicious_transitions, cv2.MORPH_CLOSE, kernel)
        suspicious_transitions = cv2.morphologyEx(suspicious_transitions, cv2.MORPH_OPEN, kernel)
        
        return suspicious_transitions
    
    def generate_edge_visualization(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        height, width = image.shape[:2]
        edge_naturalness = analysis_results["edge_naturalness_map"]
        edge_naturalness_resized = cv2.resize(edge_naturalness, (width, height), 
                                             interpolation=cv2.INTER_LINEAR)
        
        heatmap = cv2.applyColorMap((edge_naturalness_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = analysis_results["edge_score"]
        cv2.putText(overlay, f"Edge Score: {score}/100", (10, 90), font, 0.7, (255, 255, 255), 2)
        
        return overlay, heatmap
    
    def analyze_image(self, image):
        coherence_results = self.analyze_edge_coherence(image)
        suspicious_transitions = self.detect_artificial_transitions(image)
        
        percent_suspicious_transitions = (np.sum(suspicious_transitions > 0) / 
                                         suspicious_transitions.size * 100)
        
        visual_report, heatmap = self.generate_edge_visualization(image, coherence_results)
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
            "suspicious_transitions": suspicious_transitions,
            "clahe_enabled": self.use_clahe
        }


class NoiseAnalyzer:
    """Análise de ruído com suporte a CLAHE."""
    
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
    
    def analyze_high_frequency_noise(self, image):
        # FIX: Garantir que a imagem seja uint8 antes de aplicar Laplacian
        gray = self._convert_to_gray(image)
        
        # Converter para uint8 se necessário
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        # Aplicar Laplacian com tipos compatíveis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        
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
                
                hf_energy = np.sum(block_hf ** 2)
                hf_energy_map[row_idx, col_idx] = hf_energy
        
        hf_energy_normalized = cv2.normalize(hf_energy_map, None, 0, 1, cv2.NORM_MINMAX)
        hf_uniformity = 1.0 - np.std(hf_energy_normalized)
        
        return {
            "hf_energy_map": hf_energy_map,
            "hf_energy_normalized": hf_energy_normalized,
            "hf_uniformity": hf_uniformity
        }
    
    def generate_noise_visualization(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        height, width = image.shape[:2]
        noise_map_normalized = analysis_results["noise_map_normalized"]
        noise_map_resized = cv2.resize(noise_map_normalized, (width, height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        heatmap = cv2.applyColorMap((noise_map_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = analysis_results["noise_consistency_score"]
        cv2.putText(overlay, f"Noise Score: {score}/100", (10, 120), font, 0.7, (255, 255, 255), 2)
        
        return overlay, heatmap
    
    def analyze_image(self, image):
        local_analysis = self.analyze_local_noise(image)
        inconsistency_results = self.detect_noise_inconsistencies(image)
        hf_analysis = self.analyze_high_frequency_noise(image)
        
        noise_map_normalized = local_analysis["noise_map_normalized"]
        suspicious_mask = inconsistency_results["suspicious_noise_mask"]
        percent_suspicious_noise = float(np.mean(suspicious_mask) * 100)
        
        visualization_data = {
            "noise_map_normalized": noise_map_normalized,
            "noise_consistency_score": inconsistency_results["noise_consistency_score"]
        }
        
        visual_report, heatmap = self.generate_noise_visualization(image, visualization_data)
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
            "hf_energy_map": hf_analysis["hf_energy_map"],
            "clahe_enabled": self.use_clahe
        }


class LightingAnalyzer:
    """Analisador de iluminação com suporte a CLAHE."""
    
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
    
    def analyze_shadows(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            l_channel = self.apply_clahe(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            l_channel = self.apply_clahe(l_channel)
        
        _, shadow_mask = cv2.threshold(l_channel, 60, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow_mask)
        
        valid_shadows = 0
        shadow_directions = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if 100 < area < 50000:
                valid_shadows += 1
                
                shadow_region = (labels == i).astype(np.uint8)
                moments = cv2.moments(shadow_region)
                
                if moments['mu20'] != 0 and moments['mu02'] != 0:
                    angle = 0.5 * np.arctan2(2 * moments['mu11'], 
                                            moments['mu20'] - moments['mu02'])
                    shadow_directions.append(angle)
        
        if len(shadow_directions) > 1:
            mean_cos = np.mean(np.cos(shadow_directions))
            mean_sin = np.mean(np.sin(shadow_directions))
            shadow_consistency = np.sqrt(mean_cos**2 + mean_sin**2)
        else:
            shadow_consistency = 0.5
        
        shadow_score = 0
        if valid_shadows > 0:
            shadow_score += 10
        if shadow_consistency > 0.6:
            shadow_score += 15
        
        return {
            "num_shadows": valid_shadows,
            "shadow_consistency": shadow_consistency,
            "score_adjustment": shadow_score
        }
    
    def analyze_global_consistency(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            l_channel = self.apply_clahe(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            l_channel = self.apply_clahe(l_channel)
        
        h, w = l_channel.shape
        quadrants = [
            l_channel[0:h//2, 0:w//2],
            l_channel[0:h//2, w//2:w],
            l_channel[h//2:h, 0:w//2],
            l_channel[h//2:h, w//2:w]
        ]
        
        quad_means = [np.mean(q) for q in quadrants]
        mean_luminance = np.mean(quad_means)
        
        if mean_luminance > 0:
            max_deviation = np.max(np.abs(quad_means - mean_luminance))
            consistency_ratio = 1.0 - (max_deviation / mean_luminance)
        else:
            consistency_ratio = 0.5
        
        hist, _ = np.histogram(l_channel, bins=64, range=(0, 255))
        hist = hist.astype(float) / (hist.sum() + 1e-7)
        hist_entropy = entropy(hist + 1e-7)
        
        hist_diff = np.diff(hist)
        hist_smoothness = 1.0 - min(np.std(hist_diff), 1.0)
        
        global_score = 0
        if consistency_ratio > 0.75:
            global_score += 10
        if hist_entropy > 3.5:
            global_score += 10
        if hist_smoothness > 0.7:
            global_score += 10
        
        return {
            "consistency_ratio": consistency_ratio,
            "hist_entropy": hist_entropy,
            "hist_smoothness": hist_smoothness,
            "score_adjustment": global_score
        }
    
    def generate_lighting_visualization(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        visual = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = analysis_results["lighting_score"]
        
        cv2.putText(visual, f"Lighting Score: {score}/100", (10, 150), 
                   font, 0.7, (255, 255, 255), 2)
        
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        return visual, overlay
    
    def analyze_image(self, image):
        reflections = self.analyze_specular_reflections(image)
        gradients = self.analyze_lighting_gradients(image)
        shadows = self.analyze_shadows(image)
        global_consistency = self.analyze_global_consistency(image)
        
        lighting_score = (
            reflections["score_adjustment"] * self.reflection_weight +
            gradients["score_adjustment"] * self.gradient_weight +
            shadows["score_adjustment"] * self.shadow_weight +
            global_consistency["score_adjustment"] * self.global_weight
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
        
        visualization_data = {"lighting_score": lighting_score}
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
            },
            "clahe_enabled": self.use_clahe
        }


class UnifiedAnalyzer:
    """Sistema unificado com 3 modos de análise e suporte a CLAHE."""
    
    def __init__(self, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        self.texture_analyzer = TextureAnalyzer(
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size
        )
        self.edge_analyzer = EdgeAnalyzer(
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size
        )
        self.noise_analyzer = NoiseAnalyzer(
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size
        )
        self.lighting_analyzer = LightingAnalyzer(
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size
        )
    
    def update_clahe_settings(self, use_clahe=None, clahe_clip_limit=None, clahe_tile_size=None):
        if use_clahe is not None:
            self.use_clahe = use_clahe
            self.texture_analyzer.use_clahe = use_clahe
            self.edge_analyzer.use_clahe = use_clahe
            self.noise_analyzer.use_clahe = use_clahe
            self.lighting_analyzer.use_clahe = use_clahe
        
        if clahe_clip_limit is not None:
            self.clahe_clip_limit = clahe_clip_limit
            self.texture_analyzer.clahe_clip_limit = clahe_clip_limit
            self.edge_analyzer.clahe_clip_limit = clahe_clip_limit
            self.noise_analyzer.clahe_clip_limit = clahe_clip_limit
            self.lighting_analyzer.clahe_clip_limit = clahe_clip_limit
        
        if clahe_tile_size is not None:
            self.clahe_tile_size = clahe_tile_size
            self.texture_analyzer.clahe_tile_size = clahe_tile_size
            self.edge_analyzer.clahe_tile_size = clahe_tile_size
            self.noise_analyzer.clahe_tile_size = clahe_tile_size
            self.lighting_analyzer.clahe_tile_size = clahe_tile_size
    
    def analyze_texture_only(self, image):
        result = self.texture_analyzer.analyze_image(image)
        
        return {
            "mode": "Texture Only (LBP)" + (" + CLAHE" if self.use_clahe else ""),
            "score": result['score'],
            "category": result['category'],
            "description": result['description'],
            "percent_suspicious": result['percent_suspicious'],
            "visual_report": result['visual_report'],
            "heatmap": result['heatmap'],
            "clahe_enabled": result['clahe_enabled'],
            "detailed_results": {"texture": result}
        }
    
    def analyze_complete_fixed(self, image, weights=None):
        if weights is None:
            weights = {'texture': 0.30, 'edge': 0.25, 'noise': 0.20, 'lighting': 0.25}
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        texture_result = self.texture_analyzer.analyze_image(image)
        edge_result = self.edge_analyzer.analyze_image(image)
        noise_result = self.noise_analyzer.analyze_image(image)
        lighting_result = self.lighting_analyzer.analyze_image(image)
        
        combined_score = (
            texture_result['score'] * weights['texture'] +
            edge_result['edge_score'] * weights['edge'] +
            noise_result['noise_score'] * weights['noise'] +
            lighting_result['lighting_score'] * weights['lighting']
        )
        
        final_score = int(min(max(combined_score, 0), 100))
        
        if final_score <= 35:
            category = "Alta chance de manipulação"
            description = "Múltiplas análises indicam manipulação"
        elif final_score <= 55:
            category = "Textura suspeita"
            description = "Revisão manual sugerida"
        else:
            category = "Textura natural"
            description = "Baixa chance de manipulação"
        
        return {
            "mode": "Complete Analysis (Fixed Weights)" + (" + CLAHE" if self.use_clahe else ""),
            "score": final_score,
            "category": category,
            "description": description,
            "weights_used": weights,
            "clahe_enabled": self.use_clahe,
            "individual_scores": {
                "texture": texture_result['score'],
                "edge": edge_result['edge_score'],
                "noise": noise_result['noise_score'],
                "lighting": lighting_result['lighting_score']
            },
            "visual_report": texture_result['visual_report'],
            "heatmap": texture_result['heatmap'],
            "detailed_results": {
                "texture": texture_result,
                "edge": edge_result,
                "noise": noise_result,
                "lighting": lighting_result
            }
        }
    
    def detect_sky_reflection(self, image, threshold=0.20):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            return {"has_sky_reflection": False, "sky_ratio": 0, "score_adjustment": 0}
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        sky_mask = np.zeros(h.shape, dtype=np.uint8)
        blue_sky = ((h >= 90) & (h <= 130) & (s < 100) & (v > 100))
        gray_sky = ((s < 50) & (v > 120))
        sky_mask = (blue_sky | gray_sky).astype(np.uint8) * 255
        
        sky_ratio = np.sum(sky_mask > 0) / sky_mask.size
        has_sky_reflection = sky_ratio > threshold
        
        if has_sky_reflection:
            score_adjustment = min(int(sky_ratio * 100), 30)
        else:
            score_adjustment = 0
        
        return {
            "has_sky_reflection": has_sky_reflection,
            "sky_ratio": sky_ratio,
            "score_adjustment": score_adjustment
        }
    
    def detect_reflective_surface(self, image, threshold=0.30):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Garantir uint8
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        kernel_size = 15
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        mean_sq = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        
        high_variance = variance > np.percentile(variance, 70)
        low_variance = variance < np.percentile(variance, 30)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        moderate_gradients = (gradient_magnitude > 10) & (gradient_magnitude < 80)
        reflective_areas = (high_variance | (low_variance & moderate_gradients))
        reflective_ratio = np.sum(reflective_areas) / reflective_areas.size
        
        is_reflective = reflective_ratio > threshold
        
        return {
            "is_reflective": is_reflective,
            "reflective_ratio": reflective_ratio,
            "confidence": min(reflective_ratio, 1.0)
        }
    
    def analyze_adaptive(self, image, sky_threshold=0.20, sky_bonus=25,
                        reflective_threshold=0.30, reflective_bonus=15):
        texture_result = self.texture_analyzer.analyze_image(image)
        edge_result = self.edge_analyzer.analyze_image(image)
        noise_result = self.noise_analyzer.analyze_image(image)
        lighting_result = self.lighting_analyzer.analyze_image(image)
        
        sky_detection = self.detect_sky_reflection(image, sky_threshold)
        reflective_detection = self.detect_reflective_surface(image, reflective_threshold)
        
        # CRITICAL FIX: Se texture score é muito baixo (< 40), SEMPRE priorizar texture
        # Isso evita falsos negativos em imagens manipuladas com vidro
        texture_score = texture_result['score']
        
        if texture_score < 40:
            # Imagem suspeita! Priorizar análise de textura mesmo com vidro
            weights = {'texture': 0.50, 'edge': 0.20, 'noise': 0.15, 'lighting': 0.15}
            bonus = 0
            reasoning = f"Textura artificial forte (score={texture_score}) - Prioridade máxima"
            if self.use_clahe:
                reasoning += f" | CLAHE ativo"
            detection_type = "Suspicious Texture Priority"
        
        elif sky_detection["has_sky_reflection"] and texture_score >= 50:
            # Tem céu E textura OK = provavelmente legítimo
            weights = {'texture': 0.25, 'edge': 0.20, 'noise': 0.15, 'lighting': 0.40}
            # Bônus SOMENTE se texture score for razoável (>= 50)
            bonus = min(15, sky_detection["score_adjustment"])
            reasoning = f"Reflexo de céu ({sky_detection['sky_ratio']*100:.1f}%) + textura OK"
            if self.use_clahe:
                reasoning += f" | CLAHE ativo"
            detection_type = "Sky Reflection (Natural)"
        
        elif reflective_detection["is_reflective"] and texture_score >= 50:
            # Tem superfície reflexiva E textura OK
            weights = {'texture': 0.30, 'edge': 0.25, 'noise': 0.20, 'lighting': 0.25}
            bonus = min(10, reflective_bonus)
            reasoning = f"Superfície reflexiva ({reflective_detection['reflective_ratio']*100:.1f}%) + textura OK"
            if self.use_clahe:
                reasoning += f" | CLAHE ativo"
            detection_type = "Reflective Surface (Natural)"
        
        else:
            # Padrão: balanced weights
            weights = {'texture': 0.40, 'edge': 0.25, 'noise': 0.20, 'lighting': 0.15}
            bonus = 0
            reasoning = "Análise padrão (pesos balanceados)"
            if self.use_clahe:
                reasoning += f" | CLAHE ativo"
            detection_type = "Standard"
        
        combined_score = (
            texture_result['score'] * weights['texture'] +
            edge_result['edge_score'] * weights['edge'] +
            noise_result['noise_score'] * weights['noise'] +
            lighting_result['lighting_score'] * weights['lighting'] +
            bonus
        )
        
        final_score = int(min(max(combined_score, 0), 100))
        
        # CRITICAL FIX: Classificação mais rigorosa
        if final_score <= 45:
            category = "Alta chance de manipulação"
            description = "Múltiplas análises indicam manipulação"
        elif final_score <= 65:
            category = "Textura suspeita"
            description = "Revisão manual sugerida"
        else:
            category = "Textura natural"
            description = "Baixa chance de manipulação"
        
        return {
            "mode": "Adaptive Analysis" + (" + CLAHE" if self.use_clahe else ""),
            "score": final_score,
            "category": category,
            "description": description,
            "detection_type": detection_type,
            "reasoning": reasoning,
            "weights_used": weights,
            "bonus_applied": bonus,
            "clahe_enabled": self.use_clahe,
            "clahe_settings": {
                "clip_limit": self.clahe_clip_limit,
                "tile_size": self.clahe_tile_size
            } if self.use_clahe else None,
            "individual_scores": {
                "texture": texture_result['score'],
                "edge": edge_result['edge_score'],
                "noise": noise_result['noise_score'],
                "lighting": lighting_result['lighting_score']
            },
            "detections": {
                "sky": sky_detection,
                "reflective": reflective_detection
            },
            "visual_report": texture_result['visual_report'],
            "heatmap": texture_result['heatmap'],
            "detailed_results": {
                "texture": texture_result,
                "edge": edge_result,
                "noise": noise_result,
                "lighting": lighting_result
            }
        }
    
    def analyze(self, image, mode="adaptive", **kwargs):
        if mode == "texture_only":
            return self.analyze_texture_only(image)
        elif mode == "complete_fixed":
            return self.analyze_complete_fixed(image, weights=kwargs.get('weights'))
        elif mode == "adaptive":
            return self.analyze_adaptive(
                image,
                sky_threshold=kwargs.get('sky_threshold', 0.20),
                sky_bonus=kwargs.get('sky_bonus', 25),
                reflective_threshold=kwargs.get('reflective_threshold', 0.30),
                reflective_bonus=kwargs.get('reflective_bonus', 15)
            )
        else:
            raise ValueError(f"Modo inválido: '{mode}'. Use 'texture_only', 'complete_fixed' ou 'adaptive'")
    
    def generate_combined_visual_report(self, image, analysis_result):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        visual = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        score_text = f"SCORE: {analysis_result['score']}/100"
        cv2.putText(visual, score_text, (10, y_offset), font, 1.0, (255, 255, 255), 3)
        cv2.putText(visual, score_text, (10, y_offset), font, 1.0, (0, 255, 0), 2)
        y_offset += 45
        
        category_text = analysis_result['category']
        if analysis_result['score'] > 55:
            color = (0, 255, 0)
        elif analysis_result['score'] > 35:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        cv2.putText(visual, category_text, (10, y_offset), font, 0.7, (255, 255, 255), 3)
        cv2.putText(visual, category_text, (10, y_offset), font, 0.7, color, 2)
        y_offset += 35
        
        mode_text = f"Modo: {analysis_result['mode']}"
        cv2.putText(visual, mode_text, (10, y_offset), font, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        if analysis_result.get('clahe_enabled', False):
            clahe_text = "CLAHE: ON"
            if 'clahe_settings' in analysis_result and analysis_result['clahe_settings']:
                settings = analysis_result['clahe_settings']
                clahe_text += f" (clip={settings['clip_limit']}, tile={settings['tile_size']})"
            cv2.putText(visual, clahe_text, (10, y_offset), font, 0.5, (0, 255, 0), 2)
            y_offset += 25
        
        if 'reasoning' in analysis_result:
            reasoning = analysis_result['reasoning']
            max_chars = 45
            reasoning_lines = [reasoning[i:i+max_chars] for i in range(0, len(reasoning), max_chars)]
            
            for line in reasoning_lines[:2]:
                cv2.putText(visual, line, (10, y_offset), font, 0.45, (255, 255, 255), 1)
                y_offset += 22
        
        if 'individual_scores' in analysis_result:
            y_offset += 10
            scores = analysis_result['individual_scores']
            cv2.putText(visual, f"T:{scores['texture']} E:{scores['edge']} " +
                              f"N:{scores['noise']} L:{scores['lighting']}", 
                       (10, y_offset), font, 0.5, (200, 200, 200), 1)
            y_offset += 20
        
        if 'bonus_applied' in analysis_result and analysis_result['bonus_applied'] > 0:
            bonus_text = f"Bonus: +{analysis_result['bonus_applied']}"
            cv2.putText(visual, bonus_text, (10, y_offset), font, 0.5, (0, 255, 0), 2)
        
        return visual


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
