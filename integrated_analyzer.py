# integrated_analyzer.py
"""
MirrorGlass V2.1 - Analisador Integrado
Combina:
- V2: Detecção de elementos legítimos (texto/papel/reflexos)
- V1: Análise de textura LBP
- NOVO: Análise de iluminação (sombras/highlights/direção)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import pytesseract
from dataclasses import dataclass, field
from skimage.feature import local_binary_pattern
from scipy.stats import entropy


@dataclass
class RegionInfo:
    """Informações sobre uma região detectada"""
    mask: np.ndarray
    confidence: float
    type: str
    bbox: Tuple[int, int, int, int]
    metadata: Dict = None


@dataclass
class LightingAnalysisResult:
    """Resultados da análise de iluminação"""
    inconsistency_mask: np.ndarray
    inconsistency_score: float  # 0-1 (0 = consistente, 1 = inconsistente)
    shadow_map: np.ndarray
    highlight_map: np.ndarray
    lighting_direction_map: np.ndarray
    suspicious_regions: List[Tuple[int, int, int, int]]
    metadata: Dict
    
    def to_dict(self):
        return {
            'inconsistency_score': float(self.inconsistency_score),
            'suspicious_regions_count': len(self.suspicious_regions),
            'suspicious_regions': self.suspicious_regions,
            'metadata': self.metadata
        }


class LegitimateElementDetector:
    """Detecta elementos legítimos (texto, papel, reflexos)"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.detection_results = {}
        
    def detect_all(self, image: np.ndarray) -> Dict[str, RegionInfo]:
        """Detecta todos os elementos legítimos"""
        if image is None or image.size == 0:
            raise ValueError("Imagem inválida")
            
        results = {}
        
        try:
            text_regions = self._detect_text_regions(image)
            if text_regions:
                results['text'] = text_regions
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de texto: {e}")
            
        try:
            paper_regions = self._detect_paper_regions(image)
            if paper_regions:
                results['paper'] = paper_regions
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de papel: {e}")
            
        try:
            reflection_regions = self._detect_glass_reflections(image)
            if reflection_regions:
                results['reflections'] = reflection_regions
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de reflexos: {e}")
        
        self.detection_results = results
        return results
    
    def _detect_text_regions(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecta regiões com texto"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        text_mask = np.zeros((h, w), dtype=np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        try:
            data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT, config='--psm 11')
            
            valid_text_count = 0
            total_conf = 0
            detected_texts = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:
                    text = str(data['text'][i]).strip()
                    
                    if len(text) >= 2:
                        alnum_count = sum(c.isalnum() for c in text)
                        if alnum_count >= 2:
                            x, y = int(data['left'][i]), int(data['top'][i])
                            w_box, h_box = int(data['width'][i]), int(data['height'][i])
                            
                            if w_box > 0 and h_box > 0:
                                padding = 10
                                x1 = max(0, x - padding)
                                y1 = max(0, y - padding)
                                x2 = min(w, x + w_box + padding)
                                y2 = min(h, y + h_box + padding)
                                
                                text_mask[y1:y2, x1:x2] = 255
                                
                                valid_text_count += 1
                                total_conf += int(conf)
                                detected_texts.append(text)
            
            if valid_text_count > 0:
                avg_conf = total_conf / valid_text_count
                
                kernel = np.ones((15, 15), np.uint8)
                text_mask = cv2.dilate(text_mask, kernel, iterations=1)
                
                contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w_c, h_c = cv2.boundingRect(np.vstack(contours))
                    bbox = (int(x), int(y), int(w_c), int(h_c))
                else:
                    bbox = (0, 0, w, h)
                
                return RegionInfo(
                    mask=text_mask,
                    confidence=min(avg_conf / 100.0, 1.0),
                    type='text',
                    bbox=bbox,
                    metadata={'text_count': valid_text_count, 'texts': detected_texts[:10]}
                )
        except Exception as e:
            if self.debug:
                print(f"Erro OCR: {e}")
        
        return None
    
    def _detect_paper_regions(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecta papéis uniformes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        paper_mask = np.zeros((h, w), dtype=np.uint8)
        
        kernel_size = 15
        gray_float = gray.astype(np.float32)
        
        mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray_float**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        variance = np.maximum(variance, 0)
        std_dev = np.sqrt(variance)
        
        uniform_mask = (std_dev < 20).astype(np.uint8) * 255
        edges = cv2.Canny(gray, 50, 150)
        
        kernel_edge = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel_edge, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_papers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            min_area = 5000
            max_area = h * w * 0.6
            
            if area < min_area or area > max_area:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            if 4 <= len(approx) <= 6:
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                aspect_ratio = float(w_c) / h_c if h_c > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    continue
                
                roi_uniform = uniform_mask[y:y+h_c, x:x+w_c]
                if roi_uniform.size == 0:
                    continue
                    
                uniformity_ratio = np.sum(roi_uniform == 255) / roi_uniform.size
                
                if uniformity_ratio > 0.5:
                    roi_color = gray[y:y+h_c, x:x+w_c]
                    mean_brightness = np.mean(roi_color)
                    
                    if mean_brightness > 100:
                        cv2.drawContours(paper_mask, [contour], -1, 255, -1)
                        valid_papers.append({
                            'bbox': (int(x), int(y), int(w_c), int(h_c)),
                            'uniformity': float(uniformity_ratio),
                            'brightness': float(mean_brightness),
                            'area': float(area)
                        })
        
        if valid_papers:
            avg_uniformity = np.mean([p['uniformity'] for p in valid_papers])
            
            all_bboxes = [p['bbox'] for p in valid_papers]
            x_min = min(b[0] for b in all_bboxes)
            y_min = min(b[1] for b in all_bboxes)
            x_max = max(b[0] + b[2] for b in all_bboxes)
            y_max = max(b[1] + b[3] for b in all_bboxes)
            
            return RegionInfo(
                mask=paper_mask,
                confidence=float(avg_uniformity),
                type='paper',
                bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                metadata={'paper_count': len(valid_papers), 'papers': valid_papers}
            )
        
        return None
    
    def _detect_glass_reflections(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecta reflexos em vidro"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        h_channel, s_channel, v_channel = cv2.split(hsv)
        reflection_mask = np.zeros((h, w), dtype=np.uint8)
        
        low_saturation = s_channel < 50
        high_value = v_channel > 200
        
        potential_reflections = (low_saturation & high_value).astype(np.uint8) * 255
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float64)
        
        grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        smooth_gradients = ((grad_mag > 10) & (grad_mag < 80)).astype(np.uint8) * 255
        
        reflection_mask = cv2.bitwise_and(potential_reflections, smooth_gradients)
        
        kernel = np.ones((5, 5), np.uint8)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        
        reflection_area = np.sum(reflection_mask == 255)
        total_area = h * w
        reflection_ratio = reflection_area / total_area
        
        if reflection_ratio > 0.05:
            contours, _ = cv2.findContours(reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_c, h_c = cv2.boundingRect(np.vstack(contours))
                bbox = (int(x), int(y), int(w_c), int(h_c))
            else:
                bbox = (0, 0, w, h)
            
            return RegionInfo(
                mask=reflection_mask,
                confidence=min(float(reflection_ratio * 5), 1.0),
                type='reflection',
                bbox=bbox,
                metadata={'reflection_percentage': float(reflection_ratio * 100)}
            )
        
        return None
    
    def create_exclusion_mask(self, image_shape: Tuple[int, int], min_confidence: float = 0.3) -> np.ndarray:
        """Cria máscara de exclusão"""
        h, w = image_shape
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        for region_type, region_info in self.detection_results.items():
            if region_info and region_info.confidence >= min_confidence:
                exclusion_mask = cv2.bitwise_or(exclusion_mask, region_info.mask)
        
        return exclusion_mask


class EdgeAnalyzer:
    """
    NOVO - V2.2: Analisa bordas e contornos suspeitos
    Detecta transições artificiais características de IA
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def analyze_edges(self, image: np.ndarray) -> Dict:
        """
        Analisa bordas para detectar manipulações por IA
        
        IAs como Gemini/Midjourney criam:
        - Bordas suaves demais (sem ruído natural)
        - Gradientes perfeitos
        - Transições artificiais
        """
        if image is None or image.size == 0:
            raise ValueError("Imagem inválida")
        
        h, w = image.shape[:2]
        
        if self.debug:
            print("\n--- Análise de Bordas ---")
        
        # 1. Detectar bordas com múltiplos métodos
        canny_edges = self._detect_canny_edges(image)
        smooth_edges = self._detect_smooth_edges(image)
        
        # 2. Analisar qualidade das bordas
        edge_smoothness_score = self._calculate_edge_smoothness(image, canny_edges)
        
        # 3. Detectar transições suspeitas
        suspicious_transitions = self._detect_suspicious_transitions(image)
        
        # 4. Análise de gradientes (bordas perfeitas demais)
        gradient_anomaly_score = self._detect_perfect_gradients(image)
        
        # 5. Detectar áreas com bordas artificiais
        artificial_edge_mask = self._create_artificial_edge_mask(
            smooth_edges, suspicious_transitions
        )
        
        # Score final de artificialidade de bordas (0-1)
        # Quanto maior, mais artificial
        edge_artifact_score = (
            edge_smoothness_score * 0.4 +
            gradient_anomaly_score * 0.4 +
            (np.sum(artificial_edge_mask > 0) / (h * w)) * 0.2
        )
        
        if self.debug:
            print(f"Edge Smoothness: {edge_smoothness_score:.2%}")
            print(f"Gradient Anomaly: {gradient_anomaly_score:.2%}")
            print(f"Edge Artifact Score: {edge_artifact_score:.2%}")
        
        return {
            'edge_artifact_score': float(edge_artifact_score),
            'edge_smoothness_score': float(edge_smoothness_score),
            'gradient_anomaly_score': float(gradient_anomaly_score),
            'artificial_edge_mask': artificial_edge_mask,
            'canny_edges': canny_edges,
            'smooth_edges': smooth_edges
        }
    
    def _detect_canny_edges(self, image: np.ndarray) -> np.ndarray:
        """Detecta bordas usando Canny"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Suavizar levemente para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny com thresholds ajustados
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def _detect_smooth_edges(self, image: np.ndarray) -> np.ndarray:
        """Detecta bordas muito suaves (característica de IA)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Suavizar mais agressivamente
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Detectar bordas no blur
        smooth_edges = cv2.Canny(blurred, 30, 100)
        
        return smooth_edges
    
    def _calculate_edge_smoothness(self, image: np.ndarray, edges: np.ndarray) -> float:
        """
        Calcula quão suaves são as bordas
        
        Bordas naturais têm ruído/irregularidade
        Bordas de IA são suaves demais
        """
        if np.sum(edges) == 0:
            return 0.0
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        smoothness_scores = []
        
        for contour in contours:
            if len(contour) < 20:  # Ignorar contornos muito pequenos
                continue
            
            # Calcular curvatura (segunda derivada)
            # Contornos suaves têm curvatura baixa e constante
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Razão entre contorno aproximado e original
            # Quanto menor, mais suave
            smoothness = len(approx) / len(contour)
            smoothness_scores.append(smoothness)
        
        if smoothness_scores:
            # Média de suavidade (valores baixos = muito suave = suspeito)
            avg_smoothness = np.mean(smoothness_scores)
            
            # Inverter: quanto menor smoothness, maior o score de suspeita
            # Se smoothness < 0.1 = bordas MUITO suaves = score alto
            if avg_smoothness < 0.15:
                return 0.8
            elif avg_smoothness < 0.25:
                return 0.5
            else:
                return 0.2
        
        return 0.0
    
    def _detect_suspicious_transitions(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta transições de cor/brilho suspeitas
        
        IA cria transições lineares perfeitas
        Fotos reais têm transições irregulares
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        suspicious_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calcular gradiente direcional
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalizar
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detectar regiões com gradiente uniforme demais
        # Dividir em blocos e calcular variância do gradiente
        block_size = 32
        
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block_grad = magnitude_norm[i:i+block_size, j:j+block_size]
                
                if block_grad.size == 0:
                    continue
                
                # Calcular variância do gradiente
                grad_var = np.var(block_grad)
                
                # Se gradiente muito uniforme = suspeito
                if grad_var < 50:
                    mean_grad = np.mean(block_grad)
                    
                    # Se tem gradiente mas é uniforme = transição artificial
                    if mean_grad > 20:
                        suspicious_mask[i:i+block_size, j:j+block_size] = 255
        
        return suspicious_mask
    
    def _detect_perfect_gradients(self, image: np.ndarray) -> float:
        """
        Detecta gradientes 'perfeitos demais'
        
        IA cria gradientes matematicamente perfeitos
        Fotos reais têm imperfeições
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calcular Laplaciano (segunda derivada)
        # Gradientes lineares têm Laplaciano próximo de zero
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian_abs = np.abs(laplacian)
        
        # Dividir imagem em regiões
        block_size = 64
        perfect_gradient_ratio = 0
        total_blocks = 0
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_lap = laplacian_abs[i:i+block_size, j:j+block_size]
                
                if block_lap.size == 0:
                    continue
                
                total_blocks += 1
                
                # Calcular quantos pixels têm Laplaciano muito baixo
                # (indicando gradiente linear perfeito)
                low_laplacian_ratio = np.sum(block_lap < 5) / block_lap.size
                
                # Se mais de 60% dos pixels têm gradiente linear = suspeito
                if low_laplacian_ratio > 0.6:
                    # Mas deve ter alguma variação (não ser uniforme)
                    block_gray = gray[i:i+block_size, j:j+block_size]
                    if np.std(block_gray) > 10:
                        perfect_gradient_ratio += 1
        
        if total_blocks > 0:
            return perfect_gradient_ratio / total_blocks
        
        return 0.0
    
    def _create_artificial_edge_mask(
        self, 
        smooth_edges: np.ndarray,
        suspicious_transitions: np.ndarray
    ) -> np.ndarray:
        """Combina detecções para criar máscara de bordas artificiais"""
        
        # Combinar bordas suaves com transições suspeitas
        artificial_mask = cv2.bitwise_or(smooth_edges, suspicious_transitions)
        
        # Dilatar um pouco para marcar região ao redor
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        artificial_mask = cv2.dilate(artificial_mask, kernel, iterations=1)
        
        return artificial_mask


class LightingAnalyzer:
    """
    NOVO - V2.1: Analisa inconsistências de iluminação
    Detecta sombras, highlights e direção da luz
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        
    def analyze_lighting(self, image: np.ndarray) -> LightingAnalysisResult:
        """Análise completa de iluminação"""
        if image is None or image.size == 0:
            raise ValueError("Imagem inválida")
        
        h, w = image.shape[:2]
        
        if self.debug:
            print("\n--- Análise de Iluminação ---")
        
        # 1. Detectar sombras
        shadow_map = self._detect_shadows(image)
        
        # 2. Detectar highlights
        highlight_map = self._detect_highlights(image)
        
        # 3. Calcular direção da luz
        lighting_direction_map = self._calculate_lighting_directions(image)
        
        # 4. Detectar inconsistências
        inconsistency_mask, inconsistency_score, metadata = self._detect_inconsistencies(
            image, shadow_map, highlight_map, lighting_direction_map
        )
        
        # 5. Identificar regiões suspeitas
        suspicious_regions = self._find_suspicious_regions(inconsistency_mask)
        
        if self.debug:
            print(f"Lighting Inconsistency Score: {inconsistency_score:.2%}")
            print(f"Suspicious Regions: {len(suspicious_regions)}")
        
        return LightingAnalysisResult(
            inconsistency_mask=inconsistency_mask,
            inconsistency_score=inconsistency_score,
            shadow_map=shadow_map,
            highlight_map=highlight_map,
            lighting_direction_map=lighting_direction_map,
            suspicious_regions=suspicious_regions,
            metadata=metadata
        )
    
    def _detect_shadows(self, image: np.ndarray) -> np.ndarray:
        """Detecta áreas de sombra"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        h, s, v = cv2.split(hsv)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        low_value = v < 100
        low_lightness = l_channel < 80
        not_black = v > 20
        
        shadow_mask = (low_value & low_lightness & not_black).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)
        
        return shadow_mask
    
    def _detect_highlights(self, image: np.ndarray) -> np.ndarray:
        """Detecta áreas de highlight"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        high_value = v > 200
        low_saturation = s < 50
        
        b, g, r = cv2.split(image)
        rg_diff = np.abs(r.astype(float) - g.astype(float))
        gb_diff = np.abs(g.astype(float) - b.astype(float))
        rb_diff = np.abs(r.astype(float) - b.astype(float))
        
        max_diff = np.maximum(np.maximum(rg_diff, gb_diff), rb_diff)
        near_white = max_diff < 30
        
        highlight_mask = (high_value & low_saturation & near_white).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, kernel)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 0)
        
        return highlight_mask
    
    def _calculate_lighting_directions(self, image: np.ndarray) -> np.ndarray:
        """Calcula direção da luz usando gradientes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        
        grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        angle_degrees = np.degrees(angle) % 360
        
        magnitude_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        angle_weighted = angle_degrees.copy()
        angle_weighted[magnitude_norm < 0.1] = -1
        
        return angle_weighted
    
    def _detect_inconsistencies(
        self, 
        image: np.ndarray,
        shadow_map: np.ndarray,
        highlight_map: np.ndarray,
        lighting_direction_map: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict]:
        """Detecta inconsistências de iluminação"""
        h, w = image.shape[:2]
        inconsistency_mask = np.zeros((h, w), dtype=np.uint8)
        
        metadata = {}
        
        # 1. Highlights órfãos
        highlight_orphan_mask = self._detect_orphan_highlights(image, highlight_map, shadow_map)
        inconsistency_mask = cv2.bitwise_or(inconsistency_mask, highlight_orphan_mask)
        metadata['orphan_highlights_ratio'] = float(np.sum(highlight_orphan_mask > 0) / (h * w))
        
        # 2. Conflitos de direção
        direction_conflict_mask = self._detect_direction_conflicts(lighting_direction_map, block_size=64)
        inconsistency_mask = cv2.bitwise_or(inconsistency_mask, direction_conflict_mask)
        metadata['direction_conflicts_ratio'] = float(np.sum(direction_conflict_mask > 0) / (h * w))
        
        # 3. Transições abruptas
        transition_mask = self._detect_abrupt_transitions(image)
        inconsistency_mask = cv2.bitwise_or(inconsistency_mask, transition_mask)
        metadata['abrupt_transitions_ratio'] = float(np.sum(transition_mask > 0) / (h * w))
        
        # 4. Sombras anômalas
        shadow_anomaly_mask = self._detect_shadow_anomalies(shadow_map)
        inconsistency_mask = cv2.bitwise_or(inconsistency_mask, shadow_anomaly_mask)
        metadata['shadow_anomalies_ratio'] = float(np.sum(shadow_anomaly_mask > 0) / (h * w))
        
        # Score ponderado
        inconsistency_score = (
            metadata['orphan_highlights_ratio'] * 0.3 +
            metadata['direction_conflicts_ratio'] * 0.4 +
            metadata['abrupt_transitions_ratio'] * 0.2 +
            metadata['shadow_anomalies_ratio'] * 0.1
        )
        
        metadata['total_inconsistency_ratio'] = float(np.sum(inconsistency_mask > 0) / (h * w))
        metadata['weighted_score'] = float(inconsistency_score)
        
        return inconsistency_mask, inconsistency_score, metadata
    
    def _detect_orphan_highlights(self, image: np.ndarray, highlight_map: np.ndarray, shadow_map: np.ndarray) -> np.ndarray:
        """Detecta highlights sem sombras correspondentes"""
        h, w = image.shape[:2]
        orphan_mask = np.zeros((h, w), dtype=np.uint8)
        
        highlight_binary = (highlight_map > 50).astype(np.uint8)
        shadow_binary = (shadow_map > 50).astype(np.uint8)
        
        highlight_contours, _ = cv2.findContours(highlight_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in highlight_contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            search_radius = int(max(w_box, h_box) * 2)
            
            x1 = max(0, x - search_radius)
            y1 = max(0, y - search_radius)
            x2 = min(w, x + w_box + search_radius)
            y2 = min(h, y + h_box + search_radius)
            
            search_region = shadow_binary[y1:y2, x1:x2]
            
            if search_region.size > 0:
                shadow_ratio = np.sum(search_region) / search_region.size
                
                if shadow_ratio < 0.05:
                    cv2.drawContours(orphan_mask, [contour], -1, 255, -1)
        
        return orphan_mask
    
    def _detect_direction_conflicts(self, lighting_direction_map: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Detecta direções de luz conflitantes"""
        h, w = lighting_direction_map.shape
        conflict_mask = np.zeros((h, w), dtype=np.uint8)
        
        block_directions = []
        
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = lighting_direction_map[i:i+block_size, j:j+block_size]
                valid_angles = block[block >= 0]
                
                if len(valid_angles) > block_size * block_size * 0.3:
                    mean_angle = self._circular_mean(valid_angles)
                    block_directions.append({
                        'bbox': (j, i, block_size, block_size),
                        'angle': mean_angle,
                        'valid_ratio': len(valid_angles) / block.size
                    })
        
        for i, block1 in enumerate(block_directions):
            for block2 in block_directions[i+1:]:
                if self._are_adjacent(block1['bbox'], block2['bbox']):
                    angle_diff = abs(block1['angle'] - block2['angle'])
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    if angle_diff > 90:
                        x1, y1, w1, h1 = block1['bbox']
                        x2, y2, w2, h2 = block2['bbox']
                        
                        overlap_x1 = max(x1, x2)
                        overlap_y1 = max(y1, y2)
                        overlap_x2 = min(x1 + w1, x2 + w2)
                        overlap_y2 = min(y1 + h1, y2 + h2)
                        
                        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                            conflict_mask[overlap_y1:overlap_y2, overlap_x1:overlap_x2] = 255
        
        return conflict_mask
    
    def _detect_abrupt_transitions(self, image: np.ndarray) -> np.ndarray:
        """Detecta transições abruptas de iluminação"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        laplacian = cv2.Laplacian(l_channel, cv2.CV_64F, ksize=5)
        laplacian_abs = np.abs(laplacian)
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, transition_mask = cv2.threshold(laplacian_norm, 100, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        transition_mask = cv2.morphologyEx(transition_mask, cv2.MORPH_OPEN, kernel)
        
        return transition_mask
    
    def _detect_shadow_anomalies(self, shadow_map: np.ndarray) -> np.ndarray:
        """Detecta sombras anômalas (muito uniformes)"""
        h, w = shadow_map.shape
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)
        
        _, shadow_binary = cv2.threshold(shadow_map, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(shadow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            shadow_region = shadow_map[mask == 255]
            
            if len(shadow_region) > 0:
                std_dev = np.std(shadow_region)
                
                if std_dev < 5:
                    cv2.drawContours(anomaly_mask, [contour], -1, 255, -1)
        
        return anomaly_mask
    
    def _find_suspicious_regions(self, inconsistency_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Identifica bounding boxes das regiões suspeitas"""
        contours, _ = cv2.findContours(inconsistency_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        suspicious_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                x, y, w, h = cv2.boundingRect(contour)
                suspicious_regions.append((int(x), int(y), int(w), int(h)))
        
        return suspicious_regions
    
    def _circular_mean(self, angles: np.ndarray) -> float:
        """Calcula média circular de ângulos"""
        angles_rad = np.radians(angles)
        sin_mean = np.mean(np.sin(angles_rad))
        cos_mean = np.mean(np.cos(angles_rad))
        mean_rad = np.arctan2(sin_mean, cos_mean)
        return float(np.degrees(mean_rad) % 360)
    
    def _are_adjacent(self, bbox1: Tuple, bbox2: Tuple, max_gap: int = 10) -> bool:
        """Verifica se dois bboxes são adjacentes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_adjacent = ((x1 <= x2 <= x1 + w1 + max_gap) or (x2 <= x1 <= x2 + w2 + max_gap))
        y_adjacent = ((y1 <= y2 <= y1 + h1 + max_gap) or (y2 <= y1 <= y2 + h2 + max_gap))
        
        return x_adjacent and y_adjacent


class IntegratedTextureAnalyzer:
    """
    Analisador integrado V2.1
    - V2: Elementos legítimos
    - V1: Textura LBP
    - NOVO: Análise de iluminação
    """
    
    def __init__(self, P=8, R=1, block_size=20, threshold=0.50, 
                 enable_lighting_analysis=True, enable_edge_analysis=True, debug=False):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.enable_lighting_analysis = enable_lighting_analysis
        self.enable_edge_analysis = enable_edge_analysis
        self.debug = debug
        self.legitimate_detector = LegitimateElementDetector(debug=debug)
        self.lighting_analyzer = LightingAnalyzer(debug=debug) if enable_lighting_analysis else None
        self.edge_analyzer = EdgeAnalyzer(debug=debug) if enable_edge_analysis else None
    
    def analyze_image_integrated(self, image):
        """
        Análise integrada completa V2.2:
        1. V2 - Elementos legítimos (texto/papel/reflexo)
        2. V2.1 - Análise de iluminação
        3. V2.2 - NOVO: Análise de bordas/contornos
        4. V1 - Textura LBP
        """
        if self.debug:
            print("\n=== ANÁLISE INTEGRADA V2.2 ===")
        
        # FASE 1: Detectar elementos legítimos (V2)
        if self.debug:
            print("FASE 1: Elementos legítimos (V2)...")
        
        legitimate_elements = self.legitimate_detector.detect_all(image)
        exclusion_mask = self.legitimate_detector.create_exclusion_mask(image.shape[:2], min_confidence=0.3)
        
        h, w = image.shape[:2]
        total_pixels = h * w
        excluded_pixels = np.sum(exclusion_mask == 255)
        exclusion_percentage = (excluded_pixels / total_pixels) * 100
        
        if self.debug:
            print(f"  Elementos: {list(legitimate_elements.keys())}")
            print(f"  Excluído: {exclusion_percentage:.1f}%")
        
        # FASE 2: Análise de iluminação (V2.1)
        lighting_result = None
        lighting_score_penalty = 0
        
        if self.enable_lighting_analysis and self.lighting_analyzer:
            if self.debug:
                print("FASE 2: Análise de iluminação...")
            
            try:
                lighting_result = self.lighting_analyzer.analyze_lighting(image)
                lighting_score_penalty = int(lighting_result.inconsistency_score * 25)  # Até -25 pontos
                
                if self.debug:
                    print(f"  Inconsistência: {lighting_result.inconsistency_score:.2%}")
                    print(f"  Penalidade: -{lighting_score_penalty} pontos")
            except Exception as e:
                if self.debug:
                    print(f"  Erro na análise de iluminação: {e}")
        
        # FASE 3: NOVA - Análise de bordas (V2.2)
        edge_result = None
        edge_score_penalty = 0
        
        if self.enable_edge_analysis and self.edge_analyzer:
            if self.debug:
                print("FASE 3: Análise de bordas/contornos...")
            
            try:
                edge_result = self.edge_analyzer.analyze_edges(image)
                
                # Penalizar MUITO se bordas artificiais
                # Score de artefato de borda alto = penalidade grande
                edge_artifact = edge_result['edge_artifact_score']
                
                if edge_artifact > 0.6:
                    edge_score_penalty = 35  # Penalidade ALTA
                elif edge_artifact > 0.4:
                    edge_score_penalty = 25
                elif edge_artifact > 0.2:
                    edge_score_penalty = 15
                else:
                    edge_score_penalty = 5
                
                if self.debug:
                    print(f"  Artefatos de borda: {edge_artifact:.2%}")
                    print(f"  Penalidade: -{edge_score_penalty} pontos")
            except Exception as e:
                if self.debug:
                    print(f"  Erro na análise de bordas: {e}")
        
        # FASE 4: Análise de textura LBP (V1)
        if self.debug:
            print("FASE 4: Textura LBP (V1)...")
        
        texture_results = self._analyze_texture_with_mask(image, exclusion_mask)
        
        # Combinar score: textura base - penalidades
        base_score = texture_results['naturalness_score']
        final_score = max(0, base_score - lighting_score_penalty - edge_score_penalty)
        
        # Resultados integrados
        integrated_results = {
            'v2_legitimate_elements': legitimate_elements,
            'v2_exclusion_mask': exclusion_mask,
            'v2_exclusion_percentage': exclusion_percentage,
            'v2_1_lighting_analysis': lighting_result.to_dict() if lighting_result else None,
            'v2_1_lighting_score_penalty': lighting_score_penalty,
            'v2_2_edge_analysis': edge_result,
            'v2_2_edge_score_penalty': edge_score_penalty,
            'v1_texture_analysis': texture_results,
            'base_texture_score': base_score,
            'final_score': final_score,
            'final_category': self._classify_naturalness(final_score),
            'suspicious_areas_percentage': texture_results.get('suspicious_percentage', 0)
        }
        
        if self.debug:
            print(f"\n  Score base (textura): {base_score}")
            print(f"  Penalidade (luz): -{lighting_score_penalty}")
            print(f"  Penalidade (bordas): -{edge_score_penalty}")
            print(f"  Score final: {final_score}")
            print(f"  Categoria: {integrated_results['final_category'][0]}")
            print("=== ANÁLISE CONCLUÍDA ===\n")
        
        return integrated_results
    
    def _analyze_texture_with_mask(self, image, exclusion_mask):
        """Análise de textura LBP aplicando máscara de exclusão"""
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        lbp_image = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        height, width = img_gray.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        exclusion_mask_resized = cv2.resize(exclusion_mask, (width, height))
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        exclusion_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block_lbp = lbp_image[i:i+self.block_size, j:j+self.block_size]
                block_mask = exclusion_mask_resized[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                exclusion_ratio = np.sum(block_mask == 255) / (self.block_size * self.block_size)
                
                if exclusion_ratio > 0.5:
                    exclusion_map[row_idx, col_idx] = 1
                    variance_map[row_idx, col_idx] = 1.0
                    entropy_map[row_idx, col_idx] = 1.0
                else:
                    hist, _ = np.histogram(block_lbp, bins=10, range=(0, 10))
                    hist = hist.astype("float")
                    hist /= (hist.sum() + 1e-7)
                    block_entropy = entropy(hist)
                    max_entropy = np.log(10)
                    norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                    
                    block_variance = np.var(block_lbp) / 255.0
                    
                    if row_idx < rows and col_idx < cols:
                        variance_map[row_idx, col_idx] = block_variance
                        entropy_map[row_idx, col_idx] = norm_entropy
        
        naturalness_map = entropy_map * 0.7 + variance_map * 0.3
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        suspicious_mask = (norm_naturalness_map < self.threshold) & (exclusion_map == 0)
        
        valid_blocks = exclusion_map == 0
        if np.sum(valid_blocks) > 0:
            naturalness_score = int(np.mean(norm_naturalness_map[valid_blocks]) * 100)
            suspicious_percentage = float(np.sum(suspicious_mask) / np.sum(valid_blocks) * 100)
        else:
            naturalness_score = 100
            suspicious_percentage = 0.0
        
        heatmap = cv2.applyColorMap((norm_naturalness_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            'naturalness_map': norm_naturalness_map,
            'suspicious_mask': suspicious_mask,
            'exclusion_map': exclusion_map,
            'naturalness_score': naturalness_score,
            'suspicious_percentage': suspicious_percentage,
            'heatmap': heatmap,
            'entropy_map': entropy_map,
            'variance_map': variance_map
        }
    
    def _classify_naturalness(self, score):
        """Classificação do score"""
        if score <= 45:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, integrated_results):
        """Gera relatório visual integrado V2.2"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        height, width = image.shape[:2]
        
        naturalness_map = integrated_results['v1_texture_analysis']['naturalness_map']
        suspicious_mask = integrated_results['v1_texture_analysis']['suspicious_mask']
        exclusion_mask = integrated_results['v2_exclusion_mask']
        score = integrated_results['final_score']
        
        naturalness_map_resized = cv2.resize(naturalness_map, (width, height), interpolation=cv2.INTER_LINEAR)
        suspicious_mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        exclusion_mask_resized = cv2.resize(exclusion_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        highlighted = overlay.copy()
        
        # Áreas suspeitas de textura (roxo)
        contours, _ = cv2.findContours(suspicious_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (128, 0, 128), 2)
        
        # Áreas excluídas (verde)
        contours_excluded, _ = cv2.findContours(exclusion_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_excluded:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Inconsistências de iluminação (laranja)
        lighting_analysis = integrated_results.get('v2_1_lighting_analysis')
        if lighting_analysis and lighting_analysis['suspicious_regions_count'] > 0:
            for x, y, w, h in lighting_analysis['suspicious_regions']:
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 165, 255), 2)
        
        # NOVO: Bordas artificiais (ciano)
        edge_analysis = integrated_results.get('v2_2_edge_analysis')
        if edge_analysis:
            artificial_edge_mask = edge_analysis['artificial_edge_mask']
            if artificial_edge_mask is not None:
                artificial_resized = cv2.resize(artificial_edge_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Destacar áreas com bordas artificiais
                edge_contours, _ = cv2.findContours(artificial_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in edge_contours:
                    area = cv2.contourArea(contour)
                    if area > 200:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(highlighted, (x, y), (x+w, y+h), (255, 255, 0), 2)  # Ciano
        
        category, _ = integrated_results['final_category']
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        cv2.putText(highlighted, f"Score: {score}/100", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(highlighted, category, (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(highlighted, f"Excluido: {integrated_results['v2_exclusion_percentage']:.1f}%", 
                   (10, y_offset), font, 0.6, (0, 255, 0), 2)
        y_offset += 30
        
        # Penalidade de iluminação
        lighting_penalty = integrated_results.get('v2_1_lighting_score_penalty', 0)
        if lighting_penalty > 0:
            cv2.putText(highlighted, f"Luz: -{lighting_penalty}pts", 
                       (10, y_offset), font, 0.6, (0, 165, 255), 2)
            y_offset += 30
        
        # NOVO: Penalidade de bordas
        edge_penalty = integrated_results.get('v2_2_edge_score_penalty', 0)
        if edge_penalty > 0:
            cv2.putText(highlighted, f"Bordas: -{edge_penalty}pts", 
                       (10, y_offset), font, 0.6, (255, 255, 0), 2)
        
        return highlighted, heatmap), (x+w, y+h), (0, 255, 0), 2)
        
        # NOVO: Marcar inconsistências de iluminação (laranja)
        lighting_analysis = integrated_results.get('v2_1_lighting_analysis')
        if lighting_analysis and lighting_analysis['suspicious_regions_count'] > 0:
            for x, y, w, h in lighting_analysis['suspicious_regions']:
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 165, 255), 2)  # Laranja
        
        category, _ = integrated_results['final_category']
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, f"Excluido: {integrated_results['v2_exclusion_percentage']:.1f}%", 
                   (10, 90), font, 0.6, (0, 255, 0), 2)
        
        # NOVO: Mostrar penalidade de iluminação
        lighting_penalty = integrated_results.get('v2_1_lighting_score_penalty', 0)
        if lighting_penalty > 0:
            cv2.putText(highlighted, f"Luz: -{lighting_penalty}pts", 
                       (10, 120), font, 0.6, (0, 165, 255), 2)
        
        return highlighted, heatmap
