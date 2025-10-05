# integrated_analyzer.py
"""
MirrorGlass V2.5 - VERSÃO OTIMIZADA
CORRIGIDO: Timeout no Copy-Move Detection
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import pytesseract
from dataclasses import dataclass
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
from scipy.fft import dct
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegionInfo:
    """Informações sobre uma região detectada"""
    mask: np.ndarray
    confidence: float
    type: str
    bbox: Tuple[int, int, int, int]
    metadata: Dict = None


class JPEGArtifactAnalyzer:
    """Detector 1: Análise de Artefatos JPEG - OTIMIZADO"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def analyze_jpeg_artifacts(self, image: np.ndarray) -> Dict:
        """Análise JPEG simplificada e rápida"""
        if self.debug:
            print("\n--- Análise JPEG (rápida) ---")
        
        try:
            # Análise simplificada de quantização
            quantization_anomaly = self._analyze_quantization_fast(image)
            
            jpeg_artifact_score = quantization_anomaly
            
            if self.debug:
                print(f"JPEG Score: {jpeg_artifact_score:.2%}")
            
            return {
                'jpeg_artifact_score': float(jpeg_artifact_score),
                'dct_inconsistency': 0.0,
                'quantization_anomaly': float(quantization_anomaly),
                'double_jpeg_score': 0.0
            }
        except Exception as e:
            if self.debug:
                print(f"Erro JPEG: {e}")
            return {
                'jpeg_artifact_score': 0.0,
                'dct_inconsistency': 0.0,
                'quantization_anomaly': 0.0,
                'double_jpeg_score': 0.0
            }
    
    def _analyze_quantization_fast(self, image: np.ndarray) -> float:
        """Versão rápida da análise de quantização"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar se muito grande
        max_dim = 512
        h, w = gray.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # Histograma
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        
        # FFT
        hist_fft = np.fft.fft(hist)
        power_spectrum = np.abs(hist_fft[1:64])  # Reduzido de 128 para 64
        
        # Detectar picos
        threshold = np.mean(power_spectrum) * 2
        peaks = power_spectrum[power_spectrum > threshold]
        
        if len(peaks) > 0:
            peak_strength = np.mean(peaks) / (np.mean(power_spectrum) + 1e-7)
            
            if peak_strength > 5:
                return 0.7
            elif peak_strength > 3:
                return 0.5
            elif peak_strength > 2:
                return 0.3
        
        return 0.0


class CopyMoveDetector:
    """Detector 2: Copy-Move OTIMIZADO para performance"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.max_blocks = 2000  # LIMITE CRÍTICO
    
    def detect_copy_move(self, image: np.ndarray) -> Dict:
        """Detecção Copy-Move OTIMIZADA"""
        if self.debug:
            print("\n--- Copy-Move (otimizado) ---")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # OTIMIZAÇÃO 1: Redimensionar imagem grande
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)
                h, w = gray.shape
                if self.debug:
                    print(f"  Redimensionado para: {w}x{h}")
            
            # OTIMIZAÇÃO 2: Blocos maiores, menos overlap
            block_size = 32  # Era 16
            step = block_size  # Era block_size // 2 (sem overlap!)
            
            blocks = []
            positions = []
            
            for i in range(0, h - block_size, step):
                for j in range(0, w - block_size, step):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Feature simples: média dos pixels
                    feature = np.mean(block)
                    blocks.append(feature)
                    positions.append((i, j))
                    
                    # OTIMIZAÇÃO 3: Limite de blocos
                    if len(blocks) >= self.max_blocks:
                        break
                if len(blocks) >= self.max_blocks:
                    break
            
            n_blocks = len(blocks)
            
            if self.debug:
                print(f"  Blocos analisados: {n_blocks}")
            
            # OTIMIZAÇÃO 4: Comparação simplificada
            similar_pairs = 0
            threshold = 5.0  # Diferença máxima de média
            
            # Apenas amostra aleatória de comparações
            max_comparisons = min(10000, n_blocks * 5)
            comparisons_done = 0
            
            for _ in range(max_comparisons):
                i = np.random.randint(0, n_blocks)
                j = np.random.randint(0, n_blocks)
                
                if i == j:
                    continue
                
                # Distância simples
                if abs(blocks[i] - blocks[j]) < threshold:
                    # Verificar se não são adjacentes
                    pos_i = positions[i]
                    pos_j = positions[j]
                    distance = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                    
                    if distance > block_size * 2:
                        similar_pairs += 1
                
                comparisons_done += 1
            
            # Score baseado em ratio
            copy_move_ratio = similar_pairs / max(comparisons_done, 1)
            
            if copy_move_ratio > 0.15:
                copy_move_score = 0.8
            elif copy_move_ratio > 0.10:
                copy_move_score = 0.6
            elif copy_move_ratio > 0.05:
                copy_move_score = 0.4
            else:
                copy_move_score = 0.2
            
            if self.debug:
                print(f"  Similar pairs ratio: {copy_move_ratio:.2%}")
                print(f"  Score: {copy_move_score:.2%}")
            
            return {
                'copy_move_score': float(copy_move_score),
                'similar_pairs_count': int(similar_pairs),
                'total_blocks': int(n_blocks)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Erro Copy-Move: {e}")
            return {
                'copy_move_score': 0.0,
                'similar_pairs_count': 0,
                'total_blocks': 0
            }


class LegitimateElementDetector:
    """Detecta elementos legítimos - COM TIMEOUT"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.detection_results = {}
        
    def detect_all(self, image: np.ndarray) -> Dict[str, RegionInfo]:
        if image is None or image.size == 0:
            raise ValueError("Imagem inválida")
            
        results = {}
        
        # OCR com timeout implícito
        try:
            text_regions = self._detect_text_regions(image)
            if text_regions:
                results['text'] = text_regions
        except Exception as e:
            if self.debug:
                print(f"Erro texto: {e}")
            
        # Papel
        try:
            paper_regions = self._detect_paper_regions(image)
            if paper_regions:
                results['paper'] = paper_regions
        except Exception as e:
            if self.debug:
                print(f"Erro papel: {e}")
            
        # Reflexos
        try:
            reflection_regions = self._detect_glass_reflections(image)
            if reflection_regions:
                results['reflections'] = reflection_regions
        except Exception as e:
            if self.debug:
                print(f"Erro reflexos: {e}")
        
        self.detection_results = results
        return results
    
    def _detect_text_regions(self, image: np.ndarray) -> Optional[RegionInfo]:
        """OCR com configuração timeout-friendly"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # OTIMIZAÇÃO: Redimensionar se muito grande
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            h, w = gray.shape
        
        text_mask = np.zeros((h, w), dtype=np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        try:
            # Timeout config: usar modo mais rápido
            data = pytesseract.image_to_data(
                enhanced, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6 --oem 1'  # Modo mais rápido
            )
            
            valid_text_count = 0
            total_conf = 0
            detected_texts = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:
                    text = str(data['text'][i]).strip()
                    
                    if len(text) >= 2 and sum(c.isalnum() for c in text) >= 2:
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
                
                # Redimensionar máscara de volta se necessário
                if text_mask.shape != (image.shape[0], image.shape[1]):
                    text_mask = cv2.resize(text_mask, (image.shape[1], image.shape[0]))
                
                contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w_c, h_c = cv2.boundingRect(np.vstack(contours))
                    bbox = (int(x), int(y), int(w_c), int(h_c))
                else:
                    bbox = (0, 0, image.shape[1], image.shape[0])
                
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
        """Detecção de papel - mantida do original"""
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
            
            if 5000 < area < h * w * 0.6:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                if 4 <= len(approx) <= 6:
                    x, y, w_c, h_c = cv2.boundingRect(contour)
                    aspect_ratio = float(w_c) / h_c if h_c > 0 else 0
                    
                    if 0.3 < aspect_ratio < 3.0:
                        roi_uniform = uniform_mask[y:y+h_c, x:x+w_c]
                        
                        if roi_uniform.size > 0:
                            uniformity_ratio = np.sum(roi_uniform == 255) / roi_uniform.size
                            
                            if uniformity_ratio > 0.5:
                                roi_color = gray[y:y+h_c, x:x+w_c]
                                mean_brightness = np.mean(roi_color)
                                
                                if mean_brightness > 100:
                                    cv2.drawContours(paper_mask, [contour], -1, 255, -1)
                                    valid_papers.append({
                                        'bbox': (int(x), int(y), int(w_c), int(h_c)),
                                        'uniformity': float(uniformity_ratio),
                                        'brightness': float(mean_brightness)
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
                metadata={'paper_count': len(valid_papers)}
            )
        
        return None
    
    def _detect_glass_reflections(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecção de reflexos - mantida do original"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
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
        h, w = image_shape
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        for region_type, region_info in self.detection_results.items():
            if region_info and region_info.confidence >= min_confidence:
                exclusion_mask = cv2.bitwise_or(exclusion_mask, region_info.mask)
        
        return exclusion_mask


class IntegratedTextureAnalyzer:
    """Analisador V2.5 - OTIMIZADO"""
    
    def __init__(self, P=8, R=1, block_size=20, threshold=0.50, debug=False):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.debug = debug
        self.legitimate_detector = LegitimateElementDetector(debug=debug)
        self.jpeg_analyzer = JPEGArtifactAnalyzer(debug=debug)
        self.copy_move_detector = CopyMoveDetector(debug=debug)
    
    def analyze_image_integrated(self, image):
        """Análise integrada OTIMIZADA"""
        if self.debug:
            print("\n=== ANÁLISE V2.5 OTIMIZADA ===")
        
        # FASE 1: Elementos legítimos
        if self.debug:
            print("FASE 1: Detectando elementos legítimos...")
        legitimate_elements = self.legitimate_detector.detect_all(image)
        exclusion_mask = self.legitimate_detector.create_exclusion_mask(image.shape[:2], min_confidence=0.3)
        
        h, w = image.shape[:2]
        total_pixels = h * w
        excluded_pixels = np.sum(exclusion_mask == 255)
        exclusion_percentage = (excluded_pixels / total_pixels) * 100
        
        # FASE 2: Análise JPEG (simplificada)
        if self.debug:
            print("FASE 2: Análise JPEG...")
        jpeg_analysis = self.jpeg_analyzer.analyze_jpeg_artifacts(image)
        jpeg_score = jpeg_analysis['jpeg_artifact_score']
        
        # FASE 3: Copy-Move (otimizada)
        if self.debug:
            print("FASE 3: Copy-Move detection...")
        copy_move_analysis = self.copy_move_detector.detect_copy_move(image)
        copy_move_score = copy_move_analysis['copy_move_score']
        
        # FASE 4: Textura LBP
        if self.debug:
            print("FASE 4: Análise LBP...")
        texture_results = self._analyze_texture_with_mask(image, exclusion_mask)
        base_score = texture_results['naturalness_score']
        
        # Penalidades reduzidas
        jpeg_penalty = 0
        if jpeg_score > 0.6:
            jpeg_penalty = 20
        elif jpeg_score > 0.4:
            jpeg_penalty = 10
        
        copy_move_penalty = 0
        if copy_move_score > 0.7:
            copy_move_penalty = 25
        elif copy_move_score > 0.5:
            copy_move_penalty = 15
        
        final_score = max(0, base_score - jpeg_penalty - copy_move_penalty)
        
        integrated_results = {
            'v2_legitimate_elements': legitimate_elements,
            'v2_exclusion_mask': exclusion_mask,
            'v2_exclusion_percentage': exclusion_percentage,
            'v2_5_jpeg_analysis': jpeg_analysis,
            'v2_5_jpeg_penalty': jpeg_penalty,
            'v2_5_copy_move_analysis': copy_move_analysis,
            'v2_5_copy_move_penalty': copy_move_penalty,
            'v1_texture_analysis': texture_results,
            'base_texture_score': base_score,
            'final_score': final_score,
            'final_category': self._classify_naturalness(final_score),
            'suspicious_areas_percentage': texture_results.get('suspicious_percentage', 0)
        }
        
        if self.debug:
            print(f"\nRESULTADO:")
            print(f"  Base: {base_score}")
            print(f"  JPEG penalty: -{jpeg_penalty}")
            print(f"  Copy-Move penalty: -{copy_move_penalty}")
            print(f"  FINAL: {final_score}")
        
        return integrated_results
    
    def _analyze_texture_with_mask(self, image, exclusion_mask):
        """Análise LBP - mantida do original"""
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
        if score == 0:
            return "Erro de análise", "Score zero - Revisar imagem manualmente"
        elif score <= 15:
            return "Análise inconclusiva", "Score muito baixo"
        elif score <= 45:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, integrated_results):
        """Gera visualização - mantida do original"""
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
        
        contours, _ = cv2.findContours(suspicious_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (128, 0, 128), 2)
        
        contours_excluded, _ = cv2.findContours(exclusion_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_excluded:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
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
        
        jpeg_penalty = integrated_results.get('v2_5_jpeg_penalty', 0)
        if jpeg_penalty > 0:
            cv2.putText(highlighted, f"JPEG: -{jpeg_penalty}pts", 
                       (10, y_offset), font, 0.6, (255, 0, 0), 2)
            y_offset += 30
        
        copy_move_penalty = integrated_results.get('v2_5_copy_move_penalty', 0)
        if copy_move_penalty > 0:
            cv2.putText(highlighted, f"Clone: -{copy_move_penalty}pts", 
                       (10, y_offset), font, 0.6, (0, 0, 255), 2)
        
        return highlighted, heatmap
