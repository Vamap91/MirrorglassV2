# integrated_analyzer.py
"""
MirrorGlass V2 - Analisador Integrado
Combina detecção de elementos legítimos (V2) + análise de textura (V1)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
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
        
        # 1. Detectar textos e etiquetas
        try:
            text_regions = self._detect_text_regions(image)
            if text_regions:
                results['text'] = text_regions
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de texto: {e}")
            
        # 2. Detectar papéis uniformes
        try:
            paper_regions = self._detect_paper_regions(image)
            if paper_regions:
                results['paper'] = paper_regions
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de papel: {e}")
            
        # 3. Detectar reflexos de vidro
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


class IntegratedTextureAnalyzer:
    """
    Analisador de textura integrado com detecção de elementos legítimos
    Combina V2 (filtro) + V1 (análise LBP)
    """
    
    def __init__(self, P=8, R=1, block_size=20, threshold=0.50, debug=False):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.debug = debug
        self.legitimate_detector = LegitimateElementDetector(debug=debug)
    
    def analyze_image_integrated(self, image):
        """
        Análise integrada: V2 (filtro) → V1 (análise de textura)
        """
        if self.debug:
            print("\n=== INICIANDO ANÁLISE INTEGRADA ===")
        
        # FASE 1: Detectar elementos legítimos (V2)
        if self.debug:
            print("FASE 1: Detectando elementos legítimos (V2)...")
        
        legitimate_elements = self.legitimate_detector.detect_all(image)
        exclusion_mask = self.legitimate_detector.create_exclusion_mask(image.shape[:2], min_confidence=0.3)
        
        # Estatísticas da exclusão
        h, w = image.shape[:2]
        total_pixels = h * w
        excluded_pixels = np.sum(exclusion_mask == 255)
        exclusion_percentage = (excluded_pixels / total_pixels) * 100
        
        if self.debug:
            print(f"  Elementos detectados: {list(legitimate_elements.keys())}")
            print(f"  Área excluída: {exclusion_percentage:.1f}%")
        
        # FASE 2: Análise de textura LBP apenas nas áreas NÃO excluídas (V1)
        if self.debug:
            print("FASE 2: Analisando textura (V1) nas áreas não excluídas...")
        
        texture_results = self._analyze_texture_with_mask(image, exclusion_mask)
        
        # Combinar resultados
        integrated_results = {
            'v2_legitimate_elements': legitimate_elements,
            'v2_exclusion_mask': exclusion_mask,
            'v2_exclusion_percentage': exclusion_percentage,
            'v1_texture_analysis': texture_results,
            'final_score': texture_results['naturalness_score'],
            'final_category': self._classify_naturalness(texture_results['naturalness_score']),
            'suspicious_areas_percentage': texture_results.get('suspicious_percentage', 0)
        }
        
        if self.debug:
            print(f"  Score final: {integrated_results['final_score']}")
            print(f"  Categoria: {integrated_results['final_category'][0]}")
            print("=== ANÁLISE CONCLUÍDA ===\n")
        
        return integrated_results
    
    def _analyze_texture_with_mask(self, image, exclusion_mask):
        """Análise de textura LBP aplicando máscara de exclusão"""
        # Converter para escala de cinza
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        # Calcular LBP
        lbp_image = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        # Análise em blocos
        height, width = img_gray.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        # Redimensionar máscara de exclusão para tamanho dos blocos
        exclusion_mask_resized = cv2.resize(exclusion_mask, (width, height))
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        exclusion_map = np.zeros((rows, cols))  # Marca blocos excluídos
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block_lbp = lbp_image[i:i+self.block_size, j:j+self.block_size]
                block_mask = exclusion_mask_resized[i:i+self.block_size, j:j+self.block_size]
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                # Verificar se bloco está em área excluída (mais de 50% excluído)
                exclusion_ratio = np.sum(block_mask == 255) / (self.block_size * self.block_size)
                
                if exclusion_ratio > 0.5:
                    # Bloco excluído - marcar mas não analisar
                    exclusion_map[row_idx, col_idx] = 1
                    variance_map[row_idx, col_idx] = 1.0  # Valor neutro (alta naturalidade)
                    entropy_map[row_idx, col_idx] = 1.0  # Valor neutro
                else:
                    # Bloco válido - analisar normalmente
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
        
        # Calcular mapa de naturalidade (70% entropia, 30% variância)
        naturalness_map = entropy_map * 0.7 + variance_map * 0.3
        
        # Normalizar
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Máscara de áreas suspeitas (APENAS em áreas não excluídas)
        suspicious_mask = (norm_naturalness_map < self.threshold) & (exclusion_map == 0)
        
        # Calcular score APENAS considerando áreas não excluídas
        valid_blocks = exclusion_map == 0
        if np.sum(valid_blocks) > 0:
            naturalness_score = int(np.mean(norm_naturalness_map[valid_blocks]) * 100)
            suspicious_percentage = float(np.sum(suspicious_mask) / np.sum(valid_blocks) * 100)
        else:
            naturalness_score = 100  # Se tudo foi excluído, considerar natural
            suspicious_percentage = 0.0
        
        # Criar heatmap
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
        """Gera relatório visual integrado"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        height, width = image.shape[:2]
        
        # Extrair dados
        naturalness_map = integrated_results['v1_texture_analysis']['naturalness_map']
        suspicious_mask = integrated_results['v1_texture_analysis']['suspicious_mask']
        exclusion_mask = integrated_results['v2_exclusion_mask']
        score = integrated_results['final_score']
        
        # Redimensionar mapas
        naturalness_map_resized = cv2.resize(naturalness_map, (width, height), interpolation=cv2.INTER_LINEAR)
        suspicious_mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        exclusion_mask_resized = cv2.resize(exclusion_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Criar heatmap
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Destacar áreas suspeitas (roxo)
        highlighted = overlay.copy()
        contours, _ = cv2.findContours(suspicious_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (128, 0, 128), 2)
        
        # Destacar áreas excluídas (verde)
        contours_excluded, _ = cv2.findContours(exclusion_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_excluded:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Adicionar texto
        category, _ = integrated_results['final_category']
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, f"Excluido: {integrated_results['v2_exclusion_percentage']:.1f}%", 
                   (10, 90), font, 0.6, (0, 255, 0), 2)
        
        return highlighted, heatmap
