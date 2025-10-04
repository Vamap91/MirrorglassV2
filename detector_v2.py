# ============================================================
# ARQUIVO 1: detector_v2.py
# ============================================================
"""
MirrorglassV2 - Sistema de Detecção de Fraudes em Imagens Automotivas
Versão 2.0 - Foco: Eliminar falsos positivos causados por elementos legítimos
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import pytesseract
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class RegionInfo:
    """Informações sobre uma região detectada na imagem"""
    mask: np.ndarray
    confidence: float
    type: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    metadata: Dict = None
    
    def to_dict(self):
        """Converte para dicionário (para JSON)"""
        return {
            'confidence': float(self.confidence),
            'type': self.type,
            'bbox': self.bbox,
            'area_pixels': int(np.sum(self.mask == 255)),
            'metadata': self.metadata or {}
        }


class LegitimateElementDetector:
    """
    Detecta elementos legítimos que NÃO devem ser considerados suspeitos:
    - Textos e etiquetas
    - Papéis e documentos
    - Reflexos de vidro
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.detection_results = {}
        
    def detect_all(self, image: np.ndarray) -> Dict[str, RegionInfo]:
        """
        Detecta todos os elementos legítimos na imagem
        
        Args:
            image: Imagem BGR (formato OpenCV)
            
        Returns:
            Dict com máscaras de cada tipo de elemento detectado
        """
        if image is None or image.size == 0:
            raise ValueError("Imagem inválida")
            
        results = {}
        
        if self.debug:
            print(f"Analisando imagem: {image.shape}")
        
        # 1. Detectar textos e etiquetas
        try:
            text_regions = self._detect_text_regions(image)
            if text_regions:
                results['text'] = text_regions
                if self.debug:
                    print(f"✓ Texto detectado (conf: {text_regions.confidence:.2%})")
        except Exception as e:
            if self.debug:
                print(f"⚠ Erro na detecção de texto: {e}")
            
        # 2. Detectar papéis uniformes
        try:
            paper_regions = self._detect_paper_regions(image)
            if paper_regions:
                results['paper'] = paper_regions
                if self.debug:
                    print(f"✓ Papel detectado (conf: {paper_regions.confidence:.2%})")
        except Exception as e:
            if self.debug:
                print(f"⚠ Erro na detecção de papel: {e}")
            
        # 3. Detectar reflexos de vidro
        try:
            reflection_regions = self._detect_glass_reflections(image)
            if reflection_regions:
                results['reflections'] = reflection_regions
                if self.debug:
                    print(f"✓ Reflexos detectados (conf: {reflection_regions.confidence:.2%})")
        except Exception as e:
            if self.debug:
                print(f"⚠ Erro na detecção de reflexos: {e}")
        
        self.detection_results = results
        return results
    
    def _detect_text_regions(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecta regiões com texto impresso (etiquetas, documentos)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        text_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Melhorar contraste para OCR
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
                                
                                if self.debug:
                                    print(f"  Texto: '{text}' (conf: {conf})")
            
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
                    metadata={
                        'text_count': valid_text_count,
                        'texts': detected_texts[:10]
                    }
                )
        except Exception as e:
            if self.debug:
                print(f"Erro OCR: {e}")
        
        return None
    
    def _detect_paper_regions(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecta regiões com papéis uniformes"""
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
                        
                        if self.debug:
                            print(f"  Papel: bbox={x,y,w_c,h_c}, unif={uniformity_ratio:.2f}")
        
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
                metadata={
                    'paper_count': len(valid_papers),
                    'papers': valid_papers
                }
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
            
            if self.debug:
                print(f"  Reflexos: {reflection_ratio*100:.1f}% da imagem")
            
            return RegionInfo(
                mask=reflection_mask,
                confidence=min(float(reflection_ratio * 5), 1.0),
                type='reflection',
                bbox=bbox,
                metadata={
                    'reflection_percentage': float(reflection_ratio * 100)
                }
            )
        
        return None
    
    def create_exclusion_mask(self, image_shape: Tuple[int, int], 
                            min_confidence: float = 0.3) -> np.ndarray:
        """Cria máscara de exclusão combinando todas as detecções"""
        h, w = image_shape
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        for region_type, region_info in self.detection_results.items():
            if region_info and region_info.confidence >= min_confidence:
                exclusion_mask = cv2.bitwise_or(exclusion_mask, region_info.mask)
        
        return exclusion_mask
    
    def get_summary(self) -> Dict:
        """Retorna resumo das detecções em formato JSON-friendly"""
        summary = {
            'total_detections': len(self.detection_results),
            'detections': {}
        }
        
        for region_type, region_info in self.detection_results.items():
            if region_info:
                summary['detections'][region_type] = region_info.to_dict()
        
        return summary
    
    def visualize_detections(self, image: np.ndarray, 
                            show_labels: bool = True) -> np.ndarray:
        """Cria visualização das detecções para debug"""
        vis = image.copy()
        
        colors = {
            'text': (0, 255, 0),
            'paper': (255, 0, 0),
            'reflections': (0, 255, 255)
        }
        
        for region_type, region_info in self.detection_results.items():
            if region_info:
                color = colors.get(region_type, (255, 255, 255))
                
                mask_colored = np.zeros_like(image)
                mask_colored[region_info.mask == 255] = color
                vis = cv2.addWeighted(vis, 1.0, mask_colored, 0.3, 0)
                
                x, y, w_bbox, h_bbox = region_info.bbox
                cv2.rectangle(vis, (x, y), (x+w_bbox, y+h_bbox), color, 2)
                
                if show_labels:
                    label = f"{region_type}: {region_info.confidence:.0%}"
                    
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis, (x, y-text_h-10), (x+text_w+10, y), color, -1)
                    
                    cv2.putText(vis, label, (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis


class ImprovedFraudDetector:
    """
    Detector de fraudes melhorado que integra:
    1. Detecção de elementos legítimos
    2. Análise de textura em áreas relevantes
    3. Score ajustado
    """
    
    def __init__(self, debug=False):
        self.legitimate_detector = LegitimateElementDetector(debug=debug)
        self.debug = debug
        
    def analyze_image(self, image: np.ndarray) -> Dict:
        """Análise completa da imagem"""
        h, w = image.shape[:2]
        
        legitimate_elements = self.legitimate_detector.detect_all(image)
        
        exclusion_mask = self.legitimate_detector.create_exclusion_mask(
            image.shape[:2], 
            min_confidence=0.3
        )
        
        total_pixels = h * w
        excluded_pixels = np.sum(exclusion_mask == 255)
        analyzed_pixels = total_pixels - excluded_pixels
        
        exclusion_percentage = (excluded_pixels / total_pixels) * 100
        analyzed_percentage = (analyzed_pixels / total_pixels) * 100
        
        result = {
            'image_shape': (h, w),
            'total_pixels': int(total_pixels),
            'excluded_pixels': int(excluded_pixels),
            'analyzed_pixels': int(analyzed_pixels),
            'exclusion_percentage': float(exclusion_percentage),
            'analyzed_percentage': float(analyzed_percentage),
            'legitimate_elements': self.legitimate_detector.get_summary(),
            'exclusion_mask': exclusion_mask,
            'recommendation': self._get_recommendation(exclusion_percentage)
        }
        
        return result
    
    def _get_recommendation(self, exclusion_percentage: float) -> str:
        """Gera recomendação baseada na porcentagem de exclusão"""
        if exclusion_percentage > 70:
            return "ALTO: Imagem contém muitos elementos legítimos (texto/papel/reflexos). Análise limitada."
        elif exclusion_percentage > 40:
            return "MÉDIO: Presença significativa de elementos legítimos. Focar análise em áreas restantes."
        elif exclusion_percentage > 10:
            return "BAIXO: Poucos elementos legítimos detectados. Análise normal pode prosseguir."
        else:
            return "MÍNIMO: Imagem livre de elementos legítimos óbvios. Análise completa recomendada."
