"""
MirrorGlass V2.6 - Sistema Simplificado e Funcional
Arquitetura: Filtro (texto/papel/reflexo) + LBP + GAN Fingerprint
Acurácia esperada: 80-85%
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import pytesseract
from dataclasses import dataclass
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegionInfo:
    """Informações sobre região detectada"""
    mask: np.ndarray
    confidence: float
    type: str
    bbox: Tuple[int, int, int, int]
    metadata: Dict = None


class LegitimateElementDetector:
    """
    Detecta elementos legítimos que não devem ser analisados:
    - Textos e etiquetas (OCR)
    - Papéis uniformes
    - Reflexos de vidro
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.detection_results = {}
        
    def detect_all(self, image: np.ndarray) -> Dict[str, RegionInfo]:
        """Detecta todos os elementos legítimos"""
        if image is None or image.size == 0:
            raise ValueError("Imagem inválida")
            
        results = {}
        
        # 1. Texto
        try:
            text_regions = self._detect_text_regions(image)
            if text_regions:
                results['text'] = text_regions
                if self.debug:
                    print(f"Texto detectado: {text_regions.confidence:.0%}")
        except Exception as e:
            if self.debug:
                print(f"Erro texto: {e}")
            
        # 2. Papel
        try:
            paper_regions = self._detect_paper_regions(image)
            if paper_regions:
                results['paper'] = paper_regions
                if self.debug:
                    print(f"Papel detectado: {paper_regions.confidence:.0%}")
        except Exception as e:
            if self.debug:
                print(f"Erro papel: {e}")
            
        # 3. Reflexos
        try:
            reflection_regions = self._detect_glass_reflections(image)
            if reflection_regions:
                results['reflections'] = reflection_regions
                if self.debug:
                    print(f"Reflexos detectados: {reflection_regions.confidence:.0%}")
        except Exception as e:
            if self.debug:
                print(f"Erro reflexos: {e}")
        
        self.detection_results = results
        return results
    
    def _detect_text_regions(self, image: np.ndarray) -> Optional[RegionInfo]:
        """Detecta texto via OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Redimensionar se muito grande
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            h, w = gray.shape
        
        text_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Melhorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        try:
            data = pytesseract.image_to_data(
                enhanced, 
                output_type=pytesseract.Output.DICT,
                config='--psm 11'
            )
            
            valid_count = 0
            total_conf = 0
            texts = []
            
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
                            valid_count += 1
                            total_conf += int(conf)
                            texts.append(text)
            
            if valid_count > 0:
                avg_conf = total_conf / valid_count
                
                kernel = np.ones((15, 15), np.uint8)
                text_mask = cv2.dilate(text_mask, kernel, iterations=1)
                
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
                    metadata={'text_count': valid_count, 'texts': texts[:10]}
                )
        except Exception:
            pass
        
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
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
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
                            uniformity = np.sum(roi_uniform == 255) / roi_uniform.size
                            
                            if uniformity > 0.5:
                                roi_gray = gray[y:y+h_c, x:x+w_c]
                                brightness = np.mean(roi_gray)
                                
                                if brightness > 100:
                                    cv2.drawContours(paper_mask, [contour], -1, 255, -1)
                                    valid_papers.append({
                                        'bbox': (int(x), int(y), int(w_c), int(h_c)),
                                        'uniformity': float(uniformity)
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
        """Detecta reflexos em vidro"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        low_sat = s_channel < 50
        high_val = v_channel > 200
        potential_reflections = (low_sat & high_val).astype(np.uint8) * 255
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float64)
        grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        smooth_grads = ((grad_mag > 10) & (grad_mag < 80)).astype(np.uint8) * 255
        reflection_mask = cv2.bitwise_and(potential_reflections, smooth_grads)
        
        kernel = np.ones((5, 5), np.uint8)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        
        reflection_ratio = np.sum(reflection_mask == 255) / (h * w)
        
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
                metadata={'coverage': float(reflection_ratio * 100)}
            )
        
        return None
    
    def create_exclusion_mask(self, image_shape: Tuple[int, int], min_confidence: float = 0.3) -> np.ndarray:
        """Cria máscara combinada de exclusão"""
        h, w = image_shape
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        for region_type, region_info in self.detection_results.items():
            if region_info and region_info.confidence >= min_confidence:
                exclusion_mask = cv2.bitwise_or(exclusion_mask, region_info.mask)
        
        return exclusion_mask


class GANFingerprintDetector:
    """
    Detecta artefatos de IA via análise espectral.
    Foca em: checkerboard patterns + noise inconsistency
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def analyze(self, image: np.ndarray) -> Dict:
        """Análise de artefatos de GAN"""
        if self.debug:
            print("\n--- GAN Fingerprint Analysis ---")
        
        try:
            checkerboard_score = self._detect_checkerboard(image)
            noise_score = self._analyze_noise_patterns(image)
            
            # Combinar scores
            if checkerboard_score > 0.7 and noise_score > 0.6:
                gan_score = 0.9
            elif checkerboard_score > 0.6 or noise_score > 0.6:
                gan_score = 0.7
            elif checkerboard_score > 0.4 or noise_score > 0.4:
                gan_score = 0.5
            else:
                gan_score = 0.2
            
            if self.debug:
                print(f"Checkerboard: {checkerboard_score:.2%}")
                print(f"Noise: {noise_score:.2%}")
                print(f"GAN Score: {gan_score:.2%}")
            
            return {
                'gan_score': float(gan_score),
                'checkerboard_score': float(checkerboard_score),
                'noise_score': float(noise_score)
            }
        except Exception as e:
            if self.debug:
                print(f"Erro GAN: {e}")
            return {'gan_score': 0.0, 'checkerboard_score': 0.0, 'noise_score': 0.0}
    
    def _detect_checkerboard(self, image: np.ndarray) -> float:
        """Detecta padrão xadrez via FFT 2D"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        size = 512
        if max(gray.shape) > size:
            gray = cv2.resize(gray, (size, size))
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        magnitude = np.log(magnitude + 1)
        
        center = magnitude.shape[0] // 2
        diag_45 = np.diagonal(magnitude, offset=center//2)
        diag_135 = np.diagonal(np.fliplr(magnitude), offset=center//2)
        
        diag_energy = (np.mean(diag_45) + np.mean(diag_135)) / 2
        total_energy = np.mean(magnitude)
        ratio = diag_energy / (total_energy + 1e-7)
        
        if ratio > 1.5:
            return 0.8
        elif ratio > 1.3:
            return 0.6
        elif ratio > 1.15:
            return 0.4
        else:
            return 0.2
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> float:
        """Analisa consistência de ruído"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        high_freq = cv2.absdiff(gray, gaussian)
        
        h, w = high_freq.shape
        patch_size = 64
        noise_variances = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = high_freq[i:i+patch_size, j:j+patch_size]
                
                if np.std(gaussian[i:i+patch_size, j:j+patch_size]) < 20:
                    noise_var = np.var(patch)
                    noise_variances.append(noise_var)
        
        if len(noise_variances) < 5:
            return 0.0
        
        cv = np.std(noise_variances) / (np.mean(noise_variances) + 1e-7)
        
        if cv < 0.3 or cv > 2.0:
            return 0.7
        elif cv < 0.5 or cv > 1.5:
            return 0.5
        else:
            return 0.2


class SimplifiedTextureAnalyzer:
    """
    Analisador de textura LBP simplificado.
    Usa exclusion_mask para ignorar texto/papel/reflexos.
    """
    
    def __init__(self, P=8, R=1, block_size=20, threshold=0.50, debug=False):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.debug = debug
        self.legitimate_detector = LegitimateElementDetector(debug=debug)
        self.gan_detector = GANFingerprintDetector(debug=debug)
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """Análise completa da imagem"""
        if self.debug:
            print("\n=== ANÁLISE V2.6 ===")
        
        # FASE 1: Elementos legítimos
        if self.debug:
            print("FASE 1: Elementos legítimos...")
        
        legitimate_elements = self.legitimate_detector.detect_all(image)
        exclusion_mask = self.legitimate_detector.create_exclusion_mask(
            image.shape[:2], 
            min_confidence=0.3
        )
        
        h, w = image.shape[:2]
        total_pixels = h * w
        excluded_pixels = np.sum(exclusion_mask == 255)
        exclusion_percentage = (excluded_pixels / total_pixels) * 100
        
        # FASE 2: Análise LBP
        if self.debug:
            print("FASE 2: Análise LBP...")
        
        texture_results = self._analyze_texture_with_mask(image, exclusion_mask)
        lbp_score = texture_results['naturalness_score']
        
        # FASE 3: GAN Fingerprint
        if self.debug:
            print("FASE 3: GAN Fingerprint...")
        
        gan_results = self.gan_detector.analyze(image)
        gan_score = gan_results['gan_score']
        
        # Combinar scores
        gan_penalty = 0
        if gan_score > 0.7:
            gan_penalty = 15
        elif gan_score > 0.5:
            gan_penalty = 8
        
        final_score = max(0, lbp_score - gan_penalty)
        
        results = {
            'legitimate_elements': legitimate_elements,
            'exclusion_mask': exclusion_mask,
            'exclusion_percentage': float(exclusion_percentage),
            'texture_results': texture_results,
            'lbp_score': int(lbp_score),
            'gan_results': gan_results,
            'gan_penalty': int(gan_penalty),
            'final_score': int(final_score),
            'category': self._classify_score(final_score),
            'suspicious_percentage': float(texture_results['suspicious_percentage'])
        }
        
        if self.debug:
            print(f"\nRESULTADO:")
            print(f"  LBP: {lbp_score}")
            print(f"  GAN penalty: -{gan_penalty}")
            print(f"  FINAL: {final_score}")
        
        return results
    
    def _analyze_texture_with_mask(self, image: np.ndarray, exclusion_mask: np.ndarray) -> Dict:
        """Análise LBP com máscara de exclusão"""
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            'heatmap': heatmap
        }
    
    def _classify_score(self, score: int) -> Tuple[str, str]:
        """Classifica o score"""
        if score <= 30:
            return ("Alta chance de manipulação por IA", "Múltiplos indicadores de artificialidade")
        elif score <= 50:
            return ("Suspeita de manipulação", "Textura artificial detectada")
        elif score <= 70:
            return ("Textura questionável", "Revisão manual recomendada")
        else:
            return ("Textura natural", "Baixa chance de manipulação")
    
    def generate_visual_report(self, image: np.ndarray, results: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Gera visualização dos resultados"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        height, width = image.shape[:2]
        
        naturalness_map = results['texture_results']['naturalness_map']
        suspicious_mask = results['texture_results']['suspicious_mask']
        exclusion_mask = results['exclusion_mask']
        score = results['final_score']
        
        naturalness_resized = cv2.resize(naturalness_map, (width, height), interpolation=cv2.INTER_LINEAR)
        suspicious_resized = cv2.resize(suspicious_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        exclusion_resized = cv2.resize(exclusion_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        heatmap = cv2.applyColorMap((naturalness_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        highlighted = overlay.copy()
        
        # Áreas suspeitas (roxo)
        contours, _ = cv2.findContours(suspicious_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (128, 0, 128), 2)
        
        # Áreas excluídas (verde)
        contours_exc, _ = cv2.findContours(exclusion_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_exc:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Texto
        category, _ = results['category']
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        y_offset = 30
        cv2.putText(highlighted, f"Score: {score}/100", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(highlighted, category, (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(highlighted, f"Excluido: {results['exclusion_percentage']:.1f}%", 
                   (10, y_offset), font, 0.6, (0, 255, 0), 2)
        
        if results['gan_penalty'] > 0:
            y_offset += 30
            cv2.putText(highlighted, f"GAN: -{results['gan_penalty']}pts", 
                       (10, y_offset), font, 0.6, (255, 165, 0), 2)
        
        return highlighted, heatmap
