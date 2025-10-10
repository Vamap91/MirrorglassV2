# texture_analyzer.py
# Sistema de Análise Sequencial com Validação em Cadeia
# Versão: 4.4.0 - limiar dinâmico P25 + co-localização textura×borda + exit prudente na Fase 2

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import io
import base64


# ================================
# TEXTURA (SEM CLAHE)
# ================================
class TextureAnalyzer:
    """Detector primário por textura usando LBP multi-escala (SEM CLAHE)."""

    def __init__(self, block_size=24):
        self.block_size = int(block_size)

    # ---------- helpers ----------
    def _ensure_gray_uint8(self, image):
        if isinstance(image, Image.Image):
            img = np.array(image.convert("L"))
        elif image.ndim == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()

        # Normaliza tamanho para estabilidade estatística
        h, w = img.shape
        max_side = max(h, w)
        if max_side > 1200:
            scale = 1200.0 / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def calculate_lbp_multiscale(self, image):
        """
        LBP em duas escalas:
          - (P=8,  R=1)   → microtextura
          - (P=16, R=2)   → macrotextura
        Combina em [0,1] (peso maior para macro).
        """
        gray = self._ensure_gray_uint8(image)
        lbp1 = local_binary_pattern(gray, 8, 1, method="uniform")     # 0..10
        lbp2 = local_binary_pattern(gray, 16, 2, method="uniform")    # 0..18
        lbp1_n = lbp1 / float(8 + 2)
        lbp2_n = lbp2 / float(16 + 2)
        lbp_combined = np.clip(0.4 * lbp1_n + 0.6 * lbp2_n, 0.0, 1.0)
        return lbp_combined, gray

    # ---------- core ----------
    def analyze_texture_variance(self, image):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        lbp_norm, gray = self.calculate_lbp_multiscale(image_np)  # ∈ [0,1]
        H, W = lbp_norm.shape
        rows = max(1, H // self.block_size)
        cols = max(1, W // self.block_size)

        variance_map   = np.zeros((rows, cols), dtype=np.float32)
        entropy_map    = np.zeros((rows, cols), dtype=np.float32)
        uniformity_map = np.zeros((rows, cols), dtype=np.float32)
        edge_map_local = np.zeros((rows, cols), dtype=np.float32)

        # mapa de borda p/ “gate” e co-localização
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

        for i in range(0, H - self.block_size + 1, self.block_size):
            for j in range(0, W - self.block_size + 1, self.block_size):
                r = i // self.block_size
                c = j // self.block_size

                block = lbp_norm[i:i+self.block_size, j:j+self.block_size]

                # histograma 0..1 (10 bins)
                hist, _ = np.histogram(block, bins=10, range=(0, 1))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
                e = float(entropy(hist) / np.log(10))        # 0..1
                v = float(np.var(block))                     # 0..~0.2
                u = float(1.0 - np.max(hist))                # 0..1

                block_mag = mag[i:i+self.block_size, j:j+self.block_size]
                edge_map_local[r, c] = float(np.mean(block_mag))

                variance_map[r, c]   = v
                entropy_map[r, c]    = e
                uniformity_map[r, c] = u

        # força de borda normalizada por percentil (robusto a outliers)
        p90 = np.percentile(edge_map_local, 90) + 1e-6
        edge_strength = np.clip(edge_map_local / p90, 0, 1)  # 0..1

        # mapa de “naturalidade”
        naturalness_map = 0.50 * variance_map + 0.30 * entropy_map + 0.20 * uniformity_map

        # -------- Limiar DINÂMICO (percentil + ajuste por detalhe global) --------
        # Menos agressivo: P25 reduz falso-positivo em lataria/vidro reais.
        detail_level = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        thr_dyn = float(np.percentile(naturalness_map, 25))   # <— P25 (antes P30)
        if detail_level < 40:     # cena lisa → relaxa
            thr_dyn *= 0.88
        elif detail_level > 150:  # cena muito detalhada → exige mais
            thr_dyn *= 1.10
        thr_dyn = float(np.clip(thr_dyn, 0.04, 0.60))
        # -------------------------------------------------------------------------

        # máscara suspeita + gate por borda (suspeitos em borda pesam menos)
        suspicious_mask = (naturalness_map < thr_dyn).astype(np.float32)
        gate = (1.0 - 0.6 * edge_strength)  # 0.4 .. 1.0
        gated_suspicious = suspicious_mask * gate

        # Componente A: razão de suspeitos (quanto menor, melhor)
        suspicious_ratio = float(np.mean(gated_suspicious))
        comp_ratio = 1.0 - suspicious_ratio                   # 0..1

        # Componente B: nível médio normalizado no contexto da imagem
        nm_min, nm_max = float(naturalness_map.min()), float(naturalness_map.max())
        comp_level = float((naturalness_map.mean() - nm_min) / (nm_max - nm_min + 1e-6))
        comp_level = float(np.clip(comp_level, 0.0, 1.0))

        # Score final 60/40
        naturalness_score = int(np.clip(0.6 * comp_ratio + 0.4 * comp_level, 0, 1) * 100)

        # --- Co-localização textura×borda para absolvição posterior/exit prudente ---
        # Quanto da suspeita está em regiões de *baixa borda* (edge_strength < 0.3)?
        # Se for baixo, a “suspeita” coincide com bordas fortes (amassado/contorno), sinal pró-real.
        low_edge = (edge_strength < 0.3).astype(np.float32)
        total_susp = float(suspicious_mask.sum() + 1e-6)
        overlap_low_edge = float((suspicious_mask * low_edge).sum() / total_susp)  # 0..1
        # ---------------------------------------------------------------------------

        # heatmap para visualização
        disp = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return {
            "variance_map": variance_map,
            "naturalness_map": naturalness_map,
            "suspicious_mask": (suspicious_mask > 0.5),
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio,
            "mean_naturalness_raw": float(naturalness_map.mean()),
            "edge_strength_map": edge_strength,         # para análises posteriores
            "overlap_low_edge": overlap_low_edge        # <— NOVO: usado no sequencial
        }

    def classify_naturalness(self, score):
        if score <= 40:
            return "Alta chance de manipulação", "Textura muito artificial"
        elif score <= 68:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"

    def generate_visual_report(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]

        h, w = image.shape[:2]
        nm_resized = cv2.resize(naturalness_map, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        disp = cv2.normalize(nm_resized, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        highlighted = overlay.copy()

        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)

        category, _ = self.classify_naturalness(score)
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


# ================================
# BORDAS (COM CLAHE)
# ================================
class EdgeAnalyzer:
    """Análise de bordas COM CLAHE — coerência direcional + densidade."""

    def __init__(self, block_size=24, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = int(block_size)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_size = int(clahe_tile_size)

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
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)

    def compute_gradients(self, image):
        gray = self._convert_to_gray(image)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        return {"magnitude": magnitude, "direction": direction}

    def analyze_edge_coherence(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        gray = self._convert_to_gray(image)
        h, w = gray.shape

        grads = self.compute_gradients(image)
        magnitude = grads["magnitude"]
        direction = grads["direction"]

        rows = max(1, h // self.block_size)
        cols = max(1, w // self.block_size)

        coherence_map = np.zeros((rows, cols), dtype=np.float32)
        edge_density_map = np.zeros((rows, cols), dtype=np.float32)

        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                r = i // self.block_size
                c = j // self.block_size

                block_mag = magnitude[i:i+self.block_size, j:j+self.block_size]
                block_dir = direction[i:i+self.block_size, j:j+self.block_size]

                edge_density_map[r, c] = float(np.mean(block_mag))  # normaliza adiante

                if np.sum(block_mag > np.percentile(block_mag, 60)) > 8:
                    significant = block_mag > np.percentile(block_mag, 70)
                    if np.any(significant):
                        dirs = block_dir[significant]
                        mean_cos = float(np.mean(np.cos(dirs)))
                        mean_sin = float(np.mean(np.sin(dirs)))
                        circ_var = 1.0 - np.sqrt(mean_cos**2 + mean_sin**2)
                        coherence_map[r, c] = 1.0 - circ_var
                    else:
                        coherence_map[r, c] = 0.6
                else:
                    coherence_map[r, c] = 0.6  # bloco sem borda não acusa fraude

        coherence_n = cv2.normalize(coherence_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_density_n = cv2.normalize(edge_density_map, None, 0, 1, cv2.NORM_MINMAX)
        edge_naturalness = 0.6 * coherence_n + 0.4 * edge_density_n
        edge_score = int(np.clip(np.mean(edge_naturalness) * 100.0, 0, 100))
        return {"edge_score": edge_score}

    def analyze_image(self, image):
        res = self.analyze_edge_coherence(image)
        edge_score = res["edge_score"]

        if edge_score <= 35:
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


# ================================
# RUÍDO (COM CLAHE)
# ================================
class NoiseAnalyzer:
    """Análise de ruído COM CLAHE — consistência por blocos."""

    def __init__(self, block_size=32, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block_size = int(block_size)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_size = int(clahe_tile_size)

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
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRA_
