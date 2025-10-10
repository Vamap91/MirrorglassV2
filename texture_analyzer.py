# texture_analyzer.py
# Sistema de Análise Sequencial com Validação em Cadeia
# Versão: 4.2.0 - Ajustes de robustez (multi-escala, threshold adaptativo, edge-gated penalty)

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import io
import base64


# -----------------------------
# Textura (SEM CLAHE)
# -----------------------------
class TextureAnalyzer:
    """Análise de texturas com LBP multi-escala (SEM CLAHE) — Detector primário."""

    def __init__(self, block_size=24, base_threshold=0.50):
        # block_size: tamanhos de blocos para estatísticas por região
        # base_threshold: limiar base do mapa de 'naturalidade' por bloco (será adaptativo)
        self.block_size = int(block_size)
        self.base_threshold = float(base_threshold)

    def _ensure_gray_uint8(self, image):
        if isinstance(image, Image.Image):
            img = np.array(image.convert("L"))
        elif image.ndim == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()

        # normaliza tamanho para estabilidade estatística (reduz variância por resolução)
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
        LBP em duas escalas (R=1, P=8) e (R=2, P=16).
        Retorna um mapa combinado contínuo em [0, 1], enfatizando macrotextura.
        """
        gray = self._ensure_gray_uint8(image)

        lbp1 = local_binary_pattern(gray, 8, 1, method="uniform")     # 0..10
        lbp2 = local_binary_pattern(gray, 16, 2, method="uniform")    # 0..18

        # normaliza cada LBP para [0,1] e combina (peso maior p/ macro)
        lbp1_n = lbp1 / float(8 + 2)
        lbp2_n = lbp2 / float(16 + 2)
        lbp_combined = np.clip(0.4 * lbp1_n + 0.6 * lbp2_n, 0.0, 1.0)
        return lbp_combined, gray

    def analyze_texture_variance(self, image):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        lbp_norm, gray = self.calculate_lbp_multiscale(image_np)  # lbp_norm ∈ [0,1]
        height, width = lbp_norm.shape

        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)

        variance_map   = np.zeros((rows, cols), dtype=np.float32)
        entropy_map    = np.zeros((rows, cols), dtype=np.float32)
        uniformity_map = np.zeros((rows, cols), dtype=np.float32)
        edge_map_local = np.zeros((rows, cols), dtype=np.float32)

        # mapa de bordas global (p/ gate da penalidade)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

        # detalhe global p/ threshold adaptativo
        detail_level = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        thr = self.base_threshold
        if detail_level < 40:     # cena lisa (lataria, teto, parede, céu)
            thr = 0.40
        elif detail_level > 150:  # cena muito detalhada
            thr = 0.55

        # percorre blocos
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                r = i // self.block_size
                c = j // self.block_size

                block = lbp_norm[i:i+self.block_size, j:j+self.block_size]

                # histograma em [0,1] (10 bins) → entropia normalizada (0..1)
                hist, _ = np.histogram(block, bins=10, range=(0, 1))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
                e = float(entropy(hist) / np.log(10))

                # variância já em [0,1]
                v = float(np.var(block))

                # uniformidade: 1 - pico do histograma
                u = float(1.0 - np.max(hist))

                # força de borda local (média do gradiente bruto — robusto por percentil depois)
                block_mag = mag[i:i+self.block_size, j:j+self.block_size]
                edge_map_local[r, c] = float(np.mean(block_mag))

                variance_map[r, c]   = v
                entropy_map[r, c]    = e
                uniformity_map[r, c] = u

        # normaliza a força de borda por percentil (robusto a outliers)
        p90 = np.percentile(edge_map_local, 90) + 1e-6
        edge_strength = np.clip(edge_map_local / p90, 0, 1)  # 0..1

        # composição do mapa de naturalidade (menos peso para "uniformidade")
        naturalness_map = (0.50 * variance_map + 0.30 * entropy_map + 0.20 * uniformity_map)

        # máscara suspeita com threshold adaptativo
        suspicious_mask = (naturalness_map < thr).astype(np.float32)

        # GATE por borda: blocos com borda forte contam menos na penalidade
        gate = (1.0 - 0.6 * edge_strength)  # 0.4..1.0 (borda forte reduz peso)
        gated_suspicious = suspicious_mask * gate

        mean_naturalness = float(np.mean(naturalness_map))
        suspicious_ratio = float(np.mean(gated_suspicious))

        # penalidade suave — nunca < 0.55 (evita derrubar fotos reais lisas)
        if suspicious_ratio <= 0.10:
            penalty_factor = 1.0 - 0.7 * suspicious_ratio
        elif suspicious_ratio <= 0.25:
            penalty_factor = 0.93 - 1.0 * (suspicious_ratio - 0.10)
        else:
            penalty_factor = 0.78 - 1.3 * (suspicious_ratio - 0.25)
        penalty_factor = float(np.clip(penalty_factor, 0.55, 1.0))

        naturalness_score = int(np.clip(mean_naturalness * penalty_factor * 100.0, 0, 100))

        # heatmap apenas para visualização
        disp = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return {
            "variance_map": variance_map,
            "naturalness_map": naturalness_map,
            "suspicious_mask": suspicious_mask.astype(bool),
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio,
            "mean_naturalness_raw": mean_naturalness
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


# -----------------------------
# Bordas (COM CLAHE)
# -----------------------------
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

                edge_density_map[r, c] = float(np.mean(block_mag))  # sem /255 (normaliza depois)

                # somente se houver borda suficiente, mede coerência; senão, bloco "neutro-bom"
                if np.sum(block_mag > np.percentile(block_mag, 60)) > 8:
                    significant = block_mag > np.percentile(block_mag, 70)
                    if np.any(significant):
                        dirs = block_dir[significant]
                        mean_cos = float(np.mean(np.cos(dirs)))
                        mean_sin = float(np.mean(np.sin(dirs)))
                        circ_var = 1.0 - np.sqrt(mean_cos**2 + mean_sin**2)
                        coherence_map[r, c] = 1.0 - circ_var  # 0..1 (1 = coerente)
                    else:
                        coherence_map[r, c] = 0.6
                else:
                    coherence_map[r, c] = 0.6  # bloco sem borda não é evidência de fraude

        # normalização robusta
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


# -----------------------------
# Ruído (COM CLAHE)
# -----------------------------
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.apply_clahe(gray)

    def analyze_local_noise(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        gray = self._convert_to_gray(image)
        h, w = gray.shape

        rows = max(1, h // self.block_size)
        cols = max(1, w // self.block_size)
        noise_map = np.zeros((rows, cols), dtype=np.float32)

        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                r = i // self.block_size
                c = j // self.block_size
                block = gray[i:i+self.block_size, j:j+self.block_size]

                try:
                    sigma = float(estimate_sigma(block, average_sigmas=True, channel_axis=None))
                except Exception:
                    sigma = float(np.std(block))
                noise_map[r, c] = sigma

        noise_mean = float(np.mean(noise_map))
        noise_std = float(np.std(noise_map))
        noise_cv = noise_std / noise_mean if noise_mean > 0 else 0.0

        # penalização menos agressiva (clamp 0..100)
        noise_consistency_score = int(np.clip(100.0 - 160.0 * noise_cv, 0, 100))
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


# -----------------------------
# Iluminação (COM CLAHE)
# -----------------------------
class LightingAnalyzer:
    """Análise simples de iluminação COM CLAHE (gradiente global)."""

    def __init__(self, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
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

    def analyze_image(self, image):
        gray = self._convert_to_gray(image)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gx**2 + gy**2)

        smoothness = 1.0 / (np.std(magnitude) + 1.0)
        lighting_score = int(min(smoothness * 50.0, 30.0))

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


# -----------------------------
# Sequencial (validação em cadeia)
# -----------------------------
class SequentialAnalyzer:
    """Sistema de Análise Sequencial - Validação em Cadeia (early-exit + absolvedor)."""

    def __init__(self):
        self.texture_analyzer  = TextureAnalyzer()
        self.edge_analyzer     = EdgeAnalyzer(use_clahe=True)
        self.noise_analyzer    = NoiseAnalyzer(use_clahe=True)
        self.lighting_analyzer = LightingAnalyzer(use_clahe=True)

    def analyze_sequential(self, image):
        validation_chain = []
        all_scores = {}

        # FASE 1 — TEXTURA
        tex = self.texture_analyzer.analyze_image(image)
        t_score = tex['score']
        all_scores['texture'] = t_score
        validation_chain.append('texture')

        if t_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura muito artificial",
                "main_score": t_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": tex['visual_report'],
                "heatmap": tex['heatmap'],
                "percent_suspicious": tex['percent_suspicious'],
                "detailed_reason": f"Score {t_score}/100 abaixo do limiar."
            }

        if t_score > 80:
            return {
                "verdict": "NATURAL",
                "confidence": 85,
                "reason": "Textura natural com alta variabilidade",
                "main_score": t_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 1,
                "visual_report": tex['visual_report'],
                "heatmap": tex['heatmap'],
                "percent_suspicious": tex['percent_suspicious'],
                "detailed_reason": f"Score {t_score}/100 elevado."
            }

        # FASE 2 — BORDAS
        edg = self.edge_analyzer.analyze_image(image)
        e_score = edg['edge_score']
        all_scores['edge'] = e_score
        validation_chain.append('edge')

        if e_score < 35:
            return {
                "verdict": "MANIPULADA",
                "confidence": 90,
                "reason": "Textura duvidosa + bordas artificiais",
                "main_score": t_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 2,
                "visual_report": tex['visual_report'],
                "heatmap": tex['heatmap'],
                "percent_suspicious": tex['percent_suspicious'],
                "detailed_reason": f"Textura {t_score}/100 confirmada por bordas {e_score}/100."
            }

        # FASE 3 — RUÍDO
        noi = self.noise_analyzer.analyze_image(image)
        n_score = noi['noise_score']
        all_scores['noise'] = n_score
        validation_chain.append('noise')

        if n_score < 40:
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Múltiplos indicadores artificiais (ruído inconsistente)",
                "main_score": t_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 3,
                "visual_report": tex['visual_report'],
                "heatmap": tex['heatmap'],
                "percent_suspicious": tex['percent_suspicious'],
                "detailed_reason": f"Textura {t_score}/100 + ruído {n_score}/100."
            }

        # FASE 4 — ILUMINAÇÃO
        lig = self.lighting_analyzer.analyze_image(image)
        l_score = lig['lighting_score']
        all_scores['lighting'] = l_score
        validation_chain.append('lighting')

        if l_score < 10:
            return {
                "verdict": "MANIPULADA",
                "confidence": 80,
                "reason": "Física de iluminação impossível",
                "main_score": t_score,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 4,
                "visual_report": tex['visual_report'],
                "heatmap": tex['heatmap'],
                "percent_suspicious": tex['percent_suspicious'],
                "detailed_reason": f"Iluminação {l_score}/30 inconsistente."
            }

        # ABSOLVEDOR — maioria dos validadores positivos com textura intermediária
        good = 0
        good += 1 if e_score >= 68 else 0
        good += 1 if n_score >= 68 else 0
        good += 1 if l_score >= 18 else 0

        if 50 <= t_score <= 80 and good >= 2:
            main = int(t_score * 0.35 + e_score * 0.30 + n_score * 0.25 + l_score * 0.10)
            return {
                "verdict": "NATURAL",
                "confidence": 80,
                "reason": "Textura mediana, mas bordas/ruído/iluminação consistentes",
                "main_score": main,
                "all_scores": all_scores,
                "validation_chain": validation_chain,
                "phases_executed": 4,
                "visual_report": tex['visual_report'],
                "heatmap": tex['heatmap'],
                "percent_suspicious": tex['percent_suspicious'],
                "detailed_reason": f"Absolvido por maioria: edge={e_score}, noise={n_score}, lighting={l_score}, texture={t_score}."
            }

        # CASO FINAL — ponderado
        weighted = t_score * 0.50 + e_score * 0.25 + n_score * 0.15 + l_score * 0.10
        if weighted < 55:
            verdict, confidence, reason = "SUSPEITA", 70, "Indicadores ambíguos"
        else:
            verdict, confidence, reason = "INCONCLUSIVA", 60, "Revisão manual necessária"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": int(weighted),
            "all_scores": all_scores,
            "validation_chain": validation_chain,
            "phases_executed": 4,
            "visual_report": tex['visual_report'],
            "heatmap": tex['heatmap'],
            "percent_suspicious": tex['percent_suspicious'],
            "detailed_reason": f"Score ponderado: {int(weighted)}/100."
        }


# Utilitário de download (mantido)
def get_image_download_link(img, filename, text):
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
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
