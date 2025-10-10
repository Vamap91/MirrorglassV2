# texture_analyzer.py
# MirrorGlass V4.2 – Análise Sequencial com Cadeia de Validação
# (c) 2025 – ajustes: conversão RGB->GRAY determinística, LBP P=16/R=2,
# máscara de detalhe estável, blocos neutros, contraste local, absolvedor.

import cv2
import io
import base64
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma


# ----------------------------- utilidades base ----------------------------- #

def _to_rgb(img_np_or_pil):
    """Força caminho único para RGB (elimina divergências BGR↔RGB)."""
    if isinstance(img_np_or_pil, Image.Image):
        return np.array(img_np_or_pil.convert("RGB"))
    arr = img_np_or_pil
    if arr.ndim == 2:                             # GRAY -> RGB
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)   # BGR -> RGB


def _to_gray_uint8(img_np_or_pil):
    """RGB determinístico -> GRAY uint8."""
    rgb = _to_rgb(img_np_or_pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _safe_uint8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


# ----------------------------- máscara de detalhe -------------------------- #

def compute_detail_mask(gray, ksize_lap=3, win_std=7):
    """
    Máscara de áreas com detalhe (bordas/textura). Combina Laplaciano e
    desvio-padrão local. Limiariza por percentil com piso absoluto.
    """
    gray = _safe_uint8(gray)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize_lap)
    lap = np.abs(lap)

    mean = cv2.blur(gray.astype(np.float32), (win_std, win_std))
    sqr  = cv2.blur((gray.astype(np.float32) ** 2), (win_std, win_std))
    std  = np.sqrt(np.maximum(sqr - mean ** 2, 0.0))

    # Limiar por percentil + pisos absolutos (robusto a imagens lisas)
    lap_t = max(6.0, float(np.percentile(lap, 60)))
    std_t = max(5.0, float(np.percentile(std, 55)))

    m = (lap > lap_t) | (std > std_t)
    # limpa ruído
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (m > 0), float(np.mean(m > 0))


# ----------------------------- analisador de textura ------------------------ #

class TextureAnalyzer:
    """
    Detector primário (SEM CLAHE).
    Usa LBP mais rico (P=16/R=2), entropia, variância, uniformidade e
    CONTRASTE local. Blocos sem detalhe são neutros (0.50) – não derrubam score.
    """

    def __init__(self, P=16, R=2, block_size=24, threshold=0.50):
        self.P = P
        self.R = R
        self.block = block_size
        self.threshold = threshold

    def _lbp(self, gray):
        lbp = local_binary_pattern(gray, self.P, self.R, method="uniform")
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        return lbp, hist

    def analyze_texture_variance(self, image):
        gray = _to_gray_uint8(image)
        detail_mask, detail_ratio = compute_detail_mask(gray)

        lbp_img, _ = self._lbp(gray)
        h, w = lbp_img.shape
        rows = max(1, h // self.block)
        cols = max(1, w // self.block)

        variance_map  = np.zeros((rows, cols), np.float32)
        entropy_map   = np.zeros((rows, cols), np.float32)
        uniform_map   = np.zeros((rows, cols), np.float32)
        contrast_map  = np.zeros((rows, cols), np.float32)

        for i in range(0, h - self.block + 1, self.block):
            for j in range(0, w - self.block + 1, self.block):
                r, c = i // self.block, j // self.block
                blk  = lbp_img[i:i+self.block, j:j+self.block]
                gblk = gray   [i:i+self.block, j:j+self.block].astype(np.float32)

                # Peso de detalhe do bloco (fração de pixels marcados)
                mblk = detail_mask[i:i+self.block, j:j+self.block]
                w_det = float(np.mean(mblk))

                # Bloco sem detalhe → nota neutra (0.50) em todos os termos
                if w_det < 0.15:
                    variance_map[r, c] = 0.50
                    entropy_map [r, c] = 0.50
                    uniform_map [r, c] = 0.50
                    contrast_map[r, c] = 0.50
                    continue

                # Entropia de LBP (10 bins) normalizada
                h10, _ = np.histogram(blk, bins=10, range=(0, 10))
                h10 = h10.astype(np.float32) / (h10.sum() + 1e-7)
                ent  = entropy(h10)
                entropy_map[r, c] = float(ent / np.log(10))

                # Variância de LBP normalizada (método uniform → 0..P+2)
                variance_map[r, c] = float(np.var(blk / float(self.P + 2)))

                # Uniformidade (pico de histograma alto é suspeito → penaliza)
                max_hist = float(np.max(h10))
                uniform_map[r, c] = 1.0 - max_hist

                # Contraste local (robustificado e recortado ~0..1)
                contr = np.std(gblk) / (np.mean(gblk) + 1e-6)
                contrast_map[r, c] = float(np.clip(contr / 0.25, 0.0, 1.0))

        # Combinação: variância + uniformidade dominam; contraste ajuda
        natural_map = (0.35 * variance_map +
                       0.35 * uniform_map  +
                       0.15 * entropy_map  +
                       0.15 * contrast_map)

        # score base
        mean_nat = float(np.mean(natural_map))
        suspicious_mask = (natural_map < self.threshold)
        suspicious_ratio = float(np.mean(suspicious_mask))

        # penalização suave (não deixa cair por cena lisa)
        if suspicious_ratio <= 0.10:
            penalty = 1.0 - 0.6 * suspicious_ratio
        elif suspicious_ratio <= 0.25:
            penalty = 0.94 - 0.9 * (suspicious_ratio - 0.10)
        else:
            penalty = 0.805 - 1.2 * (suspicious_ratio - 0.25)
        penalty = float(np.clip(penalty, 0.55, 1.0))

        score = int(np.clip(mean_nat * penalty, 0, 1) * 100)

        # heatmaps (apenas para visual)
        vis_map = cv2.resize(natural_map, (w, h), interpolation=cv2.INTER_LINEAR)
        vis_norm = cv2.normalize(vis_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(_safe_uint8(vis_norm * 255), cv2.COLORMAP_JET)

        return {
            "naturalness_map": natural_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": score,
            "heatmap": heatmap,
            "detail_ratio": detail_ratio
        }

    @staticmethod
    def classify(score):
        if score <= 45:
            return "Alta chance de manipulação", "Textura artificial detectada"
        if score <= 70:
            return "Textura suspeita", "Revisão manual sugerida"
        return "Textura natural", "Baixa chance de manipulação"

    def analyze_image(self, image):
        gray = _to_gray_uint8(image)
        res  = self.analyze_texture_variance(gray)
        score = res["naturalness_score"]
        cat, desc = self.classify(score)

        # overlay com contornos
        rgb = _to_rgb(image)
        h, w = rgb.shape[:2]
        mask = cv2.resize(res["suspicious_mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(_safe_uint8(cv2.normalize(cv2.resize(res["naturalness_map"], (w, h), cv2.INTER_LINEAR), None, 0, 1, cv2.NORM_MINMAX) * 255), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)
        cv2.putText(overlay, f"Score: {score}/100", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return {
            "score": score,
            "category": cat,
            "description": desc,
            "percent_suspicious": float(np.mean(res["suspicious_mask"]) * 100),
            "visual_report": overlay,
            "heatmap": res["heatmap"],
            "analysis_results": res,
            "clahe_enabled": False
        }


# ----------------------------- analisadores auxiliares ---------------------- #

class _CLAHE:
    def __init__(self, enable=True, clip=2.0, tiles=8):
        self.enable = enable
        self.clip = clip
        self.tiles = tiles

    def apply(self, gray):
        if not self.enable:
            return gray
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tiles, self.tiles))
        return clahe.apply(_safe_uint8(gray))


class EdgeAnalyzer:
    """
    Bordas COM CLAHE – robusto a para-brisas/céus.
    - mede densidade só nas regiões com detalhe (máscara)
    - coerência via tensor de estrutura (mais estável)
    - continuidade de borda (tamanho médio de contornos do Canny)
    - nitidez (razão de bordas fortes/medianas)
    - cenas com pouco detalhe retornam score neutro (~50)
    """

    def __init__(self, block_size=24, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block = block_size
        self.clahe = _CLAHE(use_clahe, clahe_clip_limit, clahe_tile_size)

    def _structure_tensor_coherency(self, gx, gy, sigma=2.0):
        # tensor de estrutura
        Jxx = cv2.GaussianBlur(gx*gx, (0, 0), sigma)
        Jyy = cv2.GaussianBlur(gy*gy, (0, 0), sigma)
        Jxy = cv2.GaussianBlur(gx*gy, (0, 0), sigma)

        # autovalores fechados (2x2)
        tmp = np.sqrt(np.maximum((Jxx - Jyy)**2 + 4.0*Jxy*Jxy, 0.0))
        l1 = 0.5 * (Jxx + Jyy + tmp)
        l2 = 0.5 * (Jxx + Jyy - tmp)
        coherency = (l1 - l2) / (l1 + l2 + 1e-6)  # 0..1
        return np.clip(coherency, 0.0, 1.0)

    def analyze_image(self, image):
        gray = _to_gray_uint8(image)
        gray = self.clahe.apply(gray)

        # máscara de detalhe – só avalia borda “onde importa”
        detail_mask, detail_ratio = compute_detail_mask(gray)

        # se não há detalhe suficiente, devolve neutro (não condena)
        if detail_ratio < 0.18:
            edge_score = 50
            cat, desc = "Bordas neutras", "Pouco detalhe na cena"
            return {
                "edge_score": edge_score,
                "category": cat,
                "description": desc,
                "clahe_enabled": self.clahe.enable
            }

        # gradientes
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)

        # Canny com limiares automáticos (mediana)
        v = float(np.median(gray))
        lo = int(max(0, (1.0 - 0.33) * v))
        hi = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(gray, lo, hi)

        # densidade de borda nas regiões com detalhe
        m = detail_mask.astype(np.uint8)
        det_pix = int(np.count_nonzero(m))
        if det_pix == 0:
            dens = 0.5
        else:
            dens = float(np.count_nonzero(edges & m)) / float(det_pix)  # 0..1
        dens = float(np.clip(dens / 0.15, 0.0, 1.0))  # normaliza ≈15% como “bom”

        # coerência via tensor de estrutura (média só no detalhe)
        coher = self._structure_tensor_coherency(gx, gy)
        coher = float(np.mean(coher[detail_mask]))  # 0..1

        # continuidade de borda: comprimento médio de contornos (normalizado)
        cont_img = edges & m
        cnts, _ = cv2.findContours(cont_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(cnts) == 0:
            cont_score = 0.0
        else:
            lengths = [cv2.arcLength(c, False) for c in cnts]
            mean_len = float(np.mean(lengths))
            diag = (gray.shape[0]**2 + gray.shape[1]**2) ** 0.5
            cont_score = float(np.clip(mean_len / (0.02 * diag), 0.0, 1.0))  # ~2% da diagonal é “ok”

        # nitidez: proporção de bordas fortes vs. medianas dentro do detalhe
        p70 = np.percentile(mag[detail_mask], 70)
        p90 = np.percentile(mag[detail_mask], 90)
        strong = float(np.mean(mag[detail_mask] > p90))
        medium = float(np.mean(mag[detail_mask] > p70)) + 1e-6
        sharp = float(np.clip(strong / medium, 0.0, 1.0))

        # combinação – coerência domina, continuidade/nitidez ajudam, densidade ajusta
        edge_nat = 0.45 * coher + 0.25 * cont_score + 0.20 * sharp + 0.10 * dens
        edge_score = int(np.clip(edge_nat, 0.0, 1.0) * 100)

        # rótulo
        if edge_score <= 35:
            cat, desc = "Bordas artificiais", "Alta probabilidade de manipulação"
        elif edge_score <= 60:
            cat, desc = "Bordas suspeitas", "Requer verificação"
        else:
            cat, desc = "Bordas naturais", "Baixa probabilidade de manipulação"

        return {
            "edge_score": edge_score,
            "category": cat,
            "description": desc,
            "clahe_enabled": self.clahe.enable
        }



class NoiseAnalyzer:
    """Ruído COM CLAHE – consistência por CV dos sigmas locais."""

    def __init__(self, block_size=32, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block = block_size
        self.clahe = _CLAHE(use_clahe, clahe_clip_limit, clahe_tile_size)

    def analyze_image(self, image):
        gray = _to_gray_uint8(image)
        gray = self.clahe.apply(gray)
        detail_mask, _ = compute_detail_mask(gray)

        h, w = gray.shape
        rows = max(1, h // self.block)
        cols = max(1, w // self.block)
        sigmas = []

        for i in range(0, h - self.block + 1, self.block):
            for j in range(0, w - self.block + 1, self.block):
                mblk = detail_mask[i:i+self.block, j:j+self.block]
                if float(np.mean(mblk)) < 0.10:
                    # bloco liso → neutro (entra como média do conjunto)
                    sigmas.append(0.0)
                    continue
                blk = gray[i:i+self.block, j:j+self.block]
                try:
                    sigma = float(estimate_sigma(blk, average_sigmas=True, channel_axis=None))
                except Exception:
                    sigma = float(np.std(blk))
                sigmas.append(sigma)

        if len(sigmas) == 0:
            noise_score = 50  # neutro
        else:
            arr = np.array(sigmas, dtype=np.float32)
            mean = float(np.mean(arr))
            std  = float(np.std(arr))
            cv   = (std / (mean + 1e-6)) if mean > 0 else 0.0
            noise_score = int(np.clip(1.0 - min(cv, 0.5) * 2.0, 0, 1) * 100)

        if noise_score <= 40:
            cat, desc = "Ruído artificial", "Alta probabilidade de manipulação"
        elif noise_score <= 65:
            cat, desc = "Ruído inconsistente", "Requer verificação"
        else:
            cat, desc = "Ruído natural", "Baixa probabilidade de manipulação"

        return {
            "noise_score": noise_score,
            "category": cat,
            "description": desc,
            "clahe_enabled": self.clahe.enable
        }


class LightingAnalyzer:
    """Iluminação COM CLAHE – suavidade + gradiente global."""

    def __init__(self, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.clahe = _CLAHE(use_clahe, clahe_clip_limit, clahe_tile_size)

    def analyze_image(self, image):
        gray = _to_gray_uint8(image)
        gray = self.clahe.apply(gray)

        gxs = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gys = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        mag = np.sqrt(gxs * gxs + gys * gys)

        # suavidade (quanto menor a variância do gradiente, maior o score)
        smoothness = 1.0 / (np.std(mag) + 1.0)  # 0..~1
        # gradiente global  (tenta capturar luz consistente)
        big = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
        gxb = cv2.Sobel(big, cv2.CV_32F, 1, 0, ksize=5)
        gyb = cv2.Sobel(big, cv2.CV_32F, 0, 1, ksize=5)
        magb = np.sqrt(gxb * gxb + gyb * gyb)
        global_grad = float(np.mean(magb) / 255.0)

        lighting_score = int(np.clip(0.6 * smoothness * 100 + 0.4 * (1 - global_grad) * 100, 0, 30))

        if lighting_score >= 20:
            cat, desc = "Iluminação natural", "Física consistente"
        elif lighting_score >= 10:
            cat, desc = "Iluminação aceitável", "Pequenas inconsistências"
        else:
            cat, desc = "Iluminação suspeita", "Inconsistências detectadas"

        return {
            "lighting_score": lighting_score,
            "category": cat,
            "description": desc,
            "clahe_enabled": self.clahe.enable
        }


# ----------------------------- pipeline sequencial -------------------------- #

class SequentialAnalyzer:
    """
    Orquestra as fases. Regras:
    - Condena rápido se textura muito baixa e há detalhe suficiente.
    - Absolve em cena lisa quando os validadores não acusam forte.
    - Caso médio pondera as fases.
    """

    def __init__(self):
        self.texture_analyzer  = TextureAnalyzer()
        self.edge_analyzer     = EdgeAnalyzer(use_clahe=True)
        self.noise_analyzer    = NoiseAnalyzer(use_clahe=True)
        self.lighting_analyzer = LightingAnalyzer(use_clahe=True)

    def analyze_sequential(self, image):
        chain = []
        scores = {}

        # FASE 1 – Textura (detector primário)
        tex = self.texture_analyzer.analyze_image(image)
        tscore = tex["score"]
        detail_ratio = tex["analysis_results"]["detail_ratio"]
        chain.append("texture")
        scores["texture"] = tscore
        scores["detail_ratio"] = round(detail_ratio, 3)

        # Condenação rápida: textura muito baixa e há detalhe suficiente
        if tscore < 37 and detail_ratio >= 0.25:
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial com áreas detalhadas",
                "main_score": tscore,
                "all_scores": scores,
                "validation_chain": chain,
                "phases_executed": 1,
                "visual_report": tex["visual_report"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": tex["percent_suspicious"],
                "detailed_reason": f"Score {tscore}/100; detalhe {int(detail_ratio*100)}%."
            }

        # Absolvição rápida por textura muito boa
        if tscore > 82:
            return {
                "verdict": "NATURAL",
                "confidence": 85,
                "reason": "Textura natural com alta variabilidade",
                "main_score": tscore,
                "all_scores": scores,
                "validation_chain": chain,
                "phases_executed": 1,
                "visual_report": tex["visual_report"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": tex["percent_suspicious"],
                "detailed_reason": f"Score {tscore}/100; detalhe {int(detail_ratio*100)}%."
            }

        # FASE 2 – Bordas
        edge = self.edge_analyzer.analyze_image(image)
        escore = edge["edge_score"]
        chain.append("edge")
        scores["edge"] = escore
        if escore < 35 and tscore < 60:
            return {
                "verdict": "MANIPULADA",
                "confidence": 90,
                "reason": "Textura duvidosa + bordas artificiais",
                "main_score": tscore,
                "all_scores": scores,
                "validation_chain": chain,
                "phases_executed": 2,
                "visual_report": tex["visual_report"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": tex["percent_suspicious"],
                "detailed_reason": f"texture={tscore}, edge={escore}."
            }

        # FASE 3 – Ruído
        noise = self.noise_analyzer.analyze_image(image)
        nscore = noise["noise_score"]
        chain.append("noise")
        scores["noise"] = nscore
        if nscore < 38 and tscore < 60:
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Textura suspeita + ruído artificial",
                "main_score": tscore,
                "all_scores": scores,
                "validation_chain": chain,
                "phases_executed": 3,
                "visual_report": tex["visual_report"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": tex["percent_suspicious"],
                "detailed_reason": f"texture={tscore}, noise={nscore}."
            }

        # FASE 4 – Iluminação
        light = self.lighting_analyzer.analyze_image(image)
        lscore = light["lighting_score"]
        chain.append("lighting")
        scores["lighting"] = lscore
        if lscore < 8 and tscore < 60:
            return {
                "verdict": "MANIPULADA",
                "confidence": 80,
                "reason": "Física de iluminação inconsistente",
                "main_score": tscore,
                "all_scores": scores,
                "validation_chain": chain,
                "phases_executed": 4,
                "visual_report": tex["visual_report"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": tex["percent_suspicious"],
                "detailed_reason": f"lighting={lscore}."
            }

        # ------------------------- ABSOLVER CENA LISA ------------------------- #
        # Parabrisa/ceu/parede: pouco detalhe → não condenar se validadores OK
        if detail_ratio < 0.25:
            ok = 0
            ok += 1 if escore >= 32 else 0
            ok += 1 if nscore >= 35 else 0
            ok += 1 if lscore >= 10 else 0
            if ok >= 2 and tscore >= 50:
                return {
                    "verdict": "NATURAL",
                    "confidence": 80,
                    "reason": "Cena com pouco detalhe; validadores coerentes",
                    "main_score": int(0.40*tscore + 0.25*escore + 0.25*nscore + 0.10*lscore),
                    "all_scores": scores,
                    "validation_chain": chain,
                    "phases_executed": 4,
                    "visual_report": tex["visual_report"],
                    "heatmap": tex["heatmap"],
                    "percent_suspicious": tex["percent_suspicious"],
                    "detailed_reason": f"detail={int(detail_ratio*100)}%, edge={escore}, noise={nscore}, light={lscore}."
                }

        # ------------------------- PONDERAÇÃO FINAL -------------------------- #
        weighted = (0.55 * tscore + 0.20 * escore + 0.15 * nscore + 0.10 * lscore)
        main = int(weighted)

        if main < 55:
            verdict, conf, reason = "SUSPEITA", 70, "Indicadores ambíguos"
        else:
            verdict, conf, reason = "INCONCLUSIVA", 60, "Revisão manual sugerida"

        return {
            "verdict": verdict,
            "confidence": conf,
            "reason": reason,
            "main_score": main,
            "all_scores": scores,
            "validation_chain": chain,
            "phases_executed": 4,
            "visual_report": tex["visual_report"],
            "heatmap": tex["heatmap"],
            "percent_suspicious": tex["percent_suspicious"],
            "detailed_reason": f"Score ponderado={main}; detail={int(detail_ratio*100)}%."
        }


# ----------------------------- util download -------------------------------- #

def get_image_download_link(img, filename, text):
    """Gera link de download (JPEG RGB)."""
    if isinstance(img, np.ndarray):
        rgb = _to_rgb(img)
        img_pil = Image.fromarray(rgb)
    else:
        img_pil = img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img)

    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=95)
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
