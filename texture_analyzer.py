# texture_analyzer.py
# MirrorGlass V4.3 – Análise Sequencial com Cadeia de Validação
# (c) 2025
#
# Principais pontos:
# - Conversão determinística RGB→GRAY (evita divergências BGR/RGB)
# - Detector primário de textura (LBP P=16/R=2) com contraste local
# - Máscara de detalhe estável (Laplaciano + std local)
# - Blocos sem detalhe dão nota neutra (não derrubam score)
# - Validadores: Bordas (tensor de estrutura), Ruído (MAD high-pass),
#   Iluminação (gradiente global), e Plano (regularidade de grandes regiões lisas)
# - Absolvedor para cena lisa quando validadores não acusam
# - Reponderação automática quando há plano grande (para-brisa, parede)

import cv2
import io
import base64
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.feature import local_binary_pattern


# ----------------------------- utilidades base ----------------------------- #

def _to_rgb(img_np_or_pil):
    """Força caminho único para RGB (elimina divergências BGR↔RGB)."""
    if isinstance(img_np_or_pil, Image.Image):
        return np.array(img_np_or_pil.convert("RGB"))
    arr = img_np_or_pil
    if arr.ndim == 2:  # GRAY -> RGB
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # BGR -> RGB


def _to_gray_uint8(img_np_or_pil):
    """RGB determinístico -> GRAY uint8 [0..255]."""
    rgb = _to_rgb(img_np_or_pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _safe_uint8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


# ----------------------------- máscara de detalhe -------------------------- #

def compute_detail_mask(gray_uint8, ksize_lap=3, win_std=7):
    """
    Máscara de áreas com detalhe (bordas/textura).
    Combina Laplaciano |∇²I| e desvio-padrão local. Limiariza por percentil
    com piso absoluto, e limpa com morfologia. Retorna (mask_bool, ratio).
    """
    g = _safe_uint8(gray_uint8)

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=ksize_lap)
    lap = np.abs(lap)

    mean = cv2.blur(g.astype(np.float32), (win_std, win_std))
    sqr  = cv2.blur((g.astype(np.float32) ** 2), (win_std, win_std))
    std  = np.sqrt(np.maximum(sqr - mean ** 2, 0.0))

    # Limiar por percentil + pisos absolutos (robusto a imagens lisas)
    lap_t = max(6.0, float(np.percentile(lap, 60)))
    std_t = max(5.0, float(np.percentile(std, 55)))

    m = (lap > lap_t) | (std > std_t)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = (m > 0)
    return mask, float(np.mean(mask))


# ----------------------------- analisador de textura ------------------------ #

class TextureAnalyzer:
    """
    Detector primário (SEM CLAHE).
    LBP (P=16/R=2) + entropia, variância, uniformidade e CONTRASTE local.
    Blocos sem detalhe são neutros (0.50) – não derrubam score.
    """

    def __init__(self, P=16, R=2, block_size=24, threshold=0.50):
        self.P = int(P)
        self.R = int(R)
        self.block = int(block_size)
        self.threshold = float(threshold)

    def _lbp(self, gray_uint8):
        lbp = local_binary_pattern(gray_uint8, self.P, self.R, method="uniform")
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        return lbp, hist

    def analyze_texture_variance(self, image_or_gray):
        gray = _to_gray_uint8(image_or_gray)
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

                # Peso de detalhe do bloco
                mblk = detail_mask[i:i+self.block, j:j+self.block]
                w_det = float(np.mean(mblk))

                # Bloco sem detalhe → nota neutra
                if w_det < 0.15:
                    variance_map[r, c] = 0.50
                    entropy_map [r, c] = 0.50
                    uniform_map [r, c] = 0.50
                    contrast_map[r, c] = 0.50
                    continue

                # Entropia de LBP (10 bins)
                h10, _ = np.histogram(blk, bins=10, range=(0, 10))
                h10 = h10.astype(np.float32) / (h10.sum() + 1e-7)
                ent  = entropy(h10)
                entropy_map[r, c] = float(ent / np.log(10))

                # Variância de LBP normalizada
                variance_map[r, c] = float(np.var(blk / float(self.P + 2)))

                # Uniformidade (pico alto penaliza)
                max_hist = float(np.max(h10))
                uniform_map[r, c] = 1.0 - max_hist

                # Contraste local (coeficiente de variação limitado)
                contr = np.std(gblk) / (np.mean(gblk) + 1e-6)
                contrast_map[r, c] = float(np.clip(contr / 0.25, 0.0, 1.0))

        # Combinação
        natural_map = (0.35 * variance_map +
                       0.35 * uniform_map  +
                       0.15 * entropy_map  +
                       0.15 * contrast_map)

        mean_nat = float(np.mean(natural_map))
        suspicious_mask = (natural_map < self.threshold)
        suspicious_ratio = float(np.mean(suspicious_mask))

        # penalização suave (não derruba em cena lisa)
        if suspicious_ratio <= 0.10:
            penalty = 1.0 - 0.6 * suspicious_ratio
        elif suspicious_ratio <= 0.25:
            penalty = 0.94 - 0.9 * (suspicious_ratio - 0.10)
        else:
            penalty = 0.805 - 1.2 * (suspicious_ratio - 0.25)
        penalty = float(np.clip(penalty, 0.55, 1.0))

        score = int(np.clip(mean_nat * penalty, 0, 1) * 100)

        # heatmap apenas visual
        vis_map  = cv2.resize(natural_map, (w, h), interpolation=cv2.INTER_LINEAR)
        vis_norm = cv2.normalize(vis_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap  = cv2.applyColorMap(_safe_uint8(vis_norm * 255), cv2.COLORMAP_JET)

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
        res   = self.analyze_texture_variance(image)
        score = res["naturalness_score"]
        cat, desc = self.classify(score)

        # overlay
        rgb = _to_rgb(image)
        h, w = rgb.shape[:2]
        mask = cv2.resize(res["suspicious_mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(
            _safe_uint8(cv2.normalize(cv2.resize(res["naturalness_map"], (w, h), cv2.INTER_LINEAR), None, 0, 1, cv2.NORM_MINMAX) * 255),
            cv2.COLORMAP_JET
        )
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
        self.enable = bool(enable)
        self.clip = float(clip)
        self.tiles = int(tiles)

    def apply(self, gray_uint8):
        if not self.enable:
            return gray_uint8
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tiles, self.tiles))
        return clahe.apply(_safe_uint8(gray_uint8))


class EdgeAnalyzer:
    """
    Bordas COM CLAHE – robusto a para-brisas/céus.
    - mede densidade só nas regiões com detalhe (máscara)
    - coerência via tensor de estrutura (mais estável)
    - continuidade (tamanho médio de contornos do Canny)
    - nitidez (razão de bordas fortes/medianas)
    - cenas com pouco detalhe retornam score neutro (~50)
    """

    def __init__(self, block_size=24, use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=8):
        self.block = int(block_size)
        self.clahe = _CLAHE(use_clahe, clahe_clip_limit, clahe_tile_size)

    def _structure_tensor_coherency(self, gx, gy, sigma=2.0):
        Jxx = cv2.GaussianBlur(gx*gx, (0, 0), sigma)
        Jyy = cv2.GaussianBlur(gy*gy, (0, 0), sigma)
        Jxy = cv2.GaussianBlur(gx*gy, (0, 0), sigma)
        tmp = np.sqrt(np.maximum((Jxx - Jyy)**2 + 4.0*Jxy*Jxy, 0.0))
        l1 = 0.5 * (Jxx + Jyy + tmp)
        l2 = 0.5 * (Jxx + Jyy - tmp)
        coherency = (l1 - l2) / (l1 + l2 + 1e-6)
        return np.clip(coherency, 0.0, 1.0)

    def analyze_image(self, image):
        gray = _to_gray_uint8(image)
        gray = self.clahe.apply(gray)

        detail_mask, detail_ratio = compute_detail_mask(gray)
        if detail_ratio < 0.18:
            return {
                "edge_score": 50,
                "category": "Bordas neutras",
                "description": "Pouco detalhe na cena",
                "clahe_enabled": self.clahe.enable
            }

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)

        # Canny com limiares automáticos
        v = float(np.median(gray))
        lo = int(max(0, (1.0 - 0.33) * v))
        hi = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(gray, lo, hi)

        m = detail_mask.astype(np.uint8)
        det_pix = int(np.count_nonzero(m))
        if det_pix == 0:
            dens = 0.5
        else:
            dens = float(np.count_nonzero(edges & m)) / float(det_pix)
        dens = float(np.clip(dens / 0.15, 0.0, 1.0))

        coher = self._structure_tensor_coherency(gx, gy)
        coher = float(np.mean(coher[detail_mask]))

        cont_img = edges & m
        cnts, _ = cv2.findContours(cont_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(cnts) == 0:
            cont_score = 0.0
        else:
            lengths = [cv2.arcLength(c, False) for c in cnts]
            mean_len = float(np.mean(lengths))
            diag = (gray.shape[0]**2 + gray.shape[1]**2) ** 0.5
            cont_score = float(np.clip(mean_len / (0.02 * diag), 0.0, 1.0))

        p70 = np.percentile(mag[detail_mask], 70)
        p90 = np.percentile(mag[detail_mask], 90)
        strong = float(np.mean(mag[detail_mask] > p90))
        medium = float(np.mean(mag[detail_mask] > p70)) + 1e-6
        sharp = float(np.clip(strong / medium, 0.0, 1.0))

        edge_nat = 0.45 * coher + 0.25 * cont_score + 0.20 * sharp + 0.10 * dens
        edge_score = int(np.clip(edge_nat, 0.0, 1.0) * 100)

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
    """
    Ruído SEM CLAHE.
    - trabalha em float32 [0,1]
    - ruído = MAD do resíduo high-pass (blur leve e subtração)
    - usa máscara de detalhe para ignorar vidro/céu lisos
    """

    def __init__(self, block_size=32, detail_grad_thresh=6.0, hp_sigma=1.0):
        self.block = int(block_size)
        self.grad_t = float(detail_grad_thresh)
        self.hp_sigma = float(hp_sigma)

    def _to_gray_float01(self, image):
        g = _to_gray_uint8(image).astype(np.float32) / 255.0
        return np.clip(g, 0.0, 1.0)

    def _detail_mask(self, gray01):
        gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        p90 = np.percentile(mag, 90.0)
        thr = max(self.grad_t / 255.0, 0.15 * p90)
        mask = (mag > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ratio = float(np.mean(mask))
        return mask.astype(bool), ratio

    def _sigma_map(self, gray01):
        blur = cv2.GaussianBlur(gray01, (0, 0), self.hp_sigma)
        resid = gray01 - blur
        H, W = gray01.shape
        rows = max(1, H // self.block)
        cols = max(1, W // self.block)
        sigma = np.zeros((rows, cols), np.float32)

        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                block = resid[i:i + self.block, j:j + self.block]
                med = np.median(block)
                mad = np.median(np.abs(block - med))
                sigma[i // self.block, j // self.block] = 1.4826 * mad
        return sigma

    def analyze_image(self, image):
        gray = self._to_gray_float01(image)
        dmask, dratio = self._detail_mask(gray)
        sigma_map = self._sigma_map(gray)

        H, W = gray.shape
        rows, cols = sigma_map.shape
        dmask_small = cv2.resize(dmask.astype(np.uint8), (cols, rows), interpolation=cv2.INTER_AREA) > 0

        valid = dmask_small.sum()
        if valid < max(4, int(0.05 * rows * cols)):
            return {
                "noise_score": 50,
                "category": "Ruído neutro",
                "description": "Cena com pouco detalhe — ruído não conclusivo",
                "clahe_enabled": False
            }

        sig_vals = sigma_map[dmask_small]
        sig_mean = float(np.mean(sig_vals))
        sig_std  = float(np.std(sig_vals))
        cv = sig_std / (sig_mean + 1e-8)

        present = np.clip((sig_mean - 0.004) / (0.02 - 0.004), 0.0, 1.0)
        too_much = np.clip((sig_mean - 0.04) / 0.02, 0.0, 1.0)

        consistency  = np.clip(1.0 - 1.2 * cv, 0.0, 1.0)
        plausibility = np.clip(present - 0.5 * too_much, 0.0, 1.0)

        noise_nat = 0.65 * consistency + 0.35 * plausibility
        noise_score = int(np.clip(noise_nat, 0.0, 1.0) * 100)

        if noise_score <= 40:
            cat, desc = "Ruído artificial", "Inconsistente/irreal (alisamento ou padrões não naturais)"
        elif noise_score <= 65:
            cat, desc = "Ruído inconsistente", "Distribuição de ruído suspeita"
        else:
            cat, desc = "Ruído natural", "Força e distribuição compatíveis com captura real"

        return {
            "noise_score": int(noise_score),
            "category": cat,
            "description": desc,
            "clahe_enabled": False
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

        smoothness = 1.0 / (np.std(mag) + 1.0)
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


class PlaneRegularityAnalyzer:
    """
    Fase 5 – Regularidade de Planos:
    Identifica grande região lisa (1 - detail_mask) e mede:
      * grad_std (baixa = liso artificial),
      * entropia de orientação (baixa = orientações pobres),
      * ruído high-pass (MAD) (muito baixo = IA).
    Combina em score 0..100 (alto = natural).
    """

    def __init__(self, min_plane_ratio=0.20, blur_sigma=1.0):
        self.min_plane_ratio = float(min_plane_ratio)
        self.blur_sigma = float(blur_sigma)

    def _to_gray01(self, image):
        return _to_gray_uint8(image).astype(np.float32) / 255.0

def _largest_smooth_region(self, gray01):
    # gray01 vem em float 0..1 — converte para uint8 só para máscaras
    gray_u8 = _safe_uint8(gray01 * 255)

    # máscara de detalhe já usada no pipeline
    detail_mask, _ = compute_detail_mask(gray_u8)

    # região lisa = complemento do detalhe
    smooth = (~detail_mask).astype(np.uint8) * 255

    # limpeza morfológica
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    # ✅ PASSA uint8 para connectedComponents (0/1), não bool
    bin_img = (smooth > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bin_img, connectivity=8, ltype=cv2.CV_32S)

    if num_labels <= 1:
        return None, 0.0

    # pega o maior componente (ignora rótulo 0 = fundo)
    areas = [(lab, int(np.sum(labels == lab))) for lab in range(1, num_labels)]
    lab_max, area_max = max(areas, key=lambda x: x[1])

    mask = (labels == lab_max)          # bool
    plane_ratio = float(area_max) / float(gray_u8.size)
    return mask, plane_ratio


    def analyze_image(self, image):
        gray = self._to_gray01(image)

        plane_mask, plane_ratio = self._largest_smooth_region(gray)
        if plane_mask is None or plane_ratio < self.min_plane_ratio:
            return {
                "plane_score": 50,
                "plane_ratio": float(plane_ratio),
                "category": "Plano neutro",
                "description": "Sem grande região lisa dominante"
            }

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy) + 1e-8
        ang = (np.arctan2(gy, gx) + np.pi)  # 0..2π

        m = plane_mask
        grad_std = float(np.std(mag[m]))

        bins = 12
        h, _ = np.histogram(ang[m], bins=bins, range=(0.0, 2.0*np.pi))
        h = h.astype(np.float32) / (h.sum() + 1e-7)
        orient_entropy = float(entropy(h) / np.log(bins))  # 0..1

        blur = cv2.GaussianBlur(gray, (0, 0), self.blur_sigma)
        resid = gray - blur
        med  = np.median(resid[m])
        mad  = np.median(np.abs(resid[m] - med))
        hp_sigma_mean = float(1.4826 * mad)

        # Normalizações (quanto MAIOR, melhor)
        grad_term   = np.clip(grad_std / 0.06, 0.0, 1.0)                     # <0.02 ruim; >0.06 ok
        orient_term = np.clip((orient_entropy - 0.35) / (0.80 - 0.35), 0.0, 1.0)
        hp_term     = np.clip((hp_sigma_mean - 0.004) / (0.02 - 0.004), 0.0, 1.0)

        plane_nat = 0.45*grad_term + 0.30*orient_term + 0.25*hp_term
        plane_score = int(np.clip(plane_nat, 0.0, 1.0) * 100)

        if plane_score <= 40:
            cat, desc = "Plano artificial", "Região lisa excessivamente estável"
        elif plane_score <= 65:
            cat, desc = "Plano suspeito", "Baixa variabilidade no interior do plano"
        else:
            cat, desc = "Plano natural", "Micro-variação e ruído plausíveis"

        return {
            "plane_score": int(plane_score),
            "plane_ratio": float(plane_ratio),
            "category": cat,
            "description": desc
        }


# ----------------------------- pipeline sequencial -------------------------- #

class SequentialAnalyzer:
    """
    Orquestra as fases. Regras:
    - Condena rápido se textura muito baixa e há detalhe suficiente.
    - Absolve em cena lisa quando validadores não acusam forte.
    - Caso médio pondera as fases (com re-peso se houver plano grande).
    """

    def __init__(self):
        self.texture_analyzer  = TextureAnalyzer()
        self.edge_analyzer     = EdgeAnalyzer(use_clahe=True)
        self.noise_analyzer    = NoiseAnalyzer()
        self.lighting_analyzer = LightingAnalyzer(use_clahe=True)
        self.plane_analyzer    = PlaneRegularityAnalyzer(min_plane_ratio=0.20, blur_sigma=1.0)

    def analyze_sequential(self, image):
        chain = []
        scores = {}

        # FASE 1 – Textura
        tex = self.texture_analyzer.analyze_image(image)
        tscore = tex["score"]
        detail_ratio = tex["analysis_results"]["detail_ratio"]
        chain.append("texture")
        scores["texture"] = tscore
        scores["detail_ratio"] = round(detail_ratio, 3)

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

        # FASE 5 – Regularidade de Planos
        plane = self.plane_analyzer.analyze_image(image)
        pscore = plane["plane_score"]
        pratio = plane["plane_ratio"]
        chain.append("plane")
        scores["plane"] = pscore
        scores["plane_ratio"] = round(pratio, 3)

        if pratio >= 0.30 and pscore < 45 and (escore < 40 or nscore < 45) and tscore < 65:
            return {
                "verdict": "MANIPULADA",
                "confidence": 88,
                "reason": "Plano grande excessivamente regular",
                "main_score": tscore,
                "all_scores": scores,
                "validation_chain": chain,
                "phases_executed": 5,
                "visual_report": tex["visual_report"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": tex["percent_suspicious"],
                "detailed_reason": f"plane={pscore} (ratio={int(pratio*100)}%), edge={escore}, noise={nscore}."
            }

        # ------------------------- ABSOLVER CENA LISA ------------------------- #
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
                    "main_score": int(0.38*tscore + 0.25*escore + 0.25*nscore + 0.12*lscore),
                    "all_scores": scores,
                    "validation_chain": chain,
                    "phases_executed": 5,
                    "visual_report": tex["visual_report"],
                    "heatmap": tex["heatmap"],
                    "percent_suspicious": tex["percent_suspicious"],
                    "detailed_reason": f"detail={int(detail_ratio*100)}%, edge={escore}, noise={nscore}, light={lscore}."
                }

        # ------------------------- PONDERAÇÃO FINAL -------------------------- #
        # Se há plano grande, aumente o peso de EDGE+PLANE e reduza TEXTURE
        if pratio >= 0.30:
            weighted = (0.40 * tscore + 0.25 * escore + 0.15 * nscore + 0.10 * lscore + 0.10 * pscore)
        else:
            weighted = (0.52 * tscore + 0.22 * escore + 0.14 * nscore + 0.07 * lscore + 0.05 * pscore)

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
            "phases_executed": 5,
            "visual_report": tex["visual_report"],
            "heatmap": tex["heatmap"],
            "percent_suspicious": tex["percent_suspicious"],
            "detailed_reason": f"Score ponderado={main}; detail={int(detail_ratio*100)}%, plane_ratio={int(pratio*100)}%."
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

