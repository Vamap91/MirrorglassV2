# texture_analyzer.py
# MirrorGlass V4.2 – Sequencial + Priors de Câmera
# (jan/2025) — foco em reduzir falso-positivo em fotos reais.

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy, pearsonr
from PIL import Image
import io
import base64


# =========================
# Utilidades de pré-processo
# =========================
def to_numpy_rgb(img):
    """Aceita PIL ou np.ndarray BGR/RGB e devolve np.uint8 RGB."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    else:
        arr = img.copy()
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:  # pode vir BGR
            # heurística: se média do canal 0 for ~azul forte, é BGR
            # mas é barato sempre converter assumindo BGR:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def safe_resize_long_side(img_rgb, long_side=1024):
    """Redimensiona preservando aspecto. Usa INTER_AREA ao reduzir."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= long_side:
        return img_rgb
    if h >= w:
        new_h, new_w = long_side, int(w * (long_side / h))
    else:
        new_w, new_h = long_side, int(h * (long_side / w))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rgb2gray_u8(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


# =========================
# Priors (sinais típicos de foto)
# =========================
class PhotoPriors:
    """
    Priors que aparecem em fotos reais:
      • Desfoque/Laplaciano moderado (não zero).
      • Blocagem/periódico 8×8 (JPEG).
      • Correlação ruído × luminância (mais ruído em áreas escuras).
    Usamos esses sinais como 'absolvedores' quando os validadores não indicam IA forte.
    """

    def __init__(self):
        pass

    @staticmethod
    def blur_laplacian(gray):
        # Variância do Laplaciano – foco: 0..~3000 (depende do tamanho)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def jpeg_blockiness(gray):
        """
        Mede diferença média em fronteiras de blocos 8x8.
        Valores típicos:
          ~0.5–1.5 (uint8) em JPEGs comuns 1080p reamostrados,
          ~0.1–0.3 em PNG/sem blocagem.
        Retorna valor normalizado 0..1 (aprox).
        """
        g = gray.astype(np.float32)
        # diferenças nas colunas/linhas múltiplas de 8
        dif_c = []
        for c in range(8, g.shape[1], 8):
            dif_c.append(np.abs(g[:, c] - g[:, c - 1]).mean())
        dif_r = []
        for r in range(8, g.shape[0], 8):
            dif_r.append(np.abs(g[r, :] - g[r - 1, :]).mean())
        if not dif_c and not dif_r:
            return 0.0
        raw = (np.mean(dif_c) if dif_c else 0.0 + np.mean(dif_r) if dif_r else 0.0) / (2 if dif_c and dif_r else 1)
        # normalização suave
        return float(np.clip(raw / 2.0, 0.0, 1.0))

    @staticmethod
    def noise_luma_correlation(img_rgb):
        """
        Correlaciona ruído de alta frequência com luminância.
        Em fotos reais: correlação tende a ser NEGATIVA (regiões escuras com mais ruído).
        Retorna correlação (−1..+1). Valores <= −0.2 favorecem 'foto'.
        """
        gray = rgb2gray_u8(img_rgb).astype(np.float32)
        # high-pass simples
        hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
        # luminância (0..1)
        luma = gray / 255.0
        # variância local de ruído (janela 7x7)
        hp2 = hp * hp
        var_local = cv2.GaussianBlur(hp2, (0, 0), 2.0)
        # correlaciona var_local com luminância
        v = var_local.flatten()
        l = luma.flatten()
        # evita NaNs
        if np.std(v) < 1e-6 or np.std(l) < 1e-6:
            return 0.0
        corr = pearsonr(v, l)[0]
        return float(corr)

    def compute(self, img_rgb):
        gray = rgb2gray_u8(img_rgb)
        blur = self.blur_laplacian(gray)
        block = self.jpeg_blockiness(gray)
        corr = self.noise_luma_correlation(img_rgb)
        return {
            "blur_var_laplacian": blur,
            "jpeg_blockiness": block,
            "noise_luma_corr": corr,  # NEGATIVO favorece foto
        }


# =========================
# Analise de TEXTURA (LBP)
# =========================
class TextureAnalyzer:
    """Detector primário – LBP multiescala SEM CLAHE."""
    def __init__(self, block_size=24, threshold=0.50):
        self.block = block_size
        self.threshold = threshold
        # multiescala: (P,R)
        self.scales = [(8, 1), (16, 2)]

    def _lbp_hist(self, gray, P, R):
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float32") / (hist.sum() + 1e-7)
        return lbp, hist

    def analyze_texture_variance(self, img_rgb):
        gray = rgb2gray_u8(img_rgb)
        H, W = gray.shape

        # concatena mapas por escala
        maps = []
        for (P, R) in self.scales:
            lbp, _ = self._lbp_hist(gray, P, R)
            maps.append(lbp.astype(np.float32) / (P + 2))
        lbp_stack = np.mean(maps, axis=0)  # média das escalas

        rows = max(1, H // self.block)
        cols = max(1, W // self.block)
        variance_map = np.zeros((rows, cols), np.float32)
        entropy_map = np.zeros((rows, cols), np.float32)
        uniformity_map = np.zeros((rows, cols), np.float32)

        # usa 12 bins na entropia para mais resolução
        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                block = lbp_stack[i:i+self.block, j:j+self.block]
                hist, _ = np.histogram(block, bins=12, range=(0, 1.0))
                hist = hist.astype("float32") / (hist.sum() + 1e-7)
                e = entropy(hist)
                max_e = np.log(12.0)
                norm_e = (e / max_e) if max_e > 0 else 0.0

                v = float(np.var(block))
                # penaliza “um pico só” no hist
                uniformity_pen = 1.0 - float(np.max(hist))

                r, c = i // self.block, j // self.block
                variance_map[r, c] = v
                entropy_map[r, c] = norm_e
                uniformity_map[r, c] = uniformity_pen

        # pesos equilibrados
        naturalness_map = 0.35 * entropy_map + 0.40 * variance_map + 0.25 * uniformity_map

        suspicious_mask = naturalness_map < self.threshold
        suspicious_ratio = float(np.mean(suspicious_mask))
        mean_nat = float(np.mean(naturalness_map))

        # penalização MAIS suave (evita FP)
        penalty = 1.0 - 0.9 * suspicious_ratio
        penalty = float(np.clip(penalty, 0.6, 1.0))

        score = int(np.clip(mean_nat * penalty * 100.0, 0, 100))

        # mapas visuais
        disp = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return {
            "naturalness_map": naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": score,
            "heatmap": heatmap,
            "suspicious_ratio": suspicious_ratio,
            "mean_naturalness_raw": mean_nat,
        }

    @staticmethod
    def classify(score):
        if score < 38:
            return "Alta chance de manipulação", "Textura fortemente artificial"
        elif score < 62:
            return "Textura intermediária", "Sinais mistos"
        else:
            return "Textura natural", "Alta variabilidade"


# =========================
# BORDAS, RUÍDO, ILUMINAÇÃO
# =========================
class EdgeAnalyzer:
    def __init__(self, block_size=24, use_clahe=True, clahe_clip=2.0, clahe_tile=8):
        self.block = block_size
        self.use_clahe = use_clahe
        self.clip = clahe_clip
        self.tile = clahe_tile

    def _gray(self, img_rgb):
        g = rgb2gray_u8(img_rgb)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile))
            g = clahe.apply(g)
        return g

    def analyze(self, img_rgb):
        gray = self._gray(img_rgb)
        H, W = gray.shape
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        ang = np.arctan2(gy, gx)

        rows, cols = max(1, H // self.block), max(1, W // self.block)
        coh = np.zeros((rows, cols), np.float32)
        den = np.zeros((rows, cols), np.float32)

        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                m = mag[i:i+self.block, j:j+self.block]
                a = ang[i:i+self.block, j:j+self.block]
                r, c = i // self.block, j // self.block
                den[r, c] = float(np.mean(m)) / 255.0
                # coerência direcional
                m_th = m > np.percentile(m, 70)
                if np.any(m_th):
                    aa = a[m_th]
                    coh[r, c] = 1.0 - (1.0 - np.sqrt(np.mean(np.cos(aa))**2 + np.mean(np.sin(aa))**2))
                else:
                    coh[r, c] = 0.5

        den = cv2.normalize(den, None, 0, 1, cv2.NORM_MINMAX)
        coh = cv2.normalize(coh, None, 0, 1, cv2.NORM_MINMAX)
        edge_nat = 0.55 * coh + 0.45 * den
        score = int(np.mean(edge_nat) * 100.0)
        return {"edge_score": score}


class NoiseAnalyzer:
    def __init__(self, block_size=32, use_clahe=True, clahe_clip=2.0, clahe_tile=8):
        self.block = block_size
        self.use_clahe = use_clahe
        self.clip = clahe_clip
        self.tile = clahe_tile

    def _gray(self, img_rgb):
        g = rgb2gray_u8(img_rgb)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile))
            g = clahe.apply(g)
        return g

    def analyze(self, img_rgb):
        gray = self._gray(img_rgb)
        H, W = gray.shape
        rows, cols = max(1, H // self.block), max(1, W // self.block)
        nmap = np.zeros((rows, cols), np.float32)

        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                blk = gray[i:i+self.block, j:j+self.block]
                r, c = i // self.block, j // self.block
                try:
                    sigma = estimate_sigma(blk, average_sigmas=True, channel_axis=None)
                except Exception:
                    sigma = float(np.std(blk))
                nmap[r, c] = sigma

        mu, sd = float(np.mean(nmap)), float(np.std(nmap))
        cv = sd / (mu + 1e-6)
        # maior consistência => maior score
        score = int(np.clip(100.0 - 220.0 * cv, 0, 100))
        return {"noise_score": score}


class LightingAnalyzer:
    def __init__(self, use_clahe=True, clahe_clip=2.0, clahe_tile=8):
        self.use_clahe = use_clahe
        self.clip = clahe_clip
        self.tile = clahe_tile

    def _gray(self, img_rgb):
        g = rgb2gray_u8(img_rgb)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile))
            g = clahe.apply(g)
        return g

    def analyze(self, img_rgb):
        gray = self._gray(img_rgb)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        mag = np.sqrt(gx*gx + gy*gy)
        smoothness = 1.0 / (np.std(mag) + 1.0)
        score = int(min(smoothness * 55.0, 35.0))  # 0..35
        return {"lighting_score": score}


# =========================
# Sequencial + Arbitragem
# =========================
class SequentialAnalyzer:
    def __init__(self, long_side_px=1024):
        self.texture = TextureAnalyzer()
        self.edge = EdgeAnalyzer()
        self.noise = NoiseAnalyzer()
        self.light = LightingAnalyzer()
        self.priors = PhotoPriors()
        self.long_side = long_side_px

    def analyze_sequential(self, image):
        # ---------- pré-processo padronizado ----------
        rgb = to_numpy_rgb(image)
        rgb = safe_resize_long_side(rgb, self.long_side)

        # ---------- priors de câmera ----------
        p = self.priors.compute(rgb)
        blur_ok = p["blur_var_laplacian"] >= 40.0          # desfoque muito baixo => imagem “plana”
        jpeg_ok = p["jpeg_blockiness"] >= 0.12             # blocagem visível sugere foto
        corr_ok = p["noise_luma_corr"] <= -0.20            # ruído ↑ em áreas escuras
        photo_prior_votes = sum([blur_ok, jpeg_ok, corr_ok])

        # ---------- FASE 1: Textura ----------
        t_res = self.texture.analyze_texture_variance(rgb)
        t_score = t_res["naturalness_score"]
        vis_img, heat = self._visual(rgb, t_res)

        all_scores = {"texture": t_score}
        chain = ["texture"]

        # Só condena na Fase 1 se MUITO baixo e sem priors fortes de foto
        if t_score < 34 and photo_prior_votes == 0:
            return self._pack(
                verdict="MANIPULADA",
                confidence=90,
                reason="Textura fortemente artificial e ausência de sinais típicos de foto",
                main_score=t_score, all_scores=all_scores, chain=chain,
                vis=vis_img, heat=heat, t_res=t_res
            )

        # Se a textura já é confortável e há pelo menos 2 priors de foto -> NATURAL direto
        if t_score >= 62 and photo_prior_votes >= 2:
            return self._pack(
                verdict="NATURAL",
                confidence=85,
                reason="Textura natural com múltiplos sinais de captura fotográfica",
                main_score=t_score, all_scores=all_scores, chain=chain,
                vis=vis_img, heat=heat, t_res=t_res
            )

        # ---------- FASE 2: Bordas ----------
        e = self.edge.analyze(rgb)["edge_score"]
        all_scores["edge"] = e
        chain.append("edge")
        if e < 35 and t_score < 50 and photo_prior_votes == 0:
            return self._pack("MANIPULADA", 88,
                              "Textura fraca e bordas artificiais",
                              t_score, all_scores, chain, vis_img, heat, t_res)

        # ---------- FASE 3: Ruído ----------
        n = self.noise.analyze(rgb)["noise_score"]
        all_scores["noise"] = n
        chain.append("noise")
        if n < 35 and t_score < 50 and photo_prior_votes == 0:
            return self._pack("MANIPULADA", 85,
                              "Textura fraca e ruído inconsistente",
                              t_score, all_scores, chain, vis_img, heat, t_res)

        # ---------- FASE 4: Iluminação ----------
        l = self.light.analyze(rgb)["lighting_score"]
        all_scores["lighting"] = l
        chain.append("lighting")
        if l < 8 and t_score < 50 and photo_prior_votes == 0:
            return self._pack("MANIPULADA", 80,
                              "Física de iluminação inconsistente",
                              t_score, all_scores, chain, vis_img, heat, t_res)

        # ---------- Absolvedor de maioria (quando textura é média) ----------
        good = (1 if e >= 68 else 0) + (1 if n >= 68 else 0) + (1 if l >= 20 else 0) + (1 if photo_prior_votes >= 2 else 0)
        if 45 <= t_score <= 62 and good >= 2:
            return self._pack(
                verdict="NATURAL",
                confidence=80,
                reason="Textura intermediária, mas bordas/ruído/iluminação e/ou priors fotográficos consistentes",
                main_score=int(0.30*t_score + 0.30*e + 0.25*n + 0.15*l),
                all_scores=all_scores, chain=chain, vis=vis_img, heat=heat, t_res=t_res
            )

        # ---------- Score final ponderado ----------
        weighted = 0.45 * t_score + 0.23 * e + 0.20 * n + 0.12 * l
        if weighted >= 60 or photo_prior_votes >= 2:
            verdict, conf, reason = "NATURAL", 75, "Conjunto de indícios favorece foto/autenticidade"
        elif weighted < 50 and photo_prior_votes == 0:
            verdict, conf, reason = "SUSPEITA", 70, "Indicadores mistos com sinais fracos de foto"
        else:
            verdict, conf, reason = "INCONCLUSIVA", 60, "Revisão manual sugerida"

        return self._pack(
            verdict=verdict, confidence=conf, reason=reason,
            main_score=int(weighted), all_scores=all_scores, chain=chain,
            vis=vis_img, heat=heat, t_res=t_res
        )

    # -------- helpers de visual/empacote --------
    @staticmethod
    def _visual(img_rgb, t_res):
        nat = t_res["naturalness_map"]
        H, W = img_rgb.shape[:2]
        nat = cv2.resize(nat, (W, H), interpolation=cv2.INTER_LINEAR)
        disp = cv2.normalize(nat, None, 0, 1, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heat, 0.4, 0)
        # texto
        out = overlay.copy()
        score = t_res["naturalness_score"]
        cat, _ = TextureAnalyzer.classify(score)
        cv2.putText(out, f"Score: {score}/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(out, cat, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return out, heat

    @staticmethod
    def _pack(verdict, confidence, reason, main_score, all_scores, chain, vis, heat, t_res):
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": int(main_score),
            "all_scores": all_scores,
            "validation_chain": chain,
            "phases_executed": len(chain),
            "visual_report": vis,
            "heatmap": heat,
            "percent_suspicious": float(np.mean(t_res["suspicious_mask"]) * 100.0),
            "detailed_reason": f"Texture score={t_res['naturalness_score']} | suspicious_ratio={t_res['suspicious_ratio']:.2f}",
        }


# (mantive util para download, caso use)
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
