# texture_analyzer.py
# MirrorGlass V4.1 — Análise Sequencial com Gatekeeper de Textura + Validadores com Absolvição
# Revisão: 2025-10 — foco em estabilidade, thresholds calibrados e transparência por fase.

import cv2
import io
import base64
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma


# ------------------------------ utilidades ------------------------------

def _to_np_rgb(img):
    """PIL -> np(uint8 RGB) | np BGR/GRAY -> np RGB"""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = img.copy()
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    # BGR -> RGB
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

def _to_np_gray(img_rgb_or_bgr):
    if isinstance(img_rgb_or_bgr, Image.Image):
        return np.array(img_rgb_or_bgr.convert("L"))
    if img_rgb_or_bgr.ndim == 2:
        return img_rgb_or_bgr.copy()
    # assume BGR
    return cv2.cvtColor(img_rgb_or_bgr, cv2.COLOR_BGR2GRAY) if img_rgb_or_bgr.shape[2] == 3 and img_rgb_or_bgr.flags['C_CONTIGUOUS'] else cv2.cvtColor(img_rgb_or_bgr, cv2.COLOR_RGB2GRAY)

def _safe_downscale(rgb, max_side=1400):
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side: 
        return rgb
    s = max_side / float(m)
    return cv2.resize(rgb, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def _colormap_jet01(m):
    m01 = cv2.normalize(m.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    return cv2.applyColorMap((m01 * 255).astype(np.uint8), cv2.COLORMAP_JET)


# ------------------------------ TEXTURA (gatekeeper) ------------------------------

class TextureAnalyzer:
    """
    LBP SEM CLAHE — detector primário.
    Regra: fortemente baixo -> IA; fortemente alto -> Natural; meio termo vai para validação.
    """

    def __init__(self, P=8, R=1, block=16, thr_mask=0.35):
        self.P = P
        self.R = R
        self.block = block
        self.thr_mask = thr_mask  # limiar p/ marcar região suspeita no mapa de naturalidade

    def _lbp(self, gray):
        lbp = local_binary_pattern(gray, self.P, self.R, method="uniform")
        return lbp

    def analyze(self, image):
        rgb = _to_np_rgb(image)
        rgb = _safe_downscale(rgb, 1400)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        lbp = self._lbp(gray)

        H, W = lbp.shape
        bs = self.block
        rows = max(1, H // bs)
        cols = max(1, W // bs)

        var_map   = np.zeros((rows, cols), np.float32)
        entr_map  = np.zeros((rows, cols), np.float32)

        # análise por bloco
        for i in range(0, H - bs + 1, bs):
            for j in range(0, W - bs + 1, bs):
                blk = lbp[i:i+bs, j:j+bs]
                # entropia de hist LBP (10 bins estável para P=8 uniform)
                hist, _ = np.histogram(blk, bins=10, range=(0, 10))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
                e = entropy(hist)
                e = e / (np.log(10) + 1e-7)  # 0..1
                v = np.var(blk.astype(np.float32)) / (self.P + 2.0)**2  # 0..~1

                r = i // bs
                c = j // bs
                entr_map[r, c] = e
                var_map[r, c]  = v

        # combinação (levemente pró-entropia, como na V1)
        nat_map = entr_map * 0.65 + var_map * 0.35

        # máscara suspeita (baixa naturalidade)
        susp_mask = nat_map < self.thr_mask
        susp_ratio = float(np.mean(susp_mask))

        # score: média + leve penalização por área suspeita
        base = float(np.mean(nat_map))  # 0..1
        penalty = max(0.8, 1.0 - 0.6 * susp_ratio)  # nunca derruba demais (min 0.8)
        score = int(np.clip(base * penalty, 0, 1) * 100)

        # visuais
        heat = _colormap_jet01(nat_map)
        heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)
        disp  = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)

        mask_u8 = cv2.resize(susp_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, (0, 0, 255), 2)

        # classificação textual (apenas da textura, usada no overlay)
        if score < 35:
            cat = "Alta chance de manipulação"
        elif score < 75:
            cat = "Textura suspeita"
        else:
            cat = "Textura natural"

        return {
            "rgb": rgb,
            "score": score,
            "nat_map": nat_map,
            "susp_ratio": susp_ratio,
            "heatmap": heat,
            "overlay": disp,
            "category": cat
        }


# ------------------------------ BORDAS (validador) ------------------------------

class EdgeAnalyzer:
    """Coerência direcional + densidade de bordas com CLAHE (revela transições artificiais)."""

    def __init__(self, block=24, clahe_clip=2.0, clahe_tile=8):
        self.block = block
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

    def _prep(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
        return clahe.apply(gray)

    def analyze(self, image):
        rgb = _to_np_rgb(image)
        rgb = _safe_downscale(rgb, 1400)
        g = self._prep(rgb)
        H, W = g.shape
        bs = self.block
        rows = max(1, H // bs)
        cols = max(1, W // bs)

        # gradientes
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = cv2.phase(gx, gy)  # 0..2pi

        coh = np.zeros((rows, cols), np.float32)
        den = np.zeros((rows, cols), np.float32)

        for i in range(0, H - bs + 1, bs):
            for j in range(0, W - bs + 1, bs):
                bm = mag[i:i+bs, j:j+bs]
                ba = ang[i:i+bs, j:j+bs]
                r = i // bs
                c = j // bs

                den[r, c] = float(np.mean(bm) / 255.0)

                # coerência circular para pixels “fortes”
                th = np.percentile(bm, 70)
                M = bm > th
                if np.any(M):
                    a = ba[M]
                    mcos = np.mean(np.cos(a))
                    msin = np.mean(np.sin(a))
                    coh[r, c] = np.sqrt(mcos*mcos + msin*msin)  # 0..1
                else:
                    coh[r, c] = 0.5

        coh01 = cv2.normalize(coh, None, 0, 1, cv2.NORM_MINMAX)
        den01 = cv2.normalize(den, None, 0, 1, cv2.NORM_MINMAX)

        edge_nat = 0.6 * coh01 + 0.4 * den01
        score = int(np.clip(np.mean(edge_nat), 0, 1) * 100)

        return {"score": score}


# ------------------------------ RUÍDO (validador) ------------------------------

class NoiseAnalyzer:
    """Consistência de ruído local com CLAHE (coeficiente de variação entre blocos)."""

    def __init__(self, block=32, clahe_clip=2.0, clahe_tile=8):
        self.block = block
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

    def _prep(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
        return clahe.apply(gray)

    def analyze(self, image):
        rgb = _to_np_rgb(image)
        rgb = _safe_downscale(rgb, 1400)
        g = self._prep(rgb)

        H, W = g.shape
        bs = self.block
        rows = max(1, H // bs)
        cols = max(1, W // bs)

        sigmas = []

        for i in range(0, H - bs + 1, bs):
            for j in range(0, W - bs + 1, bs):
                blk = g[i:i+bs, j:j+bs]
                try:
                    s = estimate_sigma(blk, average_sigmas=True, channel_axis=None)
                except Exception:
                    s = float(np.std(blk))
                sigmas.append(max(1e-6, float(s)))

        sigmas = np.array(sigmas, dtype=np.float32)
        mu = float(np.mean(sigmas))
        sd = float(np.std(sigmas))
        cv = sd / (mu + 1e-6)  # coeficiente de variação

        # 0 -> inconsistente, 1 -> consistente
        # (quanto menor o CV, mais consistente)
        score = int(np.clip(1.0 - cv * 1.8, 0, 1) * 100)

        return {"score": score}


# ------------------------------ ILUMINAÇÃO (validador) ------------------------------

class LightingAnalyzer:
    """Consistência global de iluminação (gradiente suave) com CLAHE."""

    def __init__(self, clahe_clip=2.0, clahe_tile=8):
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

    def _prep(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
        return clahe.apply(gray)

    def analyze(self, image):
        rgb = _to_np_rgb(image)
        rgb = _safe_downscale(rgb, 1400)
        g = self._prep(rgb).astype(np.float32)

        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
        mag = cv2.magnitude(gx, gy)

        # suavidade do campo de iluminação
        smooth = 1.0 / (np.std(mag) / 255.0 + 1.0)
        # limita a 0..30 (mantém compatível com versões anteriores)
        score = int(np.clip(smooth * 50, 0, 30))

        return {"score": score}


# ------------------------------ SEQUENCIAL (orquestra) ------------------------------

class SequentialAnalyzer:
    """
    Fluxo:
      1) Textura (gatekeeper forte)
      2) Bordas, Ruído, Iluminação (validadores)
         - confirmação de fraude (2 ruins) OU absolvição (2 bons)
      3) Final ponderado prudente se nada decisivo
    """

    def __init__(self):
        self.texture = TextureAnalyzer(P=8, R=1, block=16, thr_mask=0.35)
        self.edge    = EdgeAnalyzer(block=24)
        self.noise   = NoiseAnalyzer(block=32)
        self.light   = LightingAnalyzer()

    def analyze_sequential(self, image):
        chain = []
        scores = {}
        notes  = []

        # -------- Fase 1: Textura (sem CLAHE)
        tex = self.texture.analyze(image)
        tscore = tex["score"]
        scores["texture"] = int(tscore)
        chain.append("texture")

        if tscore < 35:
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial detectada",
                "main_score": int(tscore),
                "all_scores": scores,
                "phase_notes": notes,
                "validation_chain": chain,
                "phases_executed": 1,
                "visual_report": tex["overlay"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": float(tex["susp_ratio"] * 100.0),
                "detailed_reason": f"Gatekeeper: textura muito baixa ({tscore}/100)."
            }

        if tscore > 75:
            return {
                "verdict": "NATURAL",
                "confidence": 85,
                "reason": "Textura natural consistente",
                "main_score": int(tscore),
                "all_scores": scores,
                "phase_notes": notes,
                "validation_chain": chain,
                "phases_executed": 1,
                "visual_report": tex["overlay"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": float(tex["susp_ratio"] * 100.0),
                "detailed_reason": f"Gatekeeper: textura alta ({tscore}/100)."
            }

        # -------- Fase 2: Bordas
        ed = self.edge.analyze(image)
        escore = ed["score"]
        scores["edge"] = int(escore)
        chain.append("edge")

        # -------- Fase 3: Ruído
        nz = self.noise.analyze(image)
        nscore = nz["score"]
        scores["noise"] = int(nscore)
        chain.append("noise")

        # -------- Fase 4: Iluminação
        lt = self.light.analyze(image)
        lscore = lt["score"]
        scores["lighting"] = int(lscore)
        chain.append("lighting")

        # ----- Regras de decisão intermediárias
        bad = 0
        bad += 1 if escore <= 40 else 0
        bad += 1 if nscore <= 40 else 0
        bad += 1 if lscore <= 10 else 0

        good = 0
        good += 1 if escore >= 70 else 0
        good += 1 if nscore >= 65 else 0
        good += 1 if lscore >= 20 else 0

        if bad >= 2:
            notes.append("Confirmado: ≥2 validadores ruins (edge/noise/light).")
            return {
                "verdict": "MANIPULADA",
                "confidence": 85,
                "reason": "Textura intermediária + validadores negativos",
                "main_score": int(tscore),
                "all_scores": scores,
                "phase_notes": notes,
                "validation_chain": chain,
                "phases_executed": 4,
                "visual_report": tex["overlay"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": float(tex["susp_ratio"] * 100.0),
                "detailed_reason": f"Indicadores ruins: edge={escore}, noise={nscore}, light={lscore}; textura={tscore}."
            }

        if 40 <= tscore <= 70 and good >= 2:
            notes.append("Absolvido: ≥2 validadores bons (edge/noise/light).")
            main = int(tscore * 0.35 + escore * 0.30 + nscore * 0.25 + lscore * 0.10)
            return {
                "verdict": "NATURAL",
                "confidence": 80,
                "reason": "Textura mediana, mas validadores consistentes",
                "main_score": main,
                "all_scores": scores,
                "phase_notes": notes,
                "validation_chain": chain,
                "phases_executed": 4,
                "visual_report": tex["overlay"],
                "heatmap": tex["heatmap"],
                "percent_suspicious": float(tex["susp_ratio"] * 100.0),
                "detailed_reason": f"Absolvição: edge={escore}, noise={nscore}, light={lscore}, texture={tscore}."
            }

        # ----- Final ponderado prudente
        weighted = (tscore * 0.50 + escore * 0.25 + nscore * 0.15 + lscore * 0.10)
        main_score = int(np.clip(weighted, 0, 100))

        if main_score < 45:
            verdict, conf, reason = "MANIPULADA", 80, "Score ponderado baixo"
        elif main_score < 60:
            verdict, conf, reason = "SUSPEITA", 70, "Indicadores ambíguos"
        else:
            verdict, conf, reason = "NATURAL", 70, "Indicadores aceitáveis"

        return {
            "verdict": verdict,
            "confidence": conf,
            "reason": reason,
            "main_score": main_score,
            "all_scores": scores,
            "phase_notes": notes,
            "validation_chain": chain,
            "phases_executed": 4,
            "visual_report": tex["overlay"],
            "heatmap": tex["heatmap"],
            "percent_suspicious": float(tex["susp_ratio"] * 100.0),
            "detailed_reason": (
                f"Scores — texture={tscore}, edge={escore}, noise={nscore}, light={lscore}. "
                f"Ponderado={main_score}/100."
            )
        }


# ------------------------------ helper de download ------------------------------

def get_image_download_link(img, filename, text):
    """Gera link base64 para baixar imagem mostrada."""
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil = Image.fromarray(img)
    else:
        pil = img

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/jpeg;base64,{b64}" download="{filename}">{text}</a>'
