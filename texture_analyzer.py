# texture_analyzer.py
# MirrorGlass V4.3 — Sistema de Análise Sequencial com Priors Fotográficos
# (c) 2025
# Requisitos: numpy, opencv-python(-headless), Pillow, scikit-image, scipy

from __future__ import annotations
import io
import base64
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma


# =========================
# Utilidades de imagem
# =========================

def to_numpy_rgb(image: Any) -> np.ndarray:
    """Converte PIL/NumPy/BGR em RGB uint8 sem alterar contraste."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    # se vier BGR típico do OpenCV, converte para RGB
    if np.mean(arr[:, :, 0] - arr[:, :, 2]) > 1.0:  # heurística leve
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def safe_resize_long_side(rgb: np.ndarray, long_side: int = 1024) -> np.ndarray:
    """Redimensiona mantendo o aspecto (long_side máx). Não aplica sharpening/CLAHE."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= long_side:
        return rgb
    scale = long_side / float(s)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


# =========================
# Priors fotográficos
# =========================

@dataclass
class PhotoPriors:
    """Extrai indícios de que a imagem é uma FOTO (não IA/arte)."""
    # nada parametrizável aqui por ora

    def compute(self, rgb: np.ndarray) -> Dict[str, float]:
        gray = rgb_to_gray_u8(rgb).astype(np.float32)

        # 1) Nitidez (variância do Laplaciano)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        blur_var_laplacian = float(np.var(lap))

        # 2) Blockiness de JPEG (diferença média nas fronteiras 8x8)
        #    Mede saltos em colunas/linhas múltiplas de 8 (artefato típico de JPEG).
        h, w = gray.shape
        # dif horizontal nas fronteiras verticais (colunas múltiplas de 8)
        cols = np.arange(8, w, 8)
        if len(cols) > 0:
            dh = np.abs(gray[:, cols] - gray[:, cols - 1]).mean()
        else:
            dh = 0.0
        # dif vertical nas fronteiras horizontais (linhas múltiplas de 8)
        rows = np.arange(8, h, 8)
        if len(rows) > 0:
            dv = np.abs(gray[rows, :] - gray[rows - 1, :]).mean()
        else:
            dv = 0.0
        jpeg_blockiness = float((dh + dv) / 2.0 / 255.0)  # normaliza para ~[0,1]

        # 3) Correlação ruído-luma (fotos tendem a ter correlação discretamente negativa)
        #    Residual de alta-frequência por subtração de desfoque leve.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        resid = gray - blur
        g = gray.astype(np.float32).ravel()
        r = resid.astype(np.float32).ravel()
        if np.std(g) > 1e-6 and np.std(r) > 1e-6:
            noise_luma_corr = float(np.corrcoef(g, np.abs(r))[0, 1])
        else:
            noise_luma_corr = 0.0

        return {
            "blur_var_laplacian": blur_var_laplacian,
            "jpeg_blockiness": jpeg_blockiness,
            "noise_luma_corr": noise_luma_corr,
        }


# =========================
# Texture (fase 1)
# =========================

class TextureAnalyzer:
    """
    Análise de textura por LBP (sem CLAHE).
    Score alto ~ textura natural / Score baixo ~ textura artificial/alisada.
    """

    def __init__(self, P: int = 8, R: int = 1, block: int = 24, thr: float = 0.50):
        self.P = P
        self.R = R
        self.block = block
        self.thr = thr  # limiar p/ máscara de suspeita, no espaço [0..1]

    def _lbp(self, gray_u8: np.ndarray) -> np.ndarray:
        return local_binary_pattern(gray_u8, self.P, self.R, method="uniform")

    def analyze_texture_variance(self, image: Any) -> Dict[str, Any]:
        rgb = to_numpy_rgb(image)
        gray = rgb_to_gray_u8(rgb)
        lbp_img = self._lbp(gray)

        h, w = lbp_img.shape
        rows = max(1, h // self.block)
        cols = max(1, w // self.block)

        variance_map = np.zeros((rows, cols), dtype=np.float32)
        entropy_map = np.zeros((rows, cols), dtype=np.float32)
        uniformity_map = np.zeros((rows, cols), dtype=np.float32)

        # variação, entropia e "uniformidade" por bloco
        for i in range(0, h - self.block + 1, self.block):
            for j in range(0, w - self.block + 1, self.block):
                r = i // self.block
                c = j // self.block
                blk = lbp_img[i:i + self.block, j:j + self.block]

                # hist LBP (10 bins cobrem método 'uniform' com P=8)
                hist, _ = np.histogram(blk, bins=10, range=(0, 10))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-7)
                ent = entropy(hist)
                ent_norm = ent / (np.log(10) + 1e-12)

                # variância do LBP (normalizada)
                var = float(np.var(blk / float(self.P + 2)))

                # penalidade por “pico” (alto máximo no histograma -> muito uniforme)
                uniform_penalty = 1.0 - float(np.max(hist))

                variance_map[r, c] = var
                entropy_map[r, c] = ent_norm
                uniformity_map[r, c] = uniform_penalty

        # combinação (pesos priorizando variância e (não) uniformidade)
        natural_map = 0.40 * variance_map + 0.30 * entropy_map + 0.30 * uniformity_map
        suspicious_mask = natural_map < self.thr
        mean_nat = float(np.mean(natural_map))
        susp_ratio = float(np.mean(suspicious_mask))

        # penalização: pequenas áreas suspeitas não derrubam tanto; áreas grandes derrubam mais
        if susp_ratio <= 0.10:
            penalty = 1.0 - 0.8 * susp_ratio
        elif susp_ratio <= 0.25:
            penalty = 0.92 - 1.2 * (susp_ratio - 0.10)
        else:
            penalty = 0.74 - 1.6 * (susp_ratio - 0.25)
        penalty = float(np.clip(penalty, 0.5, 1.0))

        score = int(np.clip(mean_nat * penalty * 100.0, 0, 100))

        # heatmap só para visualização (normalizado localmente)
        disp = cv2.normalize(natural_map, None, 0, 1, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return {
            "naturalness_map": natural_map,
            "suspicious_mask": suspicious_mask,
            "suspicious_ratio": susp_ratio,
            "naturalness_score": score,
            "heatmap": heat,
        }


# =========================
# Edge (fase 2)
# =========================

class EdgeAnalyzer:
    def __init__(self, block: int = 24):
        self.block = block

    def analyze(self, image: Any) -> Dict[str, Any]:
        rgb = to_numpy_rgb(image)
        g = rgb_to_gray_u8(rgb)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx)

        h, w = g.shape
        rows = max(1, h // self.block)
        cols = max(1, w // self.block)
        coh = np.zeros((rows, cols), dtype=np.float32)
        dens = np.zeros((rows, cols), dtype=np.float32)

        for i in range(0, h - self.block + 1, self.block):
            for j in range(0, w - self.block + 1, self.block):
                r = i // self.block
                c = j // self.block
                blk_m = mag[i:i + self.block, j:j + self.block]
                blk_a = ang[i:i + self.block, j:j + self.block]

                dens[r, c] = float(np.mean(blk_m) / 255.0)

                if np.sum(blk_m > 10) > 10:
                    sig = blk_m > np.percentile(blk_m, 70)
                    if np.any(sig):
                        a = blk_a[sig]
                        mean_cos = float(np.mean(np.cos(a)))
                        mean_sin = float(np.mean(np.sin(a)))
                        circ_var = 1.0 - np.sqrt(mean_cos ** 2 + mean_sin ** 2)
                        coh[r, c] = 1.0 - circ_var
                    else:
                        coh[r, c] = 0.5
                else:
                    coh[r, c] = 0.5

        coh_n = cv2.normalize(coh, None, 0, 1, cv2.NORM_MINMAX)
        dens_n = cv2.normalize(dens, None, 0, 1, cv2.NORM_MINMAX)
        edge_nat = 0.6 * coh_n + 0.4 * dens_n
        edge_score = int(np.clip(edge_nat.mean() * 100.0, 0, 100))
        return {"edge_score": edge_score}


# =========================
# Noise (fase 3)
# =========================

class NoiseAnalyzer:
    def __init__(self, block: int = 32):
        self.block = block

    def analyze(self, image: Any) -> Dict[str, Any]:
        rgb = to_numpy_rgb(image)
        g = rgb_to_gray_u8(rgb)

        h, w = g.shape
        rows = max(1, h // self.block)
        cols = max(1, w // self.block)
        nm = np.zeros((rows, cols), dtype=np.float32)

        for i in range(0, h - self.block + 1, self.block):
            for j in range(0, w - self.block + 1, self.block):
                r = i // self.block
                c = j // self.block
                blk = g[i:i + self.block, j:j + self.block]
                try:
                    sigma = estimate_sigma(blk, average_sigmas=True, channel_axis=None)
                    nm[r, c] = float(sigma)
                except Exception:
                    nm[r, c] = float(np.std(blk))

        m = float(np.mean(nm))
        s = float(np.std(nm))
        cv = s / m if m > 1e-6 else 0.0
        # coerência de ruído (alto quanto mais homogêneo/real)
        noise_score = int(np.clip(100.0 - (cv * 200.0), 0, 100))
        return {"noise_score": noise_score}


# =========================
# Lighting (fase 4)
# =========================

class LightingAnalyzer:
    def analyze(self, image: Any) -> Dict[str, Any]:
        rgb = to_numpy_rgb(image)
        g = rgb_to_gray_u8(rgb)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 5)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 5)
        mag = np.sqrt(gx * gx + gy * gy)
        smoothness = 1.0 / (np.std(mag) + 1.0)  # estável
        lighting_score = int(min(smoothness * 50.0, 30.0))
        return {"lighting_score": lighting_score}


# =========================
# Orquestrador (sequencial)
# =========================

class SequentialAnalyzer:
    """
    Orquestra as 4 fases + priors e aplica as regras de decisão.
    Retorna também telemetria completa para depuração.
    """

    def __init__(self, long_side_px: int = 1024):
        self.texture = TextureAnalyzer()
        self.edge = EdgeAnalyzer()
        self.noise = NoiseAnalyzer()
        self.light = LightingAnalyzer()
        self.priors = PhotoPriors()
        self.long_side = long_side_px

    # ---- helpers visuais
    def _visual(self, rgb: np.ndarray, t_res: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        img = rgb.copy()
        h, w = img.shape[:2]
        natural_map = t_res["naturalness_map"]
        mask = t_res["suspicious_mask"].astype(np.uint8)

        nm = cv2.resize(natural_map, (w, h), interpolation=cv2.INTER_LINEAR)
        mk = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        disp = cv2.normalize(nm, None, 0, 1, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)  # contorno em vermelho (RGB)

        # texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        score = int(t_res["naturalness_score"])
        cv2.putText(overlay, f"Score: {score}/100", (10, 28), font, 0.7, (255, 255, 255), 2)
        return overlay, heat

    def _pack_dbg(self, verdict: str, confidence: int, reason: str, main_score: int,
                  phase_scores: Dict[str, Optional[int]], decision_trace: list,
                  vis_img: np.ndarray, heat: np.ndarray, t_res: Dict[str, Any],
                  priors: Dict[str, float], prior_flags: Dict[str, bool],
                  prior_votes: int, gate: Optional[str]) -> Dict[str, Any]:
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "main_score": int(main_score),
            "all_scores": {k: v for k, v in phase_scores.items() if v is not None},
            "validation_chain": decision_trace,
            "phases_executed": len(decision_trace),
            "visual_report": vis_img,
            "heatmap": heat,
            "percent_suspicious": float(np.mean(t_res["suspicious_mask"]) * 100.0),
            "detailed_reason": f"susp_ratio={t_res['suspicious_ratio']:.2f}",
            # telemetria
            "phase_scores": phase_scores,
            "priors": priors,
            "prior_flags": prior_flags,
            "prior_votes": prior_votes,
            "decision_trace": decision_trace,
            "decision_gate": gate,
            "weighted_score": None,
        }

    def analyze_sequential(self, image: Any) -> Dict[str, Any]:
        # ---------- pré-processo ----------
        rgb = to_numpy_rgb(image)
        rgb = safe_resize_long_side(rgb, self.long_side)

        # ---------- priors (limiares mais rígidos) ----------
        p = self.priors.compute(rgb)
        blur_ok = p["blur_var_laplacian"] >= 120.0     # nitidez
        jpeg_ok = p["jpeg_blockiness"] >= 0.18         # blockiness
        corr_ok = p["noise_luma_corr"] <= -0.25        # correlação ruído-luma
        prior_votes = int(blur_ok) + int(jpeg_ok) + int(corr_ok)

        # ---------- FASE 1: Textura ----------
        t_res = self.texture.analyze_texture_variance(rgb)
        t = int(t_res["naturalness_score"])
        vis, heat = self._visual(rgb, t_res)

        phase_scores = {"texture": t, "edge": None, "noise": None, "lighting": None}
        pri_flags = {"blur_ok": bool(blur_ok), "jpeg_ok": bool(jpeg_ok), "corr_ok": bool(corr_ok)}
        trace = ["texture"]

        # Queda direta por textura muito baixa
        if t < 28:
            return self._pack_dbg("MANIPULADA", 92, "Textura fortemente artificial",
                                  t, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "texture_hard")
        if t < 36 and prior_votes == 0:
            return self._pack_dbg("MANIPULADA", 90, "Textura suspeita sem sinais de foto",
                                  t, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "texture_soft")

        # Absolvição direta: textura realmente alta + múltiplos sinais de foto
        if t >= 70 and prior_votes >= 2:
            return self._pack_dbg("NATURAL", 85, "Textura alta e priors fotográficos",
                                  t, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "texture_absolve")

        # ---------- FASE 2: Bordas ----------
        e = int(self.edge.analyze(rgb)["edge_score"])
        phase_scores["edge"] = e
        trace.append("edge")
        if e < 35 and t < 50 and prior_votes == 0:
            return self._pack_dbg("MANIPULADA", 88, "Textura fraca e bordas artificiais",
                                  t, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "edge")

        # ---------- FASE 3: Ruído ----------
        n = int(self.noise.analyze(rgb)["noise_score"])
        phase_scores["noise"] = n
        trace.append("noise")
        if n < 35 and t < 50 and prior_votes == 0:
            return self._pack_dbg("MANIPULADA", 85, "Textura fraca e ruído inconsistente",
                                  t, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "noise")

        # ---------- FASE 4: Iluminação ----------
        l = int(self.light.analyze(rgb)["lighting_score"])
        phase_scores["lighting"] = l
        trace.append("lighting")
        if l < 8 and t < 50 and prior_votes == 0:
            return self._pack_dbg("MANIPULADA", 80, "Iluminação fisicamente inconsistente",
                                  t, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "lighting")

        # ---------- Absolvedor por maioria (duro) ----------
        good = (1 if e >= 68 else 0) + (1 if n >= 68 else 0) + (1 if l >= 22 else 0) + (1 if prior_votes >= 2 else 0)
        at_least_edge_or_noise = (e >= 68) or (n >= 68)
        if 45 <= t <= 62 and good >= 3 and at_least_edge_or_noise:
            main = int(0.30 * t + 0.30 * e + 0.25 * n + 0.15 * l)
            return self._pack_dbg("NATURAL", 80,
                                  "Textura intermediária absolvida por bordas/ruído + priors",
                                  main, phase_scores, trace, vis, heat, t_res, p, pri_flags, prior_votes, "majority_absolve")

        # ---------- Score final (conservador) ----------
        weighted = 0.45 * t + 0.23 * e + 0.20 * n + 0.12 * l

        if weighted >= 70 and prior_votes >= 2:
            verdict, conf, reason = "NATURAL", 75, "Conjunto de indícios favorece foto"
        elif weighted < 50 and prior_votes == 0:
            verdict, conf, reason = "MANIPULADA", 80, "Indícios combinados e ausência de sinais de foto"
        else:
            # deixa em aberto para revisão humana
            if weighted < 60:
                verdict, conf, reason = "SUSPEITA", 70, "Indicadores mistos"
            else:
                verdict, conf, reason = "INCONCLUSIVA", 60, "Sinais conflitantes"

        return {
            "verdict": verdict,
            "confidence": conf,
            "reason": reason,
            "main_score": int(np.clip(weighted, 0, 100)),
            "all_scores": {k: v for k, v in phase_scores.items() if v is not None},
            "validation_chain": trace,
            "phases_executed": len(trace),
            "visual_report": vis,
            "heatmap": heat,
            "percent_suspicious": float(np.mean(t_res["suspicious_mask"]) * 100.0),
            "detailed_reason": f"texture={t}, edge={e}, noise={n}, light={l}, priors_votes={prior_votes}",
            # telemetria
            "phase_scores": phase_scores,
            "priors": p,
            "prior_flags": pri_flags,
            "prior_votes": prior_votes,
            "decision_trace": trace,
            "decision_gate": None,
            "weighted_score": float(weighted),
        }


# =========================
# Download util (mantido)
# =========================

def get_image_download_link(img, filename, text):
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img_pil = Image.fromarray(img)
    else:
        img_pil = img

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href
