# texture_analyzer.py
# MirrorGlass V4.1 — Análise Sequencial com “Máscara de Detalhe”
# Outubro/2025

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from PIL import Image
import io
import base64


# ----------------------------- Utilidades comuns -----------------------------

def _to_gray_uint8(img):
    """Converte PIL/ndarray BGR/RGB para GRAY uint8."""
    if isinstance(img, Image.Image):
        g = np.array(img.convert("L"))
    else:
        if img.ndim == 3:
            # OpenCV padrão BGR; mas os uploads do Streamlit via PIL chegam RGB.
            # Detecta heurística simples: se veio de PIL, provavelmente RGB.
            # Para garantir, usamos o canal de maior variação como referência.
            # No fim, cinza por conversão perceptual:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.flags['C_CONTIGUOUS'] else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            g = img.copy()
    g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def compute_detail_mask(gray, win=9, lap_ksize=3, lap_thr=8.0, std_thr=6.0):
    """
    Cria uma máscara binária de 'regiões com detalhe' usando:
      - magnitude do Laplaciano (alta-frequência)
      - desvio padrão local (micro-textura)
    Retorna (mask[0/1], detail_ratio, lap_norm_visual).
    """
    # Alta frequência (Laplaciano)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=lap_ksize)
    lap = np.abs(lap).astype(np.float32)

    # Desvio padrão local
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (win, win))
    mu2 = cv2.blur(g * g, (win, win))
    var = np.clip(mu2 - mu * mu, 0, None)
    std = np.sqrt(var)

    # Limiarização (robusta a escala)
    # Usa limiar dinâmico relativo ao percentil para fotos grandes
    lap_t = max(lap_thr, float(np.percentile(lap, 65)))
    std_t = max(std_thr, float(np.percentile(std, 65)))

    mask = ((lap >= lap_t) | (std >= std_t)).astype(np.uint8)

    # Abre/fecha leve p/ limpar ruído
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    detail_ratio = float(mask.mean())

    # Visual (normalizado) — útil para debugar na UI se quiser
    lap_vis = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mask, detail_ratio, lap_vis


def block_weight_from_mask(mask, H, W, block):
    """Peso do bloco = fração de pixels com detalhe dentro do bloco (0..1)."""
    bh, bw = block
    # garante dimensões pares de varredura
    rows = max(1, H // bh)
    cols = max(1, W // bw)
    weights = np.zeros((rows, cols), dtype=np.float32)
    for i in range(0, H - bh + 1, bh):
        for j in range(0, W - bw + 1, bw):
            r = i // bh
            c = j // bw
            sub = mask[i:i+bh, j:j+bw]
            weights[r, c] = float(sub.mean())
    return weights


# ----------------------------- Fase 1 — Textura ------------------------------

class TextureAnalyzer:
    """
    Textura com LBP (sem CLAHE). Agora com ponderação por 'máscara de detalhe':
    blocos lisos contam pouco (ou nada), evitando falso-positivo em vidros/ceu.
    """
    def __init__(self, P=8, R=1, block_size=24, threshold=0.50):
        self.P = P
        self.R = R
        self.block = block_size
        self.threshold = threshold  # limiar do mapa (0..1) para 'suspeito'

    def _lbp(self, gray):
        lbp = local_binary_pattern(gray, self.P, self.R, method="uniform")
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return lbp, hist

    def analyze(self, image, detail_mask=None):
        gray = _to_gray_uint8(image)
        H, W = gray.shape
        lbp_img, _ = self._lbp(gray)

        rows = max(1, H // self.block)
        cols = max(1, W // self.block)

        variance_map = np.zeros((rows, cols), np.float32)
        entropy_map = np.zeros((rows, cols), np.float32)
        uniform_map = np.zeros((rows, cols), np.float32)

        # pesos por detalhe (0..1); se não fornecido, peso=1
        if detail_mask is None:
            weights = np.ones((rows, cols), np.float32)
            detail_ratio = 1.0
        else:
            weights = block_weight_from_mask(detail_mask, H, W, (self.block, self.block))
            detail_ratio = float(detail_mask.mean())

        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                r = i // self.block
                c = j // self.block

                block = lbp_img[i:i+self.block, j:j+self.block]
                # Se quase não há detalhe nesse bloco, trate como neutro leve
                w = weights[r, c]
                if w < 0.15:
                    variance_map[r, c] = 0.25  # valor neutro
                    entropy_map[r, c] = 0.25
                    uniform_map[r, c] = 0.25
                    continue

                hist, _ = np.histogram(block, bins=10, range=(0, 10))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
                ent = entropy(hist) / np.log(10.0)

                # Nota: LBP já é '0..(P+2)'; var normalizada
                var = np.var(block / float(self.P + 2))

                # penaliza picos de um único código LBP (uniformidade excessiva)
                uniform_penalty = 1.0 - float(hist.max())

                variance_map[r, c] = var
                entropy_map[r, c] = ent
                uniform_map[r, c] = uniform_penalty

        # Combinação — prioriza var/uniformidade; entropia ajuda
        natural_map = 0.40 * variance_map + 0.40 * uniform_map + 0.20 * entropy_map

        # máscara de suspeita só onde há detalhe real
        sus_mask = (natural_map < self.threshold).astype(np.uint8)
        if detail_mask is not None:
            # expande a máscara de blocos para a imagem (visual apenas)
            pass

        # score ponderado pelos pesos dos blocos (não média simples)
        wsum = weights.sum() + 1e-7
        score = float((natural_map * weights).sum() / wsum)
        score = int(np.clip(score, 0, 1) * 100)

        # proporção suspeita ponderada
        sus_ratio = float((sus_mask * weights).sum() / wsum)

        # heatmap (visual)
        vis = cv2.normalize(natural_map, None, 0, 1, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return {
            "score": score,
            "natural_map": natural_map,
            "sus_mask": sus_mask,
            "sus_ratio": sus_ratio,
            "heatmap": heat,
            "detail_ratio": detail_ratio
        }

    @staticmethod
    def categorize(score):
        if score <= 45:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 68:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"

    @staticmethod
    def render_overlay(image, natural_map, sus_mask, score):
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        H, W = image.shape[:2]
        nat = cv2.resize(natural_map, (W, H), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(sus_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap((cv2.normalize(nat, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0.0)
        cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = overlay.copy()
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
        cat, _ = TextureAnalyzer.categorize(score)
        cv2.putText(out, f"Score: {score}/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(out, cat, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return out, heat


# --------------------------- Fase 2 — Bordas (CLAHE) -------------------------

class EdgeAnalyzer:
    """
    Coerência direcional + densidade de bordas, mas **apenas nas regiões com detalhe**.
    Em cenas lisas, devolve nota neutra (≈60) para não derrubar o veredito sozinha.
    """
    def __init__(self, block_size=24, use_clahe=True, clahe_clip=2.0, clahe_tile=8):
        self.block = block_size
        self.use_clahe = use_clahe
        self.clip = clahe_clip
        self.tile = clahe_tile

    def _prep(self, image):
        gray = _to_gray_uint8(image)
        if not self.use_clahe:
            return gray
        clahe = cv2.createCLAHE(self.clip, (self.tile, self.tile))
        return clahe.apply(gray)

    def analyze(self, image, detail_mask, detail_ratio):
        gray = self._prep(image)
        H, W = gray.shape

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        ang = np.arctan2(gy, gx)

        rows = max(1, H // self.block)
        cols = max(1, W // self.block)

        # pesos por detalhe
        weights = block_weight_from_mask(detail_mask, H, W, (self.block, self.block))
        wsum = weights.sum() + 1e-7

        coh_map = np.zeros((rows, cols), np.float32)
        den_map = np.zeros((rows, cols), np.float32)

        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                r = i // self.block
                c = j // self.block
                w = weights[r, c]

                if w < 0.15:
                    coh_map[r, c] = 0.60  # neutro leve
                    den_map[r, c] = 0.60
                    continue

                M = mag[i:i+self.block, j:j+self.block]
                A = ang[i:i+self.block, j:j+self.block]

                if np.sum(M > 10) < 10:
                    coh_map[r, c] = 0.50
                    den_map[r, c] = 0.50
                    continue

                sig = M > np.percentile(M, 70)
                if np.any(sig):
                    mc = float(np.mean(np.cos(A[sig])))
                    ms = float(np.mean(np.sin(A[sig])))
                    circ_var = 1.0 - np.sqrt(mc*mc + ms*ms)
                    coh_map[r, c] = 1.0 - circ_var
                else:
                    coh_map[r, c] = 0.5

                den_map[r, c] = float(np.mean(M) / 255.0)

        edge_nat = 0.6 * cv2.normalize(coh_map, None, 0, 1, cv2.NORM_MINMAX) + \
                   0.4 * cv2.normalize(den_map, None, 0, 1, cv2.NORM_MINMAX)

        score = int(np.clip(float((edge_nat * weights).sum() / wsum), 0, 1) * 100)

        # Em cena MUITO lisa, devolve neutro alto (não derruba)
        if detail_ratio < 0.25:
            score = max(score, 60)

        return score


# ---------------------------- Fase 3 — Ruído (CLAHE) -------------------------

class NoiseAnalyzer:
    """
    Consistência de ruído por blocos (estimate_sigma). Apenas em regiões com detalhe.
    Em cena lisa, devolve neutro (≈60).
    """
    def __init__(self, block_size=32, use_clahe=True, clahe_clip=2.0, clahe_tile=8):
        self.block = block_size
        self.use_clahe = use_clahe
        self.clip = clahe_clip
        self.tile = clahe_tile

    def _prep(self, image):
        gray = _to_gray_uint8(image)
        if not self.use_clahe:
            return gray
        clahe = cv2.createCLAHE(self.clip, (self.tile, self.tile))
        return clahe.apply(gray)

    def analyze(self, image, detail_mask, detail_ratio):
        gray = self._prep(image)
        H, W = gray.shape

        rows = max(1, H // self.block)
        cols = max(1, W // self.block)
        weights = block_weight_from_mask(detail_mask, H, W, (self.block, self.block))

        noise_map = np.zeros((rows, cols), np.float32)

        for i in range(0, H - self.block + 1, self.block):
            for j in range(0, W - self.block + 1, self.block):
                r = i // self.block
                c = j // self.block
                w = weights[r, c]

                if w < 0.15:
                    noise_map[r, c] = 0.6  # neutro leve
                    continue

                blk = gray[i:i+self.block, j:j+self.block]
                try:
                    sig = estimate_sigma(blk, average_sigmas=True, channel_axis=None)
                    noise_map[r, c] = float(sig)
                except Exception:
                    noise_map[r, c] = float(np.std(blk))

        # Coeficiente de variação (heterogeneidade do ruído)
        vals = noise_map[weights > 0.15]
        if vals.size == 0:
            return 60

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = 0.0 if mean <= 1e-6 else std / mean

        # quanto mais homogêneo (cv baixo), melhor
        score = int(np.clip(1.0 - min(cv, 0.5) * 2.0, 0, 1) * 100)

        if detail_ratio < 0.25:
            score = max(score, 60)

        return score


# --------------------------- Fase 4 — Iluminação (CLAHE) ---------------------

class LightingAnalyzer:
    """
    Usa variação da magnitude de gradiente como proxy de consistência fotométrica.
    Reescalado para 0..100. Em cena lisa, não derruba: mínimo 55–60.
    """
    def __init__(self, use_clahe=True, clahe_clip=2.0, clahe_tile=8):
        self.use_clahe = use_clahe
        self.clip = clahe_clip
        self.tile = clahe_tile

    def _prep(self, image):
        gray = _to_gray_uint8(image)
        if not self.use_clahe:
            return gray
        clahe = cv2.createCLAHE(self.clip, (self.tile, self.tile))
        return clahe.apply(gray)

    def analyze(self, image, detail_mask, detail_ratio):
        gray = self._prep(image)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        mag = np.sqrt(gx*gx + gy*gy)

        # Avalia só nas regiões com detalhe
        sel = mag[detail_mask > 0]
        if sel.size < 256:  # quase sem detalhe
            return 60

        s = float(np.std(sel))
        # Heurística: mais 'suave' => maior score
        smoothness = 1.0 / (s + 1.0)
        score = int(np.clip(smoothness * 120.0, 0, 1) * 100)

        if detail_ratio < 0.25:
            score = max(score, 60)

        return score


# ---------------------------- Orquestrador sequencial ------------------------

class SequentialAnalyzer:
    """
    Executa as quatro fases, compartilhando a máscara de detalhe.
    Regras:
      - Condena rápido só se textura < 35 **e** detail_ratio >= 0.25.
      - Em textura entre 45–75, absolve se 2 de 3 (borda/ruído/luz) >= 70.
      - Cena lisa (detail_ratio < 0.25): validadores devolvem neutro (>=60);
        só condena se algum validador cair MUITO (edge<20 ou noise<15 ou light<15).
    """
    def __init__(self):
        self.tex = TextureAnalyzer()
        self.edge = EdgeAnalyzer()
        self.noise = NoiseAnalyzer()
        self.light = LightingAnalyzer()

    def analyze_sequential(self, image):
        # 0) máscara de detalhe
        gray = _to_gray_uint8(image)
        dmask, dratio, _ = compute_detail_mask(gray)

        # 1) textura
        t = self.tex.analyze(image, dmask)
        t_score = t["score"]

        # Decisão imediata por textura (apenas se há detalhe suficiente)
        if t_score < 35 and dratio >= 0.25:
            overlay, heat = self.tex.render_overlay(image, t["natural_map"], t["sus_mask"], t_score)
            return {
                "verdict": "MANIPULADA",
                "confidence": 95,
                "reason": "Textura artificial detectada em áreas detalhadas",
                "main_score": t_score,
                "all_scores": {"texture": t_score},
                "validation_chain": ["texture"],
                "phases_executed": 1,
                "visual_report": overlay,
                "heatmap": heat,
                "percent_suspicious": round(t["sus_ratio"]*100, 2),
                "detailed_reason": f"Textura={t_score}, detalhe={int(dratio*100)}%",
            }

        # 2–4) validadores, sempre ponderados pela máscara de detalhe
        e_score = self.edge.analyze(image, dmask, dratio)
        n_score = self.noise.analyze(image, dmask, dratio)
        l_score = self.light.analyze(image, dmask, dratio)

        overlay, heat = self.tex.render_overlay(image, t["natural_map"], t["sus_mask"], t_score)

        # Regras fortes em cena lisa
        if dratio < 0.25:
            # Só condena se houver sinal muito forte em região com detalhe
            if e_score < 20 or n_score < 15 or l_score < 15 or t_score < 30:
                verdict = "MANIPULADA"
                conf = 85
                reason = "Indicadores fortes em cena majoritariamente lisa"
            else:
                # Natural por ausência de sinais fortes
                verdict = "NATURAL" if t_score >= 50 else "INCONCLUSIVA"
                conf = 75 if verdict == "NATURAL" else 60
                reason = "Cena lisa; validadores neutros; sem inconsistências fortes"
            return {
                "verdict": verdict,
                "confidence": conf,
                "reason": reason,
                "main_score": int(0.55*t_score + 0.20*e_score + 0.15*n_score + 0.10*l_score),
                "all_scores": {"texture": t_score, "edge": e_score, "noise": n_score, "lighting": l_score, "detail_ratio": int(dratio*100)},
                "validation_chain": ["texture","edge","noise","lighting"],
                "phases_executed": 4,
                "visual_report": overlay,
                "heatmap": heat,
                "percent_suspicious": round(t["sus_ratio"]*100, 2),
                "detailed_reason": f"T={t_score} E={e_score} N={n_score} L={l_score} Det={int(dratio*100)}%"
            }

        # Cena com detalhe razoável
        # “Absolvição por maioria” se textura média e 2 de 3 validam bem
        good = (e_score >= 70) + (n_score >= 70) + (l_score >= 70)
        if 45 <= t_score <= 75 and good >= 2:
            return {
                "verdict": "NATURAL",
                "confidence": 80,
                "reason": "Textura intermediária, validadores consistentes nas regiões detalhadas",
                "main_score": int(0.35*t_score + 0.30*e_score + 0.25*n_score + 0.10*l_score),
                "all_scores": {"texture": t_score, "edge": e_score, "noise": n_score, "lighting": l_score, "detail_ratio": int(dratio*100)},
                "validation_chain": ["texture","edge","noise","lighting"],
                "phases_executed": 4,
                "visual_report": overlay,
                "heatmap": heat,
                "percent_suspicious": round(t["sus_ratio"]*100, 2),
                "detailed_reason": f"Maioria absolveu: E={e_score}, N={n_score}, L={l_score}, T={t_score}."
            }

        # Condenação por múltiplos fracos
        weak = (e_score < 40) + (n_score < 40) + (l_score < 30)
        if t_score < 50 and weak >= 2:
            return {
                "verdict": "MANIPULADA",
                "confidence": 88,
                "reason": "Textura baixa e múltiplos validadores fracos em áreas detalhadas",
                "main_score": int(0.55*t_score + 0.20*e_score + 0.15*n_score + 0.10*l_score),
                "all_scores": {"texture": t_score, "edge": e_score, "noise": n_score, "lighting": l_score, "detail_ratio": int(dratio*100)},
                "validation_chain": ["texture","edge","noise","lighting"],
                "phases_executed": 4,
                "visual_report": overlay,
                "heatmap": heat,
                "percent_suspicious": round(t["sus_ratio"]*100, 2),
                "detailed_reason": f"Fracos: E<{40 if e_score<40 else e_score} N<{40 if n_score<40 else n_score} L<{30 if l_score<30 else l_score}; T={t_score}"
            }

        # Caso final — ponderado
        final_score = int(0.50*t_score + 0.25*e_score + 0.15*n_score + 0.10*l_score)
        if final_score < 55:
            verdict, conf, reason = "SUSPEITA", 70, "Indicadores ambíguos nas regiões detalhadas"
        else:
            verdict, conf, reason = "INCONCLUSIVA", 60, "Revisão manual sugerida"

        return {
            "verdict": verdict,
            "confidence": conf,
            "reason": reason,
            "main_score": final_score,
            "all_scores": {"texture": t_score, "edge": e_score, "noise": n_score, "lighting": l_score, "detail_ratio": int(dratio*100)},
            "validation_chain": ["texture","edge","noise","lighting"],
            "phases_executed": 4,
            "visual_report": overlay,
            "heatmap": heat,
            "percent_suspicious": round(t["sus_ratio"]*100, 2),
            "detailed_reason": f"Score ponderado {final_score} com Det={int(dratio*100)}%"
        }


# --------------------------- Utilitário de download --------------------------

def get_image_download_link(img, filename, text):
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(img)
    else:
        img_pil = img
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/jpeg;base64,{b64}" download="{filename}">{text}</a>'
