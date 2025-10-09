import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import json
import pandas as pd
import time
import cv2

from texture_analyzer import SequentialAnalyzer

st.set_page_config(
    page_title="MirrorGlass V4 - Detector de Fraudes",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Título limpo
st.title("🔍 MirrorGlass V4.1 - Detector de Fraudes por IA")
st.markdown("**Análise Sequencial com Validação em Cadeia**")

# Info compacta
with st.expander("ℹ️ Como funciona"):
    st.markdown("""
    **Fases de Validação:**
    1. **Textura:** Detecta uniformidade artificial (sem CLAHE)
    2. **Bordas:** Valida transições (com CLAHE)  
    3. **Ruído:** Analisa consistência (com CLAHE)
    4. **Iluminação:** Valida física (com CLAHE)
    
    **Sistema para quando tem certeza suficiente** (70% decidem na fase 1).
    """)

def get_verdict_emoji(verdict):
    if verdict == "MANIPULADA":
        return "🔴"
    elif verdict == "NATURAL":
        return "🟢"
    elif verdict == "SUSPEITA":
        return "🟡"
    else:
        return "⚪"

def analisar_sequencial(imagens, nomes):
    analyzer = SequentialAnalyzer()
    progress_bar = st.progress(0)
    status_text = st.empty()
    resultados = []
    
    for i, img in enumerate(imagens):
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {i+1}/{len(imagens)}: {nomes[i]}")
        
        try:
            report = analyzer.analyze_sequential(img)
            resultados.append({
                "nome": nomes[i],
                "verdict": report["verdict"],
                "confidence": report["confidence"],
                "reason": report["reason"],
                "main_score": report["main_score"],
                "all_scores": report["all_scores"],
                "validation_chain": report["validation_chain"],
                "phases_executed": report["phases_executed"],
                "visual_report": report["visual_report"],
                "heatmap": report["heatmap"],
                "percent_suspicious": report["percent_suspicious"],
                "detailed_reason": report["detailed_reason"]
            })
        except Exception as e:
            st.error(f"Erro: {nomes[i]} - {str(e)}")
            resultados.append({
                "nome": nomes[i],
                "verdict": "ERRO",
                "confidence": 0,
                "reason": f"Erro: {str(e)}",
                "main_score": 0,
                "all_scores": {},
                "validation_chain": [],
                "phases_executed": 0,
                "visual_report": None,
                "heatmap": None,
                "percent_suspicious": 0,
                "detailed_reason": "Falha na análise"
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return resultados

def exibir_resultados(resultados):
    if not resultados:
        st.info("Nenhum resultado disponível.")
        return None
    
    # Estatísticas no topo
    col1, col2, col3, col4 = st.columns(4)
    
    manipuladas = sum(1 for r in resultados if r["verdict"] == "MANIPULADA")
    naturais = sum(1 for r in resultados if r["verdict"] == "NATURAL")
    suspeitas = sum(1 for r in resultados if r["verdict"] in ["SUSPEITA", "INCONCLUSIVA"])
    avg_phases = np.mean([r["phases_executed"] for r in resultados if r["phases_executed"] > 0])
    
    with col1:
        st.metric("🔴 Manipuladas", manipuladas)
    with col2:
        st.metric("🟢 Naturais", naturais)
    with col3:
        st.metric("🟡 Suspeitas", suspeitas)
    with col4:
        st.metric("⚡ Fases Médias", f"{avg_phases:.1f}")
    
    st.markdown("---")
    
    # Resultados por imagem (LIMPO)
    relatorio_dados = []
    
    for res in resultados:
        emoji = get_verdict_emoji(res["verdict"])
        
        with st.container():
            # Header compacto
            col_header1, col_header2 = st.columns([3, 1])
            with col_header1:
                st.subheader(f"{emoji} {res['nome']}")
            with col_header2:
                st.metric("Score", res["main_score"], delta=None)
            
            if res["visual_report"] is None:
                st.error(f"❌ {res['reason']}")
                st.markdown("---")
                continue
            
            # Corpo: 2 colunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(res["visual_report"], use_column_width=True)
                
                # Veredito
                if res["verdict"] == "MANIPULADA":
                    st.error(f"**{res['verdict']}** ({res['confidence']}% confiança)")
                elif res["verdict"] == "NATURAL":
                    st.success(f"**{res['verdict']}** ({res['confidence']}% confiança)")
                elif res["verdict"] == "SUSPEITA":
                    st.warning(f"**{res['verdict']}** ({res['confidence']}% confiança)")
                else:
                    st.info(f"**{res['verdict']}** ({res['confidence']}% confiança)")
                
                st.caption(res["detailed_reason"])
            
            with col2:
                st.image(res["heatmap"], use_column_width=True)
                
                # Info compacta
                if res['phases_executed'] == 1:
                    st.success("⚡ Decidido na Fase 1")
                else:
                    st.info(f"Fases: {' → '.join(res['validation_chain'])}")
                
                # Scores (se houver mais de 1)
                if len(res["all_scores"]) > 1:
                    with st.expander("📊 Scores"):
                        for k, v in res["all_scores"].items():
                            st.text(f"{k.capitalize()}: {v}")
            
            st.markdown("---")
        
        relatorio_dados.append({
            "Arquivo": res["nome"],
            "Veredito": res["verdict"],
            "Confiança (%)": res["confidence"],
            "Score": res["main_score"],
            "Fases": res["phases_executed"]
        })
    
    # Tabela resumo
    if relatorio_dados:
        st.subheader("📊 Resumo")
        df = pd.DataFrame(relatorio_dados)
        st.dataframe(df, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"relatorio_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # JSON
        json_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "versao": "4.1.0",
            "total": len(resultados),
            "manipuladas": manipuladas,
            "naturais": naturais,
            "suspeitas": suspeitas,
            "resultados": [
                {
                    "nome": r["nome"],
                    "verdict": r["verdict"],
                    "confidence": r["confidence"],
                    "score": r["main_score"],
                    "phases": r["phases_executed"]
                }
                for r in resultados
            ]
        }
        
        st.download_button(
            label="📥 Baixar JSON",
            data=json.dumps(json_data, indent=2, ensure_ascii=False),
            file_name=f"relatorio_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        return df
    return None

# Interface principal
st.markdown("### 📤 Upload de Imagens")
uploaded_files = st.file_uploader(
    "Selecione as imagens para análise",
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} imagens carregadas")
    
    if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir {arquivo.name}: {e}")
        
        if imagens:
            st.markdown("## 🔍 Resultados da Análise")
            resultados = analisar_sequencial(imagens, nomes)
            exibir_resultados(resultados)
        else:
            st.error("Nenhuma imagem válida para analisar.")
else:
    st.info("👆 Faça upload de imagens para começar")

# Rodapé minimalista
st.markdown("---")
with st.expander("📖 Interpretação dos Resultados"):
    st.markdown("""
    **Vereditos:**
    - 🔴 **MANIPULADA:** IA detectada (confiança 80-95%)
    - 🟢 **NATURAL:** Imagem autêntica (confiança 85%)
    - 🟡 **SUSPEITA:** Indicadores ambíguos (confiança 70%)
    - ⚪ **INCONCLUSIVA:** Análise complexa (confiança 60%)
    
    **Scores:**
    - **0-45:** Manipulada
    - **46-65:** Suspeita
    - **66-100:** Natural
    
    **Fases:**
    - **1 fase:** Decisão rápida (70% dos casos)
    - **2+ fases:** Validação adicional necessária
    """)

st.caption("MirrorGlass V4.1.0 | Outubro 2025 | Análise Sequencial")
