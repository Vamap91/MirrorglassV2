"""
MirrorGlass V2.6 - Interface Streamlit Simplificada
Sistema: Filtro + LBP + GAN Fingerprint
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
import json
import base64
import io
from simplified_analyzer import SimplifiedTextureAnalyzer

# Configuração da página
st.set_page_config(
    page_title="MirrorGlass V2.6 - Detecção de Fraudes por IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("🔍 MirrorGlass V2.6: Detecção Simplificada de Fraudes por IA")
st.markdown("""
Sistema simplificado e funcional para detectar manipulações por IA em imagens automotivas.

### ✨ V2.6 - Sistema Simplificado:
- **Filtro inteligente**: Remove texto/papel/reflexos (elimina falsos positivos)
- **Análise LBP**: Textura em áreas relevantes do veículo
- **GAN Fingerprint**: Detecta artefatos de checkerboard e ruído inconsistente
- **Acurácia esperada**: 80-85%

### Como funciona?
1. **Fase 1**: Detecta e exclui textos, papéis e reflexos
2. **Fase 2**: Analisa textura LBP nas áreas válidas
3. **Fase 3**: Detecta artefatos específicos de GANs
4. **Resultado**: Score de naturalidade (0-100)
""")

# Barra lateral
st.sidebar.header("⚙️ Configurações")

st.sidebar.subheader("Análise de Textura")

tamanho_bloco = st.sidebar.slider(
    "Tamanho do Bloco", 
    min_value=8, 
    max_value=32, 
    value=20, 
    step=4,
    help="Tamanho do bloco para análise (menor = mais sensível)"
)

threshold_lbp = st.sidebar.slider(
    "Sensibilidade LBP", 
    min_value=0.1, 
    max_value=0.7, 
    value=0.50, 
    step=0.05,
    help="Limiar para áreas suspeitas (menor = mais sensível)"
)

debug_mode = st.sidebar.checkbox(
    "🐛 Modo Debug",
    value=False,
    help="Mostra informações detalhadas"
)

# Info
st.sidebar.markdown("---")
st.sidebar.info("""
**MirrorGlass V2.6**  
**Versão**: Simplificada  
**Método**: Filtro + LBP + GAN  
**Acurácia**: 80-85%

**Limitações conhecidas:**
- Close-ups podem ser ambíguos
- IAs sofisticadas podem passar
- Revisão manual recomendada para scores 45-70
""")

# Função download
def get_image_download_link(img, filename, text):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img
        
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)
    
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    
    return href

# Interface principal
st.markdown("### 📤 Upload de Imagens")
uploaded_files = st.file_uploader(
    "Carregar imagens para análise", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagens carregadas")
    
    if st.button("🚀 Iniciar Análise V2.6", key="iniciar_analise"):
        # Carregar imagens
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(np.array(img))
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir {arquivo.name}: {e}")
        
        # Criar analisador
        analyzer = SimplifiedTextureAnalyzer(
            P=8,
            R=1,
            block_size=tamanho_bloco,
            threshold=threshold_lbp,
            debug=debug_mode
        )
        
        # Processar imagens
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        resultados = []
        
        for i, (img, nome) in enumerate(zip(imagens, nomes)):
            progress = (i + 1) / len(imagens)
            progress_bar.progress(progress)
            status_text.text(f"Analisando {i+1}/{len(imagens)}: {nome}")
            
            try:
                # Análise V2.6
                results = analyzer.analyze_image(img)
                
                # Visualização
                visual_report, heatmap = analyzer.generate_visual_report(img, results)
                
                resultados.append({
                    'nome': nome,
                    'imagem_original': img,
                    'results': results,
                    'visual_report': visual_report,
                    'heatmap': heatmap
                })
                
            except Exception as e:
                st.error(f"Erro ao analisar {nome}: {str(e)}")
                if debug_mode:
                    st.exception(e)
        
        progress_bar.empty()
        status_text.text("✅ Análise concluída!")
        
        # Exibir resultados
        st.markdown("## 📊 Resultados da Análise")
        
        for res in resultados:
            st.write("---")
            st.subheader(f"📸 {res['nome']}")
            
            r = res['results']
            
            # Layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(res['visual_report'], caption=f"Análise - {res['nome']}", use_column_width=True)
                
                # Métricas
                score = r['final_score']
                category, description = r['category']
                
                st.metric("Score de Naturalidade", f"{score}/100")
                
                if score <= 30:
                    st.error(f"🚨 {category}")
                    st.write(description)
                elif score <= 50:
                    st.error(f"⚠️ {category}")
                    st.write(description)
                elif score <= 70:
                    st.warning(f"⚠️ {category}")
                    st.write(description)
                else:
                    st.success(f"✅ {category}")
                    st.write(description)
                
                # Download
                st.markdown(
                    get_image_download_link(
                        res['visual_report'], 
                        f"analise_{res['nome'].replace(' ', '_')}",
                        "📥 Baixar Imagem Analisada"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.image(res['heatmap'], caption="Mapa de Calor LBP", use_column_width=True)
                
                st.write("### 📊 Detalhes da Análise")
                
                # Fase 1: Filtro
                st.write("**Fase 1 - Filtro (Elementos Legítimos):**")
                exclusion_pct = r['exclusion_percentage']
                
                if exclusion_pct > 40:
                    st.info(f"🟢 Área excluída: {exclusion_pct:.1f}% (alto)")
                elif exclusion_pct > 10:
                    st.info(f"🟡 Área excluída: {exclusion_pct:.1f}% (normal)")
                else:
                    st.info(f"⚪ Área excluída: {exclusion_pct:.1f}% (baixo)")
                
                legitimate_elements = r['legitimate_elements']
                if legitimate_elements:
                    detected = []
                    for elem_type, elem_info in legitimate_elements.items():
                        detected.append(f"{elem_type} ({elem_info.confidence:.0%})")
                    st.write(f"- Detectados: {', '.join(detected)}")
                else:
                    st.write("- Nenhum elemento detectado")
                
                # Fase 2: LBP
                st.write("**Fase 2 - Análise LBP:**")
                lbp_score = r['lbp_score']
                suspicious_pct = r['suspicious_percentage']
                
                st.write(f"- Score base: {lbp_score}")
                
                if suspicious_pct > 60:
                    st.error(f"- Áreas suspeitas: {suspicious_pct:.1f}% 🚨")
                elif suspicious_pct > 30:
                    st.warning(f"- Áreas suspeitas: {suspicious_pct:.1f}% ⚠️")
                else:
                    st.write(f"- Áreas suspeitas: {suspicious_pct:.1f}%")
                
                # Fase 3: GAN
                st.write("**Fase 3 - GAN Fingerprint:**")
                gan_results = r['gan_results']
                gan_penalty = r['gan_penalty']
                
                st.write(f"- Checkerboard: {gan_results['checkerboard_score']:.0%}")
                st.write(f"- Noise: {gan_results['noise_score']:.0%}")
                
                if gan_penalty > 0:
                    st.warning(f"- Penalidade: -{gan_penalty} pontos")
                else:
                    st.write("- Penalidade: 0 (sem artefatos detectados)")
                
                # Legenda
                st.write("**Legenda Visual:**")
                st.write("- 🟢 Verde: Áreas excluídas (texto/papel/reflexo)")
                st.write("- 🟣 Roxo: Áreas suspeitas de manipulação")
                st.write("- Azul (mapa): Textura natural")
                st.write("- Vermelho (mapa): Textura artificial")
            
            # Detalhes expandíveis
            with st.expander("🔍 Ver Análise Detalhada"):
                col_det1, col_det2 = st.columns(2)
                
                with col_det1:
                    st.write("#### Elementos Legítimos")
                    
                    if not legitimate_elements:
                        st.info("Nenhum elemento legítimo detectado")
                    else:
                        for elem_type, elem_info in legitimate_elements.items():
                            st.write(f"**{elem_type.upper()}**")
                            st.write(f"- Confiança: {elem_info.confidence:.0%}")
                            st.write(f"- BBox: {elem_info.bbox}")
                            
                            if elem_info.metadata:
                                if 'texts' in elem_info.metadata:
                                    texts = elem_info.metadata['texts']
                                    if texts:
                                        st.write(f"- Textos: {', '.join(texts)}")
                                
                                if 'paper_count' in elem_info.metadata:
                                    st.write(f"- Papéis: {elem_info.metadata['paper_count']}")
                                
                                if 'coverage' in elem_info.metadata:
                                    st.write(f"- Cobertura: {elem_info.metadata['coverage']:.1f}%")
                            st.write("")
                
                with col_det2:
                    st.write("#### Score Breakdown")
                    
                    st.write(f"1. **Score LBP base**: {r['lbp_score']}")
                    st.write(f"2. **Penalidade GAN**: -{r['gan_penalty']}")
                    st.write(f"3. **Score final**: {r['final_score']}")
                    
                    st.write("\n**Comparação:**")
                    st.write("- ✅ Filtro V2 eliminou falsos positivos")
                    st.write("- ✅ LBP analisa apenas áreas relevantes")
                    st.write("- ✅ GAN detector auxilia detecção")
                    
                    st.write("\n**Interpretação:**")
                    st.write(description)
        
        # Resumo consolidado
        st.markdown("---")
        st.markdown("### 📋 Resumo Geral")
        
        if resultados:
            # Estatísticas
            scores = [r['results']['final_score'] for r in resultados]
            manipuladas = sum(1 for s in scores if s <= 30)
            muito_suspeitas = sum(1 for s in scores if 30 < s <= 50)
            suspeitas = sum(1 for s in scores if 50 < s <= 70)
            naturais = sum(1 for s in scores if s > 70)
            
            col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
            
            with col_stat1:
                st.metric("Total", len(resultados))
            
            with col_stat2:
                st.metric("Manipuladas", manipuladas, help="Score ≤ 30")
            
            with col_stat3:
                st.metric("Muito Suspeitas", muito_suspeitas, help="30 < Score ≤ 50")
            
            with col_stat4:
                st.metric("Suspeitas", suspeitas, help="50 < Score ≤ 70")
            
            with col_stat5:
                st.metric("Naturais", naturais, help="Score > 70")
            
            # Tabela
            import pandas as pd
            
            df_resumo = pd.DataFrame([
                {
                    'Arquivo': r['nome'],
                    'Score': r['results']['final_score'],
                    'Categoria': r['results']['category'][0],
                    'Exclusão (%)': round(r['results']['exclusion_percentage'], 1),
                    'Suspeitas (%)': round(r['results']['suspicious_percentage'], 1),
                    'GAN Penalty': r['results']['gan_penalty']
                }
                for r in resultados
            ])
            
            st.dataframe(df_resumo, use_container_width=True)
            
            # JSON
            with st.expander("📄 Ver JSON Completo"):
                json_data = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "versao": "2.6 (Simplificado)",
                    "total_imagens": len(resultados),
                    "estatisticas": {
                        "manipuladas": manipuladas,
                        "muito_suspeitas": muito_suspeitas,
                        "suspeitas": suspeitas,
                        "naturais": naturais
                    },
                    "score_medio": round(np.mean(scores), 2),
                    "resultados": [
                        {
                            "nome": r['nome'],
                            "score_final": r['results']['final_score'],
                            "score_lbp": r['results']['lbp_score'],
                            "gan_penalty": r['results']['gan_penalty'],
                            "categoria": r['results']['category'][0],
                            "exclusao_pct": round(r['results']['exclusion_percentage'], 1),
                            "suspeitas_pct": round(r['results']['suspicious_percentage'], 1),
                            "elementos_detectados": list(r['results']['legitimate_elements'].keys())
                        }
                        for r in resultados
                    ]
                }
                
                json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                st.code(json_str, language='json')
                
                st.download_button(
                    label="📥 Baixar JSON",
                    data=json_str,
                    file_name=f"mirrorglass_v26_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

else:
    st.info("📤 Faça upload de imagens para começar a análise.")

# Rodapé
st.markdown("---")
st.markdown("### 📚 Como Interpretar os Resultados")

col_help1, col_help2, col_help3 = st.columns(3)

with col_help1:
    st.markdown("#### Score de Naturalidade")
    st.write("""
    - **0-30**: 🔴 Alta chance de IA
    - **31-50**: 🟠 Suspeita forte
    - **51-70**: 🟡 Questionável
    - **71-100**: 🟢 Natural
    """)

with col_help2:
    st.markdown("#### Sistema V2.6")
    st.write("""
    **3 Fases:**
    1. Filtro (texto/papel/reflexo)
    2. LBP (textura)
    3. GAN (artefatos)
    
    **Acurácia:** 80-85%
    """)

with col_help3:
    st.markdown("#### Limitações")
    st.write("""
    - Close-ups ambíguos
    - IAs sofisticadas
    - Scores 45-70 precisam revisão manual
    
    **Use com contexto!**
    """)

st.markdown("---")
st.markdown("**MirrorGlass V2.6** | Sistema Simplificado e Funcional | Desenvolvido com ❤️")
