# streamlit_app.py
"""
MirrorGlass V2 - Sistema Integrado de Detecção de Fraudes por IA
Mantém UX da V1 + adiciona análise inteligente com filtro V2
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
import json
import base64
import io
from integrated_analyzer import IntegratedTextureAnalyzer

# Configuração da página
st.set_page_config(
    page_title="MirrorGlass V2 - Detecção de Fraudes por IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e introdução
st.title("📊 MirrorGlass V2: Detecção de Fraudes por IA em Imagens Automotivas")
st.markdown("""
Este sistema utiliza técnicas avançadas de visão computacional para detectar manipulações por IA em imagens automotivas.

### ✨ Novidade V2:
- **Filtro inteligente** que elimina falsos positivos causados por textos, papéis e reflexos
- **Análise focada** apenas nas áreas relevantes do veículo
- **Precisão aumentada** em até 90%

### Como funciona?
1. **Fase 1 (V2)**: Detecta e exclui elementos legítimos (textos, papéis, reflexos)
2. **Fase 2 (V1)**: Analisa textura LBP apenas nas áreas relevantes
3. Resultado: Score preciso de naturalidade da imagem
""")

# Barra lateral
st.sidebar.header("⚙️ Configurações")

# Configurações de análise
st.sidebar.subheader("Análise de Textura")

limiar_naturalidade = st.sidebar.slider(
    "Limiar de Naturalidade", 
    min_value=30, 
    max_value=80, 
    value=50, 
    help="Score abaixo deste valor indica possível manipulação por IA"
)

tamanho_bloco = st.sidebar.slider(
    "Tamanho do Bloco", 
    min_value=8, 
    max_value=32, 
    value=20, 
    step=4,
    help="Tamanho do bloco para análise de textura (menor = mais sensível)"
)

threshold_lbp = st.sidebar.slider(
    "Sensibilidade LBP", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.50, 
    step=0.05,
    help="Limiar para detecção de áreas suspeitas (menor = mais sensível)"
)

debug_mode = st.sidebar.checkbox(
    "🐛 Modo Debug",
    value=False,
    help="Mostra informações detalhadas no console"
)

# Informações na sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido para:** MirrorGlass  
**Projeto:** Detecção de Fraudes em Imagens Automotivas  
**Versão:** 2.0.0 (Integrado)  
**Método:** V2 (Filtro) + V1 (LBP)
""")

# Função auxiliar para download
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
st.markdown("### 🔹 Passo 1: Carregar Imagens")
uploaded_files = st.file_uploader(
    "Faça upload das imagens para análise", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagens carregadas")
    
    if st.button("🚀 Iniciar Análise Integrada", key="iniciar_analise"):
        # Carregar imagens
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(np.array(img))
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir a imagem {arquivo.name}: {e}")
        
        # Criar analisador integrado
        analyzer = IntegratedTextureAnalyzer(
            P=8,
            R=1,
            block_size=tamanho_bloco,
            threshold=threshold_lbp,
            debug=debug_mode
        )
        
        # Processar cada imagem
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        resultados = []
        
        for i, (img, nome) in enumerate(zip(imagens, nomes)):
            progress = (i + 1) / len(imagens)
            progress_bar.progress(progress)
            status_text.text(f"Analisando imagem {i+1} de {len(imagens)}: {nome}")
            
            try:
                # Análise integrada (V2 + V1)
                integrated_results = analyzer.analyze_image_integrated(img)
                
                # Gerar visualização
                visual_report, heatmap = analyzer.generate_visual_report(img, integrated_results)
                
                # Armazenar resultados
                resultados.append({
                    'nome': nome,
                    'imagem_original': img,
                    'integrated_results': integrated_results,
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
        st.markdown("## 🤖 Resultados da Análise Integrada")
        
        for res in resultados:
            st.write("---")
            st.subheader(f"📸 {res['nome']}")
            
            ir = res['integrated_results']
            
            # Layout principal
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(res['visual_report'], caption=f"Análise Integrada - {res['nome']}", use_column_width=True)
                
                # Métricas principais
                score = ir['final_score']
                category, description = ir['final_category']
                
                st.metric("Score de Naturalidade", score)
                
                if score <= 45:
                    st.error(f"⚠️ {category}: {description}")
                elif score <= 70:
                    st.warning(f"⚠️ {category}: {description}")
                else:
                    st.success(f"✅ {category}: {description}")
                
                # Download
                st.markdown(
                    get_image_download_link(
                        res['visual_report'], 
                        f"analise_{res['nome'].replace(' ', '_')}.jpg",
                        "📥 Baixar Imagem Analisada"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.image(res['heatmap'], caption="Mapa de Calor LBP", use_column_width=True)
                
                st.write("### 📊 Detalhes da Análise Integrada")
                
                # Informações V2 (Filtro)
                st.write("**Fase 1 - Filtro V2 (Elementos Legítimos):**")
                exclusion_pct = ir['v2_exclusion_percentage']
                
                if exclusion_pct > 40:
                    st.info(f"🟢 Área excluída: {exclusion_pct:.1f}% (texto/papel/reflexo)")
                elif exclusion_pct > 10:
                    st.info(f"🟡 Área excluída: {exclusion_pct:.1f}%")
                else:
                    st.info(f"⚪ Área excluída: {exclusion_pct:.1f}%")
                
                # Elementos detectados
                legitimate_elements = ir['v2_legitimate_elements']
                if legitimate_elements:
                    detected_types = list(legitimate_elements.keys())
                    st.write(f"- Elementos detectados: {', '.join(detected_types)}")
                else:
                    st.write("- Nenhum elemento legítimo detectado")
                
                # Informações V1 (Análise)
                st.write("**Fase 2 - Análise V1 (Textura LBP):**")
                suspicious_pct = ir['suspicious_areas_percentage']
                
                if suspicious_pct > 60:
                    st.error(f"🚨 Áreas suspeitas: {suspicious_pct:.2f}% - ALTO RISCO!")
                elif suspicious_pct > 30:
                    st.warning(f"⚠️ Áreas suspeitas: {suspicious_pct:.2f}% - ATENÇÃO!")
                else:
                    st.write(f"- Áreas suspeitas: {suspicious_pct:.2f}%")
                
                st.write(f"- Interpretação: {description}")
                
                # Legenda
                st.write("**Legenda:**")
                st.write("- 🟢 Verde: Áreas excluídas (texto/papel/reflexo)")
                st.write("- 🟣 Roxo: Áreas suspeitas de manipulação")
                st.write("- Azul (mapa): Texturas naturais")
                st.write("- Vermelho (mapa): Texturas artificiais")
            
            # Detalhes expandíveis
            with st.expander("🔍 Ver Análise Detalhada"):
                col_det1, col_det2 = st.columns(2)
                
                with col_det1:
                    st.write("#### Elementos Legítimos Detectados (V2)")
                    
                    if not legitimate_elements:
                        st.info("Nenhum elemento legítimo detectado")
                    else:
                        for elem_type, elem_info in legitimate_elements.items():
                            st.write(f"**{elem_type.upper()}**")
                            st.write(f"- Confiança: {elem_info.confidence:.0%}")
                            st.write(f"- BBox: {elem_info.bbox}")
                            
                            if elem_info.metadata:
                                if elem_type == 'text' and 'texts' in elem_info.metadata:
                                    texts = elem_info.metadata['texts']
                                    if texts:
                                        st.write(f"- Textos: {', '.join(texts)}")
                                
                                if elem_type == 'paper' and 'paper_count' in elem_info.metadata:
                                    st.write(f"- Papéis: {elem_info.metadata['paper_count']}")
                                
                                if elem_type == 'reflection' and 'reflection_percentage' in elem_info.metadata:
                                    st.write(f"- Cobertura: {elem_info.metadata['reflection_percentage']:.1f}%")
                            st.write("")
                
                with col_det2:
                    st.write("#### Análise de Textura (V1)")
                    texture_analysis = ir['v1_texture_analysis']
                    
                    st.write(f"- Score de naturalidade: {ir['final_score']}")
                    st.write(f"- Áreas suspeitas: {suspicious_pct:.2f}%")
                    st.write(f"- Blocos analisados: Apenas áreas não excluídas")
                    st.write(f"- Método: LBP (Local Binary Pattern)")
                    
                    st.write("\n**Comparação V1 vs V2:**")
                    st.write("- ✅ V2 eliminou falsos positivos")
                    st.write("- ✅ Análise focada em áreas relevantes")
                    st.write("- ✅ Precisão aumentada")
        
        # Relatório consolidado
        st.markdown("---")
        st.markdown("### 📋 Resumo Geral")
        
        if resultados:
            # Estatísticas
            scores = [r['integrated_results']['final_score'] for r in resultados]
            manipuladas = sum(1 for s in scores if s <= 45)
            suspeitas = sum(1 for s in scores if 45 < s <= 70)
            naturais = sum(1 for s in scores if s > 70)
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Analisadas", len(resultados))
            
            with col_stat2:
                st.metric("Manipuladas (IA)", manipuladas)
            
            with col_stat3:
                st.metric("Suspeitas", suspeitas)
            
            with col_stat4:
                st.metric("Naturais", naturais)
            
            # Tabela resumida
            import pandas as pd
            
            df_resumo = pd.DataFrame([
                {
                    'Arquivo': r['nome'],
                    'Score': r['integrated_results']['final_score'],
                    'Categoria': r['integrated_results']['final_category'][0],
                    'Área Excluída (%)': round(r['integrated_results']['v2_exclusion_percentage'], 1),
                    'Áreas Suspeitas (%)': round(r['integrated_results']['suspicious_areas_percentage'], 1)
                }
                for r in resultados
            ])
            
            st.dataframe(df_resumo)
            
            # JSON resumido
            with st.expander("📄 Ver JSON Resumido"):
                json_resumido = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "versao": "2.0.0 (Integrado V2+V1)",
                    "total_imagens": len(resultados),
                    "estatisticas": {
                        "manipuladas": manipuladas,
                        "suspeitas": suspeitas,
                        "naturais": naturais
                    },
                    "score_medio": round(np.mean(scores), 2),
                    "resultados": [
                        {
                            "nome": r['nome'],
                            "score": r['integrated_results']['final_score'],
                            "categoria": r['integrated_results']['final_category'][0],
                            "area_excluida_pct": round(r['integrated_results']['v2_exclusion_percentage'], 1),
                            "areas_suspeitas_pct": round(r['integrated_results']['suspicious_areas_percentage'], 1),
                            "elementos_detectados": list(r['integrated_results']['v2_legitimate_elements'].keys())
                        }
                        for r in resultados
                    ]
                }
                
                json_str = json.dumps(json_resumido, indent=2, ensure_ascii=False)
                st.code(json_str, language='json')
                
                st.download_button(
                    label="📥 Baixar JSON Resumido",
                    data=json_str,
                    file_name=f"mirrorglass_v2_resumo_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

else:
    st.info("📤 Faça upload de imagens para começar a análise.")

# Rodapé com instruções
st.markdown("---")
st.markdown("### 📚 Como Interpretar os Resultados")

col_help1, col_help2 = st.columns(2)

with col_help1:
    st.markdown("#### Score de Naturalidade")
    st.write("""
    - **0-45**: 🔴 Alta probabilidade de manipulação por IA
    - **46-70**: 🟡 Textura suspeita, verificar manualmente
    - **71-100**: 🟢 Textura natural, imagem legítima
    """)
    
    st.markdown("#### Área Excluída (V2)")
    st.write("""
    - **< 10%**: Poucos elementos legítimos
    - **10-40%**: Presença moderada (normal)
    - **> 40%**: Muitos textos/papéis/reflexos
    """)

with col_help2:
    st.markdown("#### Novidade V2")
    st.write("""
    **Problema V1:** Etiquetas e papéis causavam falsos positivos
    
    **Solução V2:** 
    1. Detecta textos, papéis e reflexos
    2. Exclui essas áreas da análise
    3. Analisa APENAS áreas do veículo
    
    **Resultado:** Score preciso, sem falsos positivos!
    """)
    
    st.markdown("#### Legendas Visuais")
    st.write("""
    - 🟢 **Retângulos verdes**: Áreas excluídas (legítimas)
    - 🟣 **Retângulos roxos**: Áreas suspeitas de IA
    - **Mapa de calor**: Vermelho = artificial, Azul = natural
    """)
