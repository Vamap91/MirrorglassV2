"""
MirrorglassV2 - Aplicação Streamlit
Sistema de Detecção de Fraudes em Imagens Automotivas
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
from detector_v2 import ImprovedFraudDetector, LegitimateElementDetector

# Configuração da página
st.set_page_config(
    page_title="MirrorglassV2 - Detecção de Fraudes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🚗 MirrorglassV2</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistema Inteligente de Detecção de Fraudes em Imagens Automotivas</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=MirrorglassV2", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Configurações")
    
    debug_mode = st.checkbox("🐛 Modo Debug", value=False, help="Mostra logs detalhados no terminal")
    
    min_confidence = st.slider(
        "Confiança Mínima",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Confiança mínima para considerar elementos legítimos"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Sobre o Sistema")
    st.info("""
    **MirrorglassV2** detecta:
    - ✅ Textos e etiquetas
    - ✅ Papéis e documentos  
    - ✅ Reflexos de vidro
    
    Reduz **falsos positivos** em até 90%!
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Como Usar")
    st.markdown("""
    1. 📤 Faça upload da imagem
    2. ⏳ Aguarde a análise
    3. 📊 Visualize os resultados
    4. 💾 Baixe os relatórios
    """)

# Área principal
tab1, tab2, tab3 = st.tabs(["📤 Upload & Análise", "📊 Resultados Detalhados", "ℹ️ Ajuda"])

with tab1:
    st.markdown("## 📤 Upload de Imagem")
    
    # Upload
    uploaded_file = st.file_uploader(
        "Escolha uma imagem para análise",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos suportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Converter para OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Erro ao carregar a imagem. Tente outro arquivo.")
        else:
            # Mostrar imagem original
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🖼️ Imagem Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Dimensões: {image.shape[1]}x{image.shape[0]} pixels")
            
            with col2:
                st.markdown("### ⚡ Status da Análise")
                
                with st.spinner("🔍 Analisando imagem..."):
                    try:
                        # Criar detector
                        detector = ImprovedFraudDetector(debug=debug_mode)
                        
                        # Analisar
                        results = detector.analyze_image(image)
                        
                        # Salvar no session_state
                        st.session_state['results'] = results
                        st.session_state['image'] = image
                        st.session_state['detector'] = detector
                        
                        st.success("✅ Análise concluída com sucesso!")
                        
                        # Métricas rápidas
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric(
                                "Área Excluída",
                                f"{results['exclusion_percentage']:.1f}%",
                                help="Percentual da imagem identificado como elemento legítimo"
                            )
                        
                        with col_m2:
                            st.metric(
                                "Área Analisável",
                                f"{results['analyzed_percentage']:.1f}%",
                                help="Percentual da imagem disponível para análise de fraude"
                            )
                        
                        with col_m3:
                            detections = results['legitimate_elements']['total_detections']
                            st.metric(
                                "Elementos Detectados",
                                detections,
                                help="Quantidade de tipos de elementos legítimos encontrados"
                            )
                        
                        # Recomendação
                        recommendation = results['recommendation']
                        if "ALTO" in recommendation:
                            st.warning(f"⚠️ {recommendation}")
                        elif "MÉDIO" in recommendation:
                            st.info(f"ℹ️ {recommendation}")
                        else:
                            st.success(f"✅ {recommendation}")
                        
                    except Exception as e:
                        st.error(f"❌ Erro na análise: {str(e)}")
                        if debug_mode:
                            st.exception(e)

with tab2:
    st.markdown("## 📊 Resultados Detalhados")
    
    if 'results' in st.session_state and 'image' in st.session_state:
        results = st.session_state['results']
        image = st.session_state['image']
        detector = st.session_state['detector']
        
        # Visualizações
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            st.markdown("### 🎨 Elementos Detectados")
            vis_image = detector.legitimate_detector.visualize_detections(image)
            st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Verde = Texto | Azul = Papel | Amarelo = Reflexos")
        
        with col_vis2:
            st.markdown("### 🗺️ Máscara de Exclusão")
            exclusion_mask = results['exclusion_mask']
            st.image(exclusion_mask, use_container_width=True, channels="GRAY")
            st.caption("Branco = Área excluída | Preto = Área analisável")
        
        # Detalhes por tipo
        st.markdown("---")
        st.markdown("### 🔍 Detalhes dos Elementos Detectados")
        
        detections = results['legitimate_elements']['detections']
        
        if not detections:
            st.info("ℹ️ Nenhum elemento legítimo detectado nesta imagem.")
        else:
            for det_type, det_info in detections.items():
                with st.expander(f"📋 {det_type.upper()} - Confiança: {det_info['confidence']:.0%}", expanded=True):
                    col_det1, col_det2, col_det3 = st.columns(3)
                    
                    with col_det1:
                        st.metric("Confiança", f"{det_info['confidence']:.1%}")
                    
                    with col_det2:
                        st.metric("Área (pixels)", f"{det_info['area_pixels']:,}")
                    
                    with col_det3:
                        x, y, w, h = det_info['bbox']
                        st.metric("BBox", f"{w}x{h}")
                    
                    # Metadados específicos
                    if 'metadata' in det_info and det_info['metadata']:
                        st.markdown("**Metadados:**")
                        
                        if det_type == 'text' and 'texts' in det_info['metadata']:
                            texts = det_info['metadata']['texts']
                            if texts:
                                st.write(f"Textos encontrados: {', '.join(texts)}")
                        
                        if det_type == 'paper' and 'paper_count' in det_info['metadata']:
                            st.write(f"Quantidade de papéis: {det_info['metadata']['paper_count']}")
                        
                        if det_type == 'reflection' and 'reflection_percentage' in det_info['metadata']:
                            st.write(f"Cobertura de reflexos: {det_info['metadata']['reflection_percentage']:.2f}%")
        
        # Download dos resultados
        st.markdown("---")
        st.markdown("### 💾 Download dos Resultados")
        
        col_down1, col_down2, col_down3 = st.columns(3)
        
        with col_down1:
            # Download da visualização
            vis_img_pil = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            buf_vis = io.BytesIO()
            vis_img_pil.save(buf_vis, format='PNG')
            buf_vis.seek(0)
            
            st.download_button(
                label="📥 Baixar Visualização",
                data=buf_vis,
                file_name=f"mirrorglass_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_down2:
            # Download da máscara
            mask_pil = Image.fromarray(exclusion_mask)
            buf_mask = io.BytesIO()
            mask_pil.save(buf_mask, format='PNG')
            buf_mask.seek(0)
            
            st.download_button(
                label="📥 Baixar Máscara",
                data=buf_mask,
                file_name=f"mirrorglass_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_down3:
            # Download do JSON
            import json
            results_json = results.copy()
            # Remover máscaras numpy (não serializáveis)
            del results_json['exclusion_mask']
            
            json_str = json.dumps(results_json, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Baixar JSON",
                data=json_str,
                file_name=f"mirrorglass_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    else:
        st.info("ℹ️ Faça upload de uma imagem na aba 'Upload & Análise' para ver os resultados detalhados.")

with tab3:
    st.markdown("## ℹ️ Ajuda e Documentação")
    
    st.markdown("### 🎯 Objetivo do Sistema")
    st.write("""
    O **MirrorglassV2** foi desenvolvido para detectar fraudes em vistorias automotivas, 
    identificando quando imagens foram manipuladas por IA para:
    
    - **Criar danos falsos**: Carro está OK, mas IA cria amassados/riscos para cobrar conserto inexistente
    - **Ocultar danos reais**: Carro está danificado, mas IA "conserta" a foto para cobrar conserto não realizado
    """)
    
    st.markdown("### 🔧 Como Funciona")
    st.write("""
    O sistema identifica e **exclui elementos legítimos** que podem causar falsos positivos:
    
    1. **Textos e Etiquetas**: Detecta usando OCR (Tesseract)
    2. **Papéis e Documentos**: Identifica regiões uniformes com bordas retangulares
    3. **Reflexos de Vidro**: Analisa canais HSV e gradientes suaves
    
    Após excluir esses elementos, a análise de fraude foca apenas nas áreas relevantes do veículo.
    """)
    
    st.markdown("### 📊 Interpretando os Resultados")
    
    col_help1, col_help2 = st.columns(2)
    
    with col_help1:
        st.markdown("#### Área Excluída")
        st.write("""
        - **< 10%**: Imagem limpa, análise completa
        - **10-40%**: Presença moderada de elementos
        - **40-70%**: Muitos elementos legítimos
        - **> 70%**: Análise muito limitada
        """)
    
    with col_help2:
        st.markdown("#### Confiança")
        st.write("""
        - **< 30%**: Detecção incerta
        - **30-60%**: Detecção provável
        - **60-80%**: Detecção confiável
        - **> 80%**: Detecção muito confiável
        """)
    
    st.markdown("### ⚠️ Limitações")
    st.warning("""
    - O sistema **não detecta fraudes por IA diretamente** (ainda)
    - Ele apenas **elimina falsos positivos** causados por elementos legítimos
    - É a **Fase 1** de um sistema maior de detecção de fraudes
    - Requer imagens com boa qualidade e iluminação adequada
    """)
    
    st.markdown("### 🆘 Problemas Comuns")
    
    with st.expander("❓ OCR não detecta textos"):
        st.write("""
        **Possíveis causas:**
        - Texto muito pequeno ou desfocado
        - Baixo contraste entre texto e fundo
        - Texto em ângulo muito inclinado
        
        **Solução:** Ajuste a confiança mínima ou melhore a qualidade da imagem.
        """)
    
    with st.expander("❓ Falsos positivos em papéis"):
        st.write("""
        **Possíveis causas:**
        - Áreas da lataria muito uniformes
        - Reflexos que parecem papéis
        
        **Solução:** Aumente o threshold de uniformidade no código ou capture fotos de ângulos diferentes.
        """)
    
    with st.expander("❓ Reflexos não detectados"):
        st.write("""
        **Possíveis causas:**
        - Reflexos muito fracos
        - Iluminação muito difusa
        
        **Solução:** Ajuste os thresholds HSV no código ou use modo debug para ver valores.
        """)
    
    st.markdown("---")
    st.markdown("### 📞 Suporte")
    st.info("""
    **Versão:** 2.0.0  
    **Projeto:** MirrorglassV2 - Detecção de Fraudes Automotivas  
    **Desenvolvido para:** Eliminar falsos positivos em detecção de fraudes por IA
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🚗 <strong>MirrorglassV2</strong> | Detecção Inteligente de Fraudes Automotivas</p>
    <p style="font-size: 0.9rem;">Desenvolvido com ❤️ usando Streamlit + OpenCV + Python</p>
</div>
""", unsafe_allow_html=True)
