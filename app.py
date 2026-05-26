import os
# Rezolvarea conflictului de librării OpenMP (OMP: Error #15) pe Mac local pentru PyTorch + FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import streamlit.components.v1 as components
import shutil
import torch
torch.classes.__path__ = [] 
import pickle
import numpy as np
from pathlib import Path
from code_parser import (
    unzip_project, 
    scan_project_files, 
    build_file_tree, 
    parse_and_chunk_file,
    generate_uml_class_diagram,
    generate_dependency_diagram
)
from vector_store import CodeBERTIndexer, DEVICE

# Configurare pagină Streamlit
st.set_page_config(
    page_title="AI Codebase Explainer - Local",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Încărcare și injectare CSS customizat pentru design premium
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css("style.css")

# Directorul temporar pentru proiectul încărcat
TEMP_DIR = Path("./temp_project_workspace")

# Inițializare stări în session_state dacă nu există
if "project_processed" not in st.session_state:
    st.session_state.project_processed = False
if "file_tree" not in st.session_state:
    st.session_state.file_tree = None
if "files_list" not in st.session_state:
    st.session_state.files_list = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "microsoft/codebert-base"
if "indexer" not in st.session_state:
    st.session_state.indexer = CodeBERTIndexer(st.session_state.selected_model)
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "uml_diagram" not in st.session_state:
    st.session_state.uml_diagram = ""
if "dependency_diagram" not in st.session_state:
    st.session_state.dependency_diagram = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_tokens" not in st.session_state:
    st.session_state.search_tokens = None
if "search_query_cached" not in st.session_state:
    st.session_state.search_query_cached = ""

def clean_workspace():
    """Curăță fișierele anterioare și stările salvate."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    st.session_state.project_processed = False
    st.session_state.file_tree = None
    st.session_state.files_list = []
    st.session_state.chunks = []
    st.session_state.stats = {}
    st.session_state.selected_file = None
    st.session_state.uml_diagram = ""
    st.session_state.dependency_diagram = ""
    st.session_state.search_results = None
    st.session_state.search_tokens = None
    st.session_state.search_query_cached = ""
    st.session_state.indexer = CodeBERTIndexer(st.session_state.selected_model)

def render_mermaid(mermaid_code, height=500):
    """
    Randează un diagramă Mermaid.js într-un element iframe securizat și personalizat,
    cu suport complet pentru zoom interactiv (wheel), pan (drag) și butoane de control.
    """
    # Escapăm caractere care ar putea sparge stringul javascript
    escaped_code = mermaid_code.replace("`", "\\`").replace("${", "\\${")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body, html {{
            background-color: #0d1117;
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
          }}
          #diagram-container {{
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
          }}
          svg {{
            width: 100% !important;
            height: 100% !important;
          }}
        </style>
        <!-- Încărcăm biblioteca oficială svg-pan-zoom pentru pan & zoom interactiv -->
        <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
      </head>
      <body>
        <div id="diagram-container">
          <div id="loading" style="color: #94a3b8; font-family: sans-serif; text-align: center; margin-top: 20%;">Se randează diagrama...</div>
        </div>
        
        <!-- Script ascuns în care punem definiția Mermaid -->
        <script id="mermaid-data" type="text/plain">{escaped_code}</script>
        
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          
          mermaid.initialize({{
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
            themeVariables: {{
              background: '#0d1117',
              primaryColor: '#8b5cf6',
              primaryTextColor: '#c9d1d9',
              lineColor: '#38bdf8',
              secondaryColor: '#1e293b'
            }}
          }});
          
          async function initDiagram() {{
            try {{
              const container = document.getElementById('diagram-container');
              const code = document.getElementById('mermaid-data').textContent;
              
              // Randăm manual codul Mermaid în SVG
              const {{ svg }} = await mermaid.render('mermaid-svg', code);
              container.innerHTML = svg;
              
              const svgElement = container.querySelector('svg');
              svgElement.setAttribute('id', 'rendered-svg');
              svgElement.style.width = '100%';
              svgElement.style.height = '100%';
              
              // Atașăm motorul de pan & zoom
              svgPanZoom('#rendered-svg', {{
                zoomEnabled: true,
                controlIconsEnabled: true,
                fit: true,
                center: true,
                minZoom: 0.1,
                maxZoom: 10,
                zoomScaleFactor: 0.15
              }});
            }} catch (error) {{
              console.error(error);
              document.getElementById('diagram-container').innerHTML = 
                `<div style="color: #ef4444; font-family: sans-serif; padding: 20px;">
                  Eroare randare diagramă: Moștenirea sau conexiunea conține elemente neacceptate în sintaxă.
                 </div>`;
            }}
          }}
          
          // Pornim randarea după ce pagina s-a încărcat
          window.addEventListener('load', initDiagram);
        </script>
      </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<h1 class="neon-glow-header">Code Explainer</h1>', unsafe_allow_html=True)
    st.markdown("Analiză structurală și căutare semantică **100% Offline** bazată pe **Transformer (CodeBERT)**.")
    st.write("---")
    
    # Afișăm statusul hardware acceleration
    device_label = "Apple Silicon (MPS)" if "mps" in str(DEVICE).lower() else "GPU (CUDA)" if "cuda" in str(DEVICE).lower() else "CPU local"
    st.markdown(f"**Dispozitiv Transformer:**\n`{device_label}`")
    
    st.write("---")
    
    # Alegere Model Transformer (Wow and customizability feature)
    st.markdown("### Model Transformer")
    selected_model_name = st.selectbox(
        "Alege modelul de vectorizare:",
        [
            "microsoft/codebert-base",
            "microsoft/graphcodebert-base",
            "sentence-transformers/all-MiniLM-L6-v2"
        ],
        index=0,
        help="CodeBERT este optimizat pentru cod. GraphCodeBERT folosește fluxul de date structural. MiniLM este ultra-rapid și mic."
    )
    
    # Dacă modelul se schimbă, resetăm indexul deoarece dimensiunile și spațiile vectoriale diferă
    if selected_model_name != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_name
        st.session_state.indexer = CodeBERTIndexer(selected_model_name)
        st.session_state.project_processed = False
        st.session_state.search_results = None
        st.session_state.search_tokens = None
        st.warning("Modelul s-a schimbat! Re-vectorizați proiectul folosind noul Transformer.")
        
    st.write("---")
    
    # Încărcare proiect ZIP
    st.markdown("### Încărcare Proiect")
    uploaded_file = st.file_uploader("Încarcă o arhivă .zip a proiectului", type=["zip"])
    
    # Buton de Reset
    if st.button("Șterge datele / Încarcă alt proiect", use_container_width=True):
        clean_workspace()
        st.rerun()

# ----------------- PROCESARE COD & EMBEDDINGS -----------------
if uploaded_file is not None and not st.session_state.project_processed:
    with st.spinner("Se extrage arhiva proiectului..."):
        clean_workspace()
        zip_bytes = uploaded_file.read()
        unzip_project(zip_bytes, TEMP_DIR)
        
    # Scanare fișiere proiect
    all_files = scan_project_files(TEMP_DIR)
    
    if not all_files:
        st.error("Nu s-au găsit fișiere de cod acceptate în arhivă!")
    else:
        st.session_state.files_list = all_files
        st.session_state.file_tree = build_file_tree(TEMP_DIR)
        
        # Generare statistici proiect
        total_files = len(all_files)
        total_lines = 0
        extensions = {}
        
        for file in all_files:
            suffix = file.suffix.lower()
            extensions[suffix] = extensions.get(suffix, 0) + 1
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines += len(f.readlines())
            except:
                pass
                
        st.session_state.stats = {
            "total_files": total_files,
            "total_lines": total_lines,
            "extensions": extensions
        }
        
        # Parsare și Chunking fișiere
        chunks = []
        with st.spinner("Se parsează fișierele și se extrag metadatele AST..."):
            for file in all_files:
                file_chunks = parse_and_chunk_file(file, TEMP_DIR)
                chunks.extend(file_chunks)
        st.session_state.chunks = chunks
        
        # Afișăm detaliile despre procesul de vectorizare cu CodeBERT
        st.info(f"S-au extras {len(chunks)} blocuri logice de cod din proiect. Începem procesul de vectorizare cu modelul Transformer CodeBERT...")
        
        # Vectorizare cu CodeBERT (PyTorch) + FAISS
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        def update_progress(percentage, text):
            progress_bar.progress(percentage)
            progress_text.text(text)
            
        try:
            indexer = st.session_state.indexer
            indexer.build_index(chunks, progress_bar_callback=update_progress)
            indexer.save_index(TEMP_DIR)
            
            progress_bar.empty()
            progress_text.empty()
            
            # Generăm diagramele Mermaid deterministic (offline prin AST)
            st.session_state.uml_diagram = generate_uml_class_diagram(all_files, TEMP_DIR)
            st.session_state.dependency_diagram = generate_dependency_diagram(all_files, TEMP_DIR)
            
            st.success("Analiza semantică și structurală a codebase-ului s-a încheiat cu succes!")
            st.session_state.project_processed = True
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Eroare la indexare/vectorizare CodeBERT: {str(e)}")
            st.session_state.project_processed = False

# ----------------- MAIN UI -----------------
if not st.session_state.project_processed:
    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>Bun venit la AI Codebase Explainer (Local)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Încarcă un fișier <b>.zip</b> al proiectului în sidebar pentru a începe analiza structurală complet locală.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <h3>CodeBERT Transformer</h3>
            <p>Rulează modelul pre-antrenat local pentru a genera reprezentări dense de 768-D pentru fiecare metodă, clasă sau funcție.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <h3>Căutare Semantică FAISS</h3>
            <p>Interogare instantanee în cod pe bază de concepte, nu doar cuvinte cheie rigide, folosind indexarea similarității cosine.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <h3>Parsare AST Deterministică</h3>
            <p>Generează automat diagrame de clase UML reale și diagrame de conexiuni între module prin analiza arborelui sintactic.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    tab1, tab2, tab3 = st.tabs([
        "Dashboard & Explorator Cod", 
        "Arhitectură UML & Relații", 
        "Căutare Semantică în Proiect"
    ])
    
    # ----------------- TAB 1: DASHBOARD & CODE VIEW -----------------
    with tab1:
        st.markdown("## Dashboard Proiect")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Fișiere de Cod", st.session_state.stats["total_files"])
        with c2:
            st.metric("Linii Totale de Cod", f"{st.session_state.stats['total_lines']:,}")
        with c3:
            st.metric("Fragmente Analizate (AST Chunks)", len(st.session_state.chunks))
        with c4:
            st.metric("Dispozitiv Utilizat", str(DEVICE).upper())
            
        st.write("---")
        
        # Specificații Tehnice Transformer (Wow Academic highlight)
        with st.expander("Specificații Tehnice Detaliate: Model Transformer CodeBERT", expanded=False):
            st.markdown("""
            <div style="background: rgba(13, 17, 23, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.25); margin-bottom: 20px;">
                <h4 style="color: #38bdf8; margin-top:0; font-weight:700;">Arhitectura Rețelei CodeBERT (Microsoft Pretrained Transformer)</h4>
                <p>Modelul utilizat în mod direct pentru înțelegerea proiectului dvs. folosește o arhitectură bazată pe <b>Transformer Encoder</b> (similară cu RoBERTa) antrenată bimodal pe limbaj natural și cod sursă.</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                        <b>Parametrii Generali ai Rețelei:</b>
                        <ul style="margin-top: 5px; padding-left: 20px;">
                            <li><b>Straturi (Transformer Layers):</b> 12 straturi dense</li>
                            <li><b>Capete de Atenție (Attention Heads):</b> 12 per strat</li>
                            <li><b>Dimensiune Ascunsă (Hidden Size):</b> 768 dimensiuni</li>
                            <li><b>Dimensiune FFN:</b> 3072 dimensiuni</li>
                        </ul>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                        <b>Tokenizare și Intrare:</b>
                        <ul style="margin-top: 5px; padding-left: 20px;">
                            <li><b>Vocabular Model:</b> 50,265 tokeni unici</li>
                            <li><b>Tokenizare:</b> Byte-Pair Encoding (BPE)</li>
                            <li><b>Fereastră Context:</b> 512 tokeni</li>
                            <li><b>Parametri Totali:</b> ~125 Milioane</li>
                        </ul>
                    </div>
                </div>
                <p style="font-size: 0.95em; color: #94a3b8; margin-bottom:0; font-style: italic;">Aplicația noastră aplică o operație matematică de <b>Mean Pooling</b> pe tensorul de ieșire al Transformer-ului, calculând reprezentarea vectorială medie a tuturor tokenilor, ponderată prin masca de atenție.</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### Structură Fișiere")
            st.write("Alege un fișier din listă pentru a-l vizualiza în panoul din dreapta:")
            
            rel_files = [str(f.relative_to(TEMP_DIR)) for f in st.session_state.files_list]
            selected_rel_path = st.selectbox("Alege fișierul:", sorted(rel_files))
            
            if selected_rel_path:
                st.session_state.selected_file = selected_rel_path
                
            st.markdown("#### Tehnologii Detectate:")
            for ext, count in st.session_state.stats["extensions"].items():
                st.markdown(f'<span class="custom-badge custom-badge-blue"><b>{ext}</b> ({count} fișiere)</span>', unsafe_allow_html=True)
                
        with col_right:
            if st.session_state.selected_file:
                st.markdown(f"### Vizualizare: `{st.session_state.selected_file}`")
                
                full_file_path = TEMP_DIR / st.session_state.selected_file
                try:
                    with open(full_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_code = f.read()
                        
                    suffix = full_file_path.suffix.lower()
                    lang_map = {
                        '.py': 'python', '.js': 'javascript', '.ts': 'typescript', 
                        '.tsx': 'typescript', '.jsx': 'javascript', '.html': 'html', 
                        '.css': 'css', '.java': 'java', '.cpp': 'cpp', '.cs': 'csharp',
                        '.sh': 'bash', '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json'
                    }
                    syntax_lang = lang_map.get(suffix, 'text')
                    st.code(file_code, language=syntax_lang, line_numbers=True)
                except Exception as e:
                    st.error(f"Nu s-a putut citi fișierul: {str(e)}")
            else:
                st.info("Selectați un fișier din stânga pentru a-i vedea conținutul cu evidențierea sintaxei.")

    # ----------------- TAB 2: ARHITECTURĂ UML (AST DRIVEN) -----------------
    with tab2:
        st.markdown("## Analiză Arhitecturală Structurală (Fără AI Extern)")
        st.markdown("Diagrama UML și relațiile de import de mai jos sunt generate în timp real, analizând **determinismul sintactic al codului dumneavoastră (AST)**. Ele vor fi întotdeauna 100% fidele realității.")
        
        # Slider pentru redimensionarea diagramelor
        diag_height = st.slider("Ajustează înălțimea diagramelor (pixeli):", min_value=300, max_value=1200, value=500, step=50)
        st.write("---")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("### Diagramă de Clase (UML Class Diagram)")
            st.write("Harta moștenirilor, claselor și metodelor extrase din AST:")
            render_mermaid(st.session_state.uml_diagram, height=diag_height)
            with st.expander("Codul sursă Mermaid UML:"):
                st.code(st.session_state.uml_diagram, language="mermaid")
                
        with diag_col2:
            st.markdown("### Harta Dependențelor între Module")
            st.write("Relațiile de import și dependințele de fișiere detectate:")
            render_mermaid(st.session_state.dependency_diagram, height=diag_height)
            with st.expander("Codul sursă Mermaid Module:"):
                st.code(st.session_state.dependency_diagram, language="mermaid")

    # ----------------- TAB 3: CĂUTARE SEMANTICĂ LOCALĂ -----------------
    with tab3:
        sub_tab_search, sub_tab_attention = st.tabs([
            "Căutare Semantică în Proiect",
            "Explorator Atenție CodeBERT (Attention Heatmap)"
        ])
        
        # --- SUB-TAB 3.1: CĂUTARE SEMANTICĂ ---
        with sub_tab_search:
            st.markdown("## Căutare Semantică Locală (CodeBERT + FAISS)")
            st.markdown("Introduceți un concept de programare, o sarcină sau un nume de funcție pe care doriți să îl găsiți (ex: *'criptează parola'*, *'socket connection'*, *'multi-threading'*, *'trimite e-mail'*). Modelul **CodeBERT** va analiza contextul semantic din spate și va localiza codul potrivit.")
            st.write("---")
            
            # Folosim un formular Streamlit (st.form) pentru a preveni re-rularea modelului Transformer la fiecare interacțiune
            with st.form("search_form"):
                query_input = st.text_input(
                    "Ce dorești să cauți în proiectul tău?", 
                    value=st.session_state.search_query_cached,
                    placeholder="ex: conexiune socket, thread-ul clientului, salvare baza de date..."
                )
                submitted = st.form_submit_button("Caută în codebase")
                
            if submitted and query_input:
                with st.spinner("Modelul Transformer vectorizează textul și caută în FAISS..."):
                    try:
                        indexer = st.session_state.indexer
                        tokens, ids = indexer.tokenize(query_input)
                        results = indexer.search(query_input, top_k=4)
                        
                        # Salvare în cache-ul session_state
                        st.session_state.search_query_cached = query_input
                        st.session_state.search_results = results
                        st.session_state.search_tokens = (tokens, ids)
                    except Exception as e:
                        st.error(f"Eroare la rularea interogării CodeBERT: {str(e)}")
                        
            # Afișăm rezultatele stocate în cache (astfel încât expander-ele să funcționeze instantaneu fără re-calculare)
            if st.session_state.search_results is not None:
                # 1. Vizualizator Tokeni (Transformer Tokenization Inspector)
                tokens, ids = st.session_state.search_tokens
                with st.expander("Tokenization Inspector (Cum descompune Transformer-ul căutarea ta):", expanded=True):
                    st.markdown("**Segmentare cuvinte în tokeni sub-word și asocieri de Vocabular:**")
                    
                    tokens_html = ""
                    for tok, tid in zip(tokens, ids):
                        display_tok = tok.replace('Ġ', ' ␣').replace('Ċ', ' \\n')
                        tokens_html += f"""
                        <div style="display: inline-block; background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 6px; padding: 6px 12px; margin: 4px; text-align: center; font-family: monospace;">
                            <span style="color: #4ade80; font-weight: bold; font-size: 1.1em;">{display_tok}</span>
                            <br>
                            <span style="color: #94a3b8; font-size: 0.8em;">ID: {tid}</span>
                        </div>
                        """
                    st.markdown(f'<div style="background: rgba(15, 23, 42, 0.5); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">{tokens_html}</div>', unsafe_allow_html=True)
                    st.write(f"*Total tokeni generați: {len(tokens)} (max 512 context).*")
                
                # 2. Afișare rezultate FAISS
                results = st.session_state.search_results
                if not results:
                    st.warning("Nu s-au găsit fragmente relevante în baza de date vectorială.")
                else:
                    st.success(f"S-au găsit cele mai relevante {len(results)} fragmente de cod:")
                    
                    for idx, chunk in enumerate(results):
                        chunk_type_label = chunk["type"].upper()
                        lines_label = f"Lines {chunk['start_line']}-{chunk['end_line']}"
                        
                        st.markdown(f"""
                        <div class="glass-card" style="margin-bottom: 20px;">
                            <h4 style="margin: 0; color: #38bdf8; display: flex; justify-content: space-between;">
                                <span>[{idx+1}] Fișier: {chunk['file_path']}</span>
                                <span style="font-size: 0.8em; color: #8b5cf6;">Similaritate Scenariu (Cosine): {chunk['score']:.4f}</span>
                            </h4>
                            <div style="margin: 10px 0;">
                                <span class="custom-badge custom-badge-green"><b>Tip fragment:</b> {chunk_type_label}</span>
                                <span class="custom-badge custom-badge-blue"><b>Interval linii:</b> {lines_label}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Analiză Structurală Automată (AST Docs)"):
                            st.markdown(f"**Rezumat structural:** {chunk['summary']}")
                            st.markdown(f"**Documentație (Docstring):**")
                            st.info(chunk["docstring"])
                            
                            if "args" in chunk and chunk["args"]:
                                st.markdown(f"**Argumente acceptate:** `{', '.join(chunk['args'])}`")
                            if "parents" in chunk and chunk["parents"]:
                                st.markdown(f"**Moștenește:** `{', '.join(chunk['parents'])}`")
                            if "methods" in chunk and chunk["methods"]:
                                st.markdown("**Metode identificate:**")
                                for m in chunk["methods"]:
                                    st.markdown(f"- `{m}`")
                                    
                        suffix = Path(chunk["file_path"]).suffix.lower()
                        lang_map = {
                            '.py': 'python', '.js': 'javascript', '.ts': 'typescript', 
                            '.tsx': 'typescript', '.jsx': 'javascript', '.java': 'java', 
                            '.cpp': 'cpp', '.cs': 'csharp'
                        }
                        syntax_lang = lang_map.get(suffix, 'text')
                        st.code(chunk["content"], language=syntax_lang, line_numbers=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("Introduceți o interogare în formularul de mai sus și apăsați butonul Caută.")
                
        # --- SUB-TAB 3.2: EXPLORATOR ATENȚIE CODEBERT (WOW HIGHLIGHT) ---
        with sub_tab_attention:
            st.markdown("## Explorator Atenție CodeBERT (Self-Attention Weight Heatmap)")
            st.markdown("Această funcționalitate demonstrează în timp real pilonul principal al arhitecturii **Transformer**: mecanismul de **Self-Attention** (auto-atenție). Introduceți o linie scurtă de cod sau o frază mai jos, iar modelul CodeBERT va calcula și va desena o hartă termică (heatmap) a corelațiilor dintre tokeni în ultimul său strat de atenție.")
            st.write("---")
            
            # Formular separat pentru atenție pentru a rula la cerere
            with st.form("attention_form"):
                code_input = st.text_input("Introdu o linie de cod sau o frază scurtă:", value="def connect_to_server(ip, port):")
                submitted_attention = st.form_submit_button("Generează Harta de Atenție")
                
            if submitted_attention and code_input:
                with st.spinner("Se extrage tensorul de atenție din Transformer..."):
                    try:
                        indexer = st.session_state.indexer
                        att_tokens, att_matrix = indexer.get_attention_matrix(code_input)
                        
                        import matplotlib.pyplot as plt
                        
                        # Curățăm tokenii pentru o afișare mai lizibilă în grafic
                        display_tokens = [t.replace('Ġ', ' ').replace('Ċ', ' \\n') for t in att_tokens]
                        
                        # Generare heatmap Matplotlib stilizat cu fundal întunecat pentru a se potrivi cu tema neon
                        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0d1117')
                        ax.set_facecolor('#0d1117')
                        
                        # Plasma colormap oferă nuanțe de neon (violet-roz-galben) care arată premium
                        im = ax.imshow(att_matrix, cmap='plasma', interpolation='nearest')
                        
                        # Configurare axe și etichete
                        ax.set_xticks(np.arange(len(display_tokens)))
                        ax.set_yticks(np.arange(len(display_tokens)))
                        ax.set_xticklabels(display_tokens, rotation=90, color='#c9d1d9', fontsize=10)
                        ax.set_yticklabels(display_tokens, color='#c9d1d9', fontsize=10)
                        
                        # Stilizare grid și border
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#30363d')
                        ax.spines['bottom'].set_color('#30363d')
                        
                        # Adăugare colorbar cu text stilizat
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.yaxis.set_tick_params(color='#c9d1d9')
                        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#c9d1d9')
                        cbar.set_label('Intensitate Auto-Atenție', color='#c9d1d9', labelpad=10)
                        
                        plt.title("Matricea de Auto-Atenție (Mean Head Attention - Stratul 12)", color='#38bdf8', fontsize=14, pad=20, fontweight='bold')
                        plt.tight_layout()
                        
                        # Afișare în Streamlit
                        st.pyplot(fig)
                        
                        st.markdown("""
                        **Cum se interpretează acest grafic?**
                        * Fiecare celulă din matrice `(i, j)` indică intensitatea atenției pe care tokenul din stânga `i` o acordă tokenului de jos `j`.
                        * Culorile luminoase (galben, portocaliu) indică o atenție puternică, în timp ce culorile închise (albastru, violet) reprezintă o corelație redusă.
                        * Acest comportament demonstrează capacitatea nativă a arhitecturii **Transformer** de a asocia contextul global fără a fi limitată de distanța dintre tokeni, spre deosebire de rețelele RNN sau LSTM.
                        """)
                    except Exception as e:
                        st.error(f"Nu s-a putut genera harta de atenție: {str(e)}")
            else:
                st.info("Apăsați butonul 'Generează Harta de Atenție' de mai sus pentru a vizualiza rețeaua.")

