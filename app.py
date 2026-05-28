import os
# Rezolvarea conflictului de librării OpenMP (OMP: Error #15) pe Mac local pentru PyTorch + FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import streamlit.components.v1 as components
import shutil
import torch
torch.classes.__path__ = [] 
import pickle
import traceback
import numpy as np
import random
from pathlib import Path
import torch
torch.classes.__path__ = []

from code_parser import (
    unzip_project,
    scan_project_files,
    build_file_tree,
    parse_and_chunk_file,
    generate_uml_class_diagram,
    generate_dependency_diagram,
    generate_sequence_diagram,
    generate_flowchart_diagram,
    generate_package_diagram,
)
from vector_store import CodeBERTIndexer, DEVICE
from security_analyzer import analyze_python_file
import traceback

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
if "security_findings" not in st.session_state:
    st.session_state.security_findings = []
if "quiz_ast_questions" not in st.session_state:
    st.session_state.quiz_ast_questions = []
if "analysis_duplicates" not in st.session_state:
    st.session_state.analysis_duplicates = None
if "analysis_smells" not in st.session_state:
    st.session_state.analysis_smells = None
if "quiz_ast_submitted" not in st.session_state:
    st.session_state.quiz_ast_submitted = False
if "quiz_semantic_q" not in st.session_state:
    st.session_state.quiz_semantic_q = None
if "quiz_semantic_answer" not in st.session_state:
    st.session_state.quiz_semantic_answer = None
if "quiz_semantic_scores" not in st.session_state:
    st.session_state.quiz_semantic_scores = None

CODE_SMELLS = [
    {
        "query": "function with too many parameters and arguments complex signature",
        "name": "Prea mulți parametri",
        "rec": "Grupează parametrii înrudiți într-un obiect de configurare sau folosește `**kwargs`. Funcțiile cu mai mult de 4-5 argumente sunt greu de testat și de înțeles.",
        "threshold": 0.58,
        "severity": "warning"
    },
    {
        "query": "missing error handling try except exception no validation",
        "name": "Lipsă gestionare erori",
        "rec": "Adaugă blocuri `try/except` pentru operații critice (I/O, rețea, baze de date). Definește excepții custom pentru erori specifice domeniului.",
        "threshold": 0.55,
        "severity": "error"
    },
    {
        "query": "deeply nested code multiple indentation levels loops conditions",
        "name": "Cod prea imbricat",
        "rec": "Folosește **early return** (guard clauses) sau extrage logica imbricată în funcții separate pentru a reduce adâncimea și a crește lizibilitatea.",
        "threshold": 0.55,
        "severity": "warning"
    },
    {
        "query": "function without documentation docstring no comments explanation",
        "name": "Lipsă documentație",
        "rec": "Adaugă un docstring care descrie scopul funcției, parametrii (tip și semnificație) și valoarea returnată. Folosește formatul Google sau NumPy.",
        "threshold": 0.58,
        "severity": "info"
    },
    {
        "query": "very long function doing too many things violates single responsibility",
        "name": "Funcție prea lungă (SRP)",
        "rec": "Aplică principiul **Single Responsibility**: împarte funcția în mai multe funcții cu scop unic. O funcție bună face un singur lucru și îl face bine.",
        "threshold": 0.55,
        "severity": "warning"
    },
    {
        "query": "hardcoded magic numbers string literals constants no configuration",
        "name": "Valori hardcodate",
        "rec": "Mută valorile literale (`42`, `'localhost'`, `3000`) în constante cu nume descriptive sau într-un fișier de configurare separat.",
        "threshold": 0.55,
        "severity": "info"
    },
    {
        "query": "global variable modification side effects mutable state",
        "name": "Modificare stare globală",
        "rec": "Evită modificarea stării globale. Preferă funcții pure cu parametri expliciți și valori returnate — mai ușor de testat și de depanat.",
        "threshold": 0.54,
        "severity": "error"
    },
    {
        "query": "duplicate repeated code copy paste similar logic redundant",
        "name": "Cod duplicat / copy-paste",
        "rec": "Extrage logica repetată într-o funcție utilitară și refolosește-o. Principiul DRY (Don't Repeat Yourself) reduce suprafața de bug-uri.",
        "threshold": 0.57,
        "severity": "warning"
    },
    {
        "query": "unused variable import dead code unreachable",
        "name": "Cod mort / variabile neutilizate",
        "rec": "Șterge importurile și variabilele neutilizate. Codul mort crește complexitatea fără beneficiu și poate induce în eroare.",
        "threshold": 0.55,
        "severity": "info"
    },
    {
        "query": "broad except catching all exceptions swallowing errors silently",
        "name": "Except prea generic",
        "rec": "Înlocuiește `except Exception` sau `except:` cu excepții specifice. Prinderea tuturor erorilor ascunde bug-uri reale.",
        "threshold": 0.55,
        "severity": "error"
    },
]

SEVERITY_COLOR = {"error": "#ef4444", "warning": "#f59e0b", "info": "#38bdf8"}
SEVERITY_LABEL = {"error": "Eroare", "warning": "Avertisment", "info": "Sugestie"}

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def get_all_embeddings(indexer):
    if indexer.index is None or indexer.index.ntotal == 0:
        return None
    n = indexer.index.ntotal
    dim = indexer.index.d
    embeddings = np.zeros((n, dim), dtype='float32')
    for i in range(n):
        embeddings[i] = indexer.index.reconstruct(i)
    return embeddings

def find_duplicates(embeddings, chunks, threshold=0.88):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normalized = (embeddings / norms).astype('float32')
    sim_matrix = np.dot(normalized, normalized.T)
    pairs = []
    n = min(len(chunks), embeddings.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim > threshold and chunks[i]["file_path"] != chunks[j]["file_path"] or \
               (sim > threshold and chunks[i].get("name") != chunks[j].get("name")):
                pairs.append((i, j, sim))
    return sorted(pairs, key=lambda x: -x[2])[:15]

def analyze_code_smells(chunks, indexer, progress_cb=None):
    smell_queries = [s["query"] for s in CODE_SMELLS]
    if progress_cb:
        progress_cb(0.05, "Se vectorizează descriptorii de code smells...")
    smell_embeddings = indexer.get_embeddings(smell_queries)

    results = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        if progress_cb:
            progress_cb(0.05 + 0.9 * idx / total, f"Analizăm chunk-ul {idx+1}/{total}: {chunk.get('name','?')}")

        chunk_emb = indexer.get_embeddings([chunk["content"][:512]])[0]
        detected = []

        # CodeBERT smell matching
        for i, smell in enumerate(CODE_SMELLS):
            sim = cosine_sim(chunk_emb, smell_embeddings[i])
            if sim >= smell["threshold"]:
                detected.append({
                    "name": smell["name"],
                    "rec": smell["rec"],
                    "severity": smell["severity"],
                    "score": sim
                })

        # AST-based checks (hibrid)
        args = chunk.get("args", [])
        if len(args) > 5:
            detected.append({
                "name": "Prea mulți parametri (AST)",
                "rec": f"Funcția are **{len(args)} argumente** (`{', '.join(args)}`). Consideră refactorizarea cu un obiect de configurare.",
                "severity": "warning",
                "score": 1.0
            })
        lines = chunk.get("end_line", 0) - chunk.get("start_line", 0)
        if lines > 60 and chunk["type"] == "function":
            detected.append({
                "name": "Funcție prea lungă (AST)",
                "rec": f"Funcția are **{lines} linii**. Împarte-o în sub-funcții cu responsabilitate unică.",
                "severity": "warning",
                "score": 1.0
            })
        if chunk.get("docstring", "").startswith("Fără descriere") and chunk["type"] in ("function", "class"):
            detected.append({
                "name": "Lipsă docstring (AST)",
                "rec": "Elementul nu are docstring. Adaugă o descriere a scopului, parametrilor și valorii returnate.",
                "severity": "info",
                "score": 1.0
            })

        if detected:
            results.append({"chunk": chunk, "smells": detected})

    return sorted(results, key=lambda x: len(x["smells"]), reverse=True)

def generate_ast_questions(chunks):
    questions = []
    func_chunks = [c for c in chunks if c["type"] == "function"]
    class_chunks = [c for c in chunks if c["type"] == "class"]
    all_files = list(set(c["file_path"] for c in chunks))

    # Tip 1: Câte argumente are funcția X?
    if func_chunks:
        for chunk in random.sample(func_chunks, min(3, len(func_chunks))):
            correct = len(chunk.get("args", []))
            wrongs = sorted(set([max(0, correct - 1), correct + 1, correct + 2, correct + 3]) - {correct})[:3]
            options = [str(correct)] + [str(w) for w in wrongs]
            random.shuffle(options)
            questions.append({
                "question": f"Câte argumente acceptă funcția `{chunk['name']}` din `{chunk['file_path']}`?",
                "code": chunk["content"][:500],
                "options": options,
                "correct": str(correct),
                "explanation": f"Argumentele sunt: `{', '.join(chunk['args']) if chunk.get('args') else 'niciun argument'}`."
            })

    # Tip 2: Din ce fișier face parte funcția X?
    if func_chunks and len(all_files) > 1:
        for chunk in random.sample(func_chunks, min(2, len(func_chunks))):
            correct_file = chunk["file_path"]
            other_files = [f for f in all_files if f != correct_file]
            wrong_files = random.sample(other_files, min(3, len(other_files)))
            options = [correct_file] + wrong_files
            random.shuffle(options)
            questions.append({
                "question": f"În ce fișier este definită funcția `{chunk['name']}`?",
                "code": chunk["content"][:500],
                "options": options,
                "correct": correct_file,
                "explanation": f"Funcția `{chunk['name']}` se află în `{correct_file}` (liniile {chunk['start_line']}-{chunk['end_line']})."
            })

    # Tip 3: Din ce clasă moștenește clasa X?
    inheriting = [c for c in class_chunks if c.get("parents")]
    if inheriting and len(class_chunks) > 1:
        for chunk in random.sample(inheriting, min(2, len(inheriting))):
            correct = chunk["parents"][0]
            other_names = [c["name"] for c in class_chunks if c["name"] not in (chunk["name"], correct)]
            if len(other_names) >= 1:
                wrong = random.sample(other_names, min(3, len(other_names)))
                options = [correct] + wrong
                random.shuffle(options)
                questions.append({
                    "question": f"Din ce clasă moștenește `{chunk['name']}`?",
                    "code": chunk["content"][:500],
                    "options": options,
                    "correct": correct,
                    "explanation": f"Clasa `{chunk['name']}` moștenește din `{correct}`."
                })

    random.shuffle(questions)
    return questions[:5]

def generate_restructuring_suggestion(chunk, smell_name):
    """Returnează (before_code, after_code) cu codul real din chunk ca 'before'."""
    name = chunk.get("name", "function_name")
    args = chunk.get("args", [])
    methods = chunk.get("methods", [])
    chunk_type = chunk.get("type", "function")
    lines_count = chunk.get("end_line", 0) - chunk.get("start_line", 0)
    arg_str = ", ".join(args)
    actual_code = chunk.get("content", "").strip()
    # Trunchiem codul original la max 40 linii pentru afișare
    actual_lines = actual_code.splitlines()
    before = "\n".join(actual_lines[:40])
    if len(actual_lines) > 40:
        before += f"\n    ... ({len(actual_lines) - 40} linii omise)"

    if "Prea mulți parametri" in smell_name:
        fields = "\n".join(f"    {a}: Any" for a in args)
        after = f"""\
from dataclasses import dataclass
from typing import Any

@dataclass
class {name.capitalize()}Config:
{fields}

def {name}(cfg: {name.capitalize()}Config):
    # Acces: cfg.{args[0] if args else 'param'}, cfg.{args[1] if len(args)>1 else 'param2'}, ...
    ..."""
        return before, after

    if "lungă" in smell_name.lower() or "srp" in smell_name.lower():
        n = max(lines_count // 3, 5)
        sub1 = f"_validate_{name}"
        sub2 = f"_process_{name}"
        sub3 = f"_format_{name}_result"
        after = f"""\
def {sub1}({arg_str}):
    \"\"\"Validează input-ul și precondițiile ({n} linii din {name}).\"\"\"
    ...

def {sub2}({arg_str}):
    \"\"\"Logica principală de procesare ({n} linii din {name}).\"\"\"
    ...

def {sub3}(result):
    \"\"\"Formatează și returnează rezultatul final ({n} linii din {name}).\"\"\"
    ...

def {name}({arg_str}):
    \"\"\"Orchestrează fluxul — fiecare pas delegat unui sub-modul.\"\"\"
    {sub1}({arg_str})
    result = {sub2}({arg_str})
    return {sub3}(result)"""
        return before, after

    if "docstring" in smell_name.lower() or "documentație" in smell_name.lower():
        if chunk_type == "function":
            args_doc = "\n".join(f"        {a}: [tip] — [descriere]" for a in args) or "        # fără parametri"
            # Inserăm docstring-ul în codul real
            first_line = actual_lines[0] if actual_lines else f"def {name}({arg_str}):"
            rest = "\n".join(actual_lines[1:6]) if len(actual_lines) > 1 else "    ..."
            after = f"""\
{first_line}
    \"\"\"
    [Descrie ce face `{name}` în 1-2 propoziții.]

    Args:
{args_doc}

    Returns:
        [tip]: [Ce returnează și în ce condiții.]

    Raises:
        ValueError: [Când input-ul e invalid.]
    \"\"\"
{rest}
    ..."""
        else:
            methods_preview = "\n".join(f"        {m.split('(')[0]}(): [descriere]" for m in methods[:5])
            first_line = actual_lines[0] if actual_lines else f"class {name}:"
            after = f"""\
{first_line}
    \"\"\"
    [Descrie scopul clasei `{name}`.]

    Attributes:
        [attr] ([tip]): [descriere]

    Methods:
{methods_preview or '        # listează metodele principale'}
    \"\"\"
    ..."""
        return before, after

    if "Except prea generic" in smell_name:
        import re
        # Găsim blocul try/except real și îl refactorizăm
        after = f"""\
import logging
logger = logging.getLogger(__name__)

# Înlocuiește blocul except: sau except Exception: cu variante specifice:
try:
    result = {name}({arg_str})
except ValueError as e:
    logger.error("Input invalid în {name}: %s", e)
    raise
except (IOError, OSError) as e:
    logger.error("Eroare I/O în {name}: %s", e)
    raise
except Exception as e:
    logger.critical("Eroare neașteptată în {name}: %s", e, exc_info=True)
    raise"""
        return before, after

    if "SQL" in smell_name:
        after = f"""\
# Varianta 1 — parametrizare directă (cursor DB-API):
def {name}(user_id: int):
    cursor.execute(
        "SELECT * FROM users WHERE id = %s",
        (user_id,)          # argument separat — nu se interpolează în query
    )
    return cursor.fetchone()

# Varianta 2 — ORM SQLAlchemy (recomandat în proiecte mari):
def {name}(user_id: int):
    return db.session.query(User).filter(User.id == user_id).first()"""
        return before, after

    if "Hardcoded Secret" in smell_name or "hardcodat" in smell_name.lower():
        after = f"""\
import os
from dotenv import load_dotenv

load_dotenv()  # citește .env din rădăcina proiectului

{name} = os.getenv("{name.upper()}")
if not {name}:
    raise EnvironmentError(
        "Variabila de mediu '{name.upper()}' nu este setată. "
        "Adaug-o în fișierul .env (și adaugă .env în .gitignore!)"
    )

# --- .env (NU commit în git!) ---
# {name.upper()}=valoarea_ta_secreta"""
        return before, after

    if "Command Injection" in smell_name or "Shell" in smell_name:
        after = f"""\
import subprocess

# Argumentele ca listă — shell=False implicit, fără injecție posibilă:
result = subprocess.run(
    ["comanda", arg1, arg2],   # niciodată string interpolat cu input extern
    capture_output=True,
    text=True,
    timeout=30,
    check=True                 # ridică CalledProcessError la exit code != 0
)
output = result.stdout"""
        return before, after

    if "global" in smell_name.lower():
        after = f"""\
# Varianta 1 — funcție pură (preferată, ușor de testat):
def {name}({arg_str}, state: dict) -> dict:
    return {{**state, "result": ...}}   # returnează stare nouă, nu modifică global

# Varianta 2 — clasă cu stare encapsulată:
class {name.capitalize()}Manager:
    def __init__(self):
        self._state: dict = {{}}

    def {name}(self, {arg_str}):
        self._state["result"] = ...
        return self._state.copy()  # returnăm copie — protejăm starea internă"""
        return before, after

    if "duplicat" in smell_name.lower():
        after = f"""\
# Extrage logica comună într-o funcție utilitară partajată:
def _shared_{name}_logic({arg_str}):
    \"\"\"Logica extrasă din {name} (aplicată peste tot unde era duplicată).\"\"\"
    ...   # codul comun din ambele copii

# Înlocuiești FIECARE copie cu apelul la funcția comună:
def {name}({arg_str}):
    return _shared_{name}_logic({arg_str})

# Dacă al doilea loc era într-o altă funcție (ex: {name}_v2):
def {name}_v2({arg_str}):
    return _shared_{name}_logic({arg_str})"""
        return before, after

    return None


def generate_semantic_question(chunks):
    chunk = random.choice(chunks)
    correct_desc = chunk.get("docstring") or chunk.get("summary", "")
    other_chunks = [c for c in chunks if c != chunk]
    distractors = random.sample(other_chunks, min(2, len(other_chunks)))
    distractor_descs = [c.get("docstring") or c.get("summary", "") for c in distractors]
    options = [correct_desc] + distractor_descs
    random.shuffle(options)
    return {
        "code": chunk["content"][:600],
        "options": options,
        "correct": correct_desc,
        "chunk_name": chunk.get("name", "fragment")
    }

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
    st.session_state.security_findings = []
    st.session_state.quiz_ast_questions = []
    st.session_state.quiz_ast_submitted = False
    st.session_state.quiz_semantic_q = None
    st.session_state.quiz_semantic_answer = None
    st.session_state.quiz_semantic_scores = None
    st.session_state.analysis_duplicates = None
    st.session_state.analysis_smells = None

def render_mermaid(mermaid_code, height=500):
    """
    Randează un diagramă Mermaid.js într-un iframe Streamlit,
    cu zoom + pan custom (fără svg-pan-zoom, stabil și fără erori SVGMatrix).
    """

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
          display: block;
          overflow: visible;
}}

          svg {{
            width: 100% !important;
            height: 100% !important;
            max-width: none;
            transform-origin: center;
            user-select: none;
          }}
        </style>

        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

          mermaid.initialize({{
            startOnLoad: false,
            theme: 'base',
            securityLevel: 'loose'
          }});

          async function initDiagram() {{
            try {{
              const container = document.getElementById('diagram-container');
              const code = document.getElementById('mermaid-data').textContent;

              console.log("MERMAID CODE:");
              console.log(code);

              const result = await mermaid.render('mermaid-svg', code);

              console.log("MERMAID RESULT:");
              console.log(result);

              container.innerHTML = result.svg;

              ;

              const svg = container.querySelector('svg');
              
              await new Promise(requestAnimationFrame);
              await new Promise(requestAnimationFrame);
              requestAnimationFrame(() => {{
                  requestAnimationFrame(() => {{
                
                    const realSvg = container.querySelector("svg");

                    const bbox = realSvg.getBBox();
                    console.log("REAL bbox:", bbox);

                    if (!bbox.width || !bbox.height) {{
                      console.warn("Still not ready, retrying...");
                      setTimeout(initDiagram, 50);
                      return;
                    }}

                    realSvg.setAttribute(
                      "viewBox",
                      `${{bbox.x - 10}} ${{ bbox.y - 10 }} ${{ bbox.width + 20 }} $${{ bbox.height + 20 }}`
                    );

                  }});
              }});
              
              console.log("SVG INJECTED");
              console.log("SVG ELEMENT:");
              console.log(svg);

              if (!svg) {{
                throw new Error("SVG-ul Mermaid nu a fost generat.");
              }}

              // styling stabil
              svg.style.width = '100%';
              svg.style.height = '100%';
              svg.style.maxWidth = 'none';
              svg.style.transformOrigin = 'center';

              let scale = 1;
              let panX = 0;
              let panY = 0;

              let dragging = false;
              let startX = 0;
              let startY = 0;

              function applyTransform() {{
                svg.style.transform =
                  `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
              }}

              // 🔥 ZOOM
              container.addEventListener('wheel', (e) => {{
                e.preventDefault();

                const zoomStep = 0.1;
                const direction = e.deltaY > 0 ? -1 : 1;

                scale = Math.min(10, Math.max(0.2, scale + direction * zoomStep));

                applyTransform();
              }}, {{ passive: false }});

              // 🔥 PAN START
              container.addEventListener('mousedown', (e) => {{
                dragging = true;
                startX = e.clientX - panX;
                startY = e.clientY - panY;
                container.style.cursor = 'grabbing';
              }});

              container.addEventListener('mousemove', (e) => {{
                if (!dragging) return;

                panX = e.clientX - startX;
                panY = e.clientY - startY;

                applyTransform();
              }});

              container.addEventListener('mouseup', () => {{
                dragging = false;
                container.style.cursor = 'grab';
              }});

              container.addEventListener('mouseleave', () => {{
                dragging = false;
                container.style.cursor = 'grab';
              }});

            }} catch (error) {{
              console.error(error);
              console.error("MERMAID ERROR:", error);

              document.getElementById('diagram-container').innerHTML =
                `<div style="color:#ef4444;font-family:sans-serif;padding:20px;">
                  Eroare randare diagramă: ${{{{error}}}}
                </div>`;
            }}
          }}

          window.addEventListener('load', initDiagram);
        </script>
      </head>

      <body>
        <div id="diagram-container">
          <div style="color:#94a3b8;font-family:sans-serif;margin-top:20%;">
            Se randează diagrama...
          </div>
        </div>

        <script id="mermaid-data" type="text/plain">{escaped_code}</script>
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
    
    security_findings = []

    for file in all_files:

        if file.suffix.lower() == ".py":

            findings = analyze_python_file(file, TEMP_DIR, indexer=st.session_state.indexer)

            security_findings.extend(findings)

    st.session_state.security_findings = security_findings
    
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

            # Analiză de securitate AST + CodeBERT semantic (rulează după ce modelul e încărcat)
            with st.spinner("Se rulează auditul de securitate (AST + CodeBERT semantic)..."):
                security_findings = []
                for file in all_files:
                    if file.suffix.lower() == ".py":
                        try:
                            findings = analyze_python_file(file, TEMP_DIR, indexer=st.session_state.indexer)
                            security_findings.extend(findings)
                        except Exception:
                            pass
                st.session_state.security_findings = security_findings

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard & Explorator Cod",
        "Arhitectură UML & Relații",
        "Căutare Semantică în Proiect",
        "Securitate",
        "Quiz Cod & Semantic",
        "Analiză & Recomandări"
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
        st.markdown("## Diagrame UML — Generate Determinist din AST")
        st.markdown("Alege tipul de diagramă dorit. Toate sunt generate **100% offline** prin analiza arborelui sintactic al codului tău.")

        UML_TYPES = {
            "Diagramă de Clase": "class",
            "Dependențe între Module": "dependency",
            "Diagramă de Secvență": "sequence",
            "Flowchart (Lanț de Apeluri)": "flowchart",
            "Diagramă de Pachete": "package",
        }
        UML_DESCRIPTIONS = {
            "class": "Clasele, metodele, câmpurile și relațiile de moștenire detectate prin AST.",
            "dependency": "Relațiile de import între fișierele proiectului.",
            "sequence": "Apelurile de metode între clase, extrase din corpul funcțiilor.",
            "flowchart": "Lanțul de apeluri de funcții la nivel de modul.",
            "package": "Gruparea fișierelor pe directoare/pachete și dependențele dintre ele.",
        }

        col_sel, col_h = st.columns([2, 1])
        with col_sel:
            selected_uml = st.radio(
                "Tip diagramă:",
                list(UML_TYPES.keys()),
                horizontal=True,
                key="uml_type_radio"
            )
        with col_h:
            diag_height = st.slider("Înălțime (px):", 300, 1200, 550, 50)

        uml_key = UML_TYPES[selected_uml]
        st.caption(UML_DESCRIPTIONS[uml_key])
        st.write("---")

        # Generăm diagrama selectată la cerere
        @st.cache_data(show_spinner=False)
        def get_diagram(diagram_type, files_key):
            files = st.session_state.files_list
            if diagram_type == "class":
                return generate_uml_class_diagram(files, TEMP_DIR)
            elif diagram_type == "dependency":
                return generate_dependency_diagram(files, TEMP_DIR)
            elif diagram_type == "sequence":
                return generate_sequence_diagram(files, TEMP_DIR)
            elif diagram_type == "flowchart":
                return generate_flowchart_diagram(files, TEMP_DIR)
            elif diagram_type == "package":
                return generate_package_diagram(files, TEMP_DIR)
            return ""

        files_key = str([str(f) for f in st.session_state.files_list])
        with st.spinner(f"Se generează {selected_uml}..."):
            diagram_code = get_diagram(uml_key, files_key)

        render_mermaid(diagram_code, height=diag_height)
        with st.expander("Cod sursă Mermaid:"):
            st.code(diagram_code, language="mermaid")

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
                        att_matrix = np.array(att_matrix)

                        st.write("Attention shape:", att_matrix.shape)

                        if len(att_matrix.shape) == 4:
                            att_matrix = att_matrix[-1][0]

                        elif len(att_matrix.shape) == 3:
                            att_matrix = att_matrix[0]

                        if len(att_matrix.shape) != 2:
                            st.error(f"Format attention invalid: {att_matrix.shape}")
                            st.stop()
                        
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
                        st.error("Detalii eroare:")
                        st.text(traceback.format_exc())
            else:
                st.info("Apăsați butonul 'Generează Harta de Atenție' de mai sus pentru a vizualiza rețeaua.")
    # ----------------- TAB 4: ANALIZĂ DE SECURITATE -----------------
    with tab4:

        st.markdown("## Security Audit — Analiză Statică AST + Validare Semantică CodeBERT")
        st.markdown("Auditorul scanează toate fișierele Python prin **analiza AST** pentru pattern-uri de vulnerabilitate cunoscute, apoi validează fiecare găsire cu **CodeBERT** prin similaritate cosinus față de exemple de cod nesigur.")
        st.write("---")

        findings = st.session_state.security_findings

        if not findings:
            st.success("Nu au fost detectate vulnerabilități în fișierele Python din proiect.")
        else:
            # Sumar pe severitate
            high = [f for f in findings if f["severity"] == "HIGH"]
            med  = [f for f in findings if f["severity"] == "MEDIUM"]
            low  = [f for f in findings if f["severity"] == "LOW"]

            s1, s2, s3 = st.columns(3)
            s1.metric("Critice (HIGH)", len(high))
            s2.metric("Medii (MEDIUM)", len(med))
            s3.metric("Scăzute (LOW)", len(low))
            st.write("---")
            st.warning(f"**{len(findings)} probleme de securitate** detectate în codebase:")

            SEV_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#38bdf8"}
            SEV_REC = {
                "Command Injection": "Evită `os.system()`. Folosește `subprocess.run([...])` cu argumente ca listă, fără `shell=True`.",
                "Shell Execution": "Nu folosi `shell=True` în subprocess. Pasează comanda ca listă de argumente.",
                "Possible SQL Injection": "Folosește prepared statements / parametrizare ORM în loc de concatenare string.",
                "Unsafe Deserialization": "`pickle.loads()` poate executa cod arbitrar. Folosește JSON sau `ast.literal_eval` pentru date nesigure.",
                "Weak Cryptography": "MD5 și SHA1 sunt compromise. Folosește SHA-256 sau bcrypt/argon2 pentru parole.",
                "Dynamic Code Execution": "`eval()`/`exec()` pe input extern este extrem de periculos. Refactorizează logica.",
                "Hardcoded Secret": "Nu hardcoda credențiale în cod. Folosește variabile de mediu sau un secret manager.",
            }

            for finding in sorted(findings, key=lambda x: {"HIGH":0,"MEDIUM":1,"LOW":2}.get(x["severity"],3)):
                severity = finding["severity"]
                color = SEV_COLOR.get(severity, "#94a3b8")
                rec = SEV_REC.get(finding["type"], "Revizuiește manual această secțiune de cod.")
                sem_match = finding.get("semantic_match") or "N/A"
                sem_score = finding.get("semantic_score", 0.0)
                score_bar = int(min(sem_score * 100, 100))

                st.markdown(f"""
                <div style="border:1px solid {color}; border-left:5px solid {color}; padding:16px; border-radius:10px; margin-bottom:16px; background:rgba(255,255,255,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <h4 style="margin:0; color:{color};">{finding["type"]}</h4>
                        <span style="background:{color}22; color:{color}; padding:3px 10px; border-radius:20px; font-size:0.8em; font-weight:bold;">{severity}</span>
                    </div>
                    <p style="margin:4px 0; color:#94a3b8;"><b>Fișier:</b> {finding["file"]} · <b>Linia:</b> {finding["line"]}</p>
                    <p style="margin:4px 0; color:#c9d1d9;">{finding["description"]}</p>
                    <pre style="background:#0d1117; padding:8px; border-radius:6px; color:#4ade80; font-size:0.85em; margin:8px 0;">{finding["code"]}</pre>
                    <div style="margin:8px 0;">
                        <span style="color:#38bdf8; font-size:0.85em;"><b>AI Semantic Match:</b> {sem_match} &nbsp;·&nbsp; <b>Confidence:</b> {sem_score:.3f}</span>
                        <div style="background:#1e293b; border-radius:4px; height:5px; margin-top:4px;">
                            <div style="background:#8b5cf6; width:{score_bar}%; height:5px; border-radius:4px;"></div>
                        </div>
                    </div>
                    <div style="background:rgba(56,189,248,0.07); border-left:3px solid #38bdf8; padding:8px 12px; border-radius:0 6px 6px 0; margin-top:8px;">
                        <span style="color:#38bdf8; font-size:0.85em;">💡 <b>Recomandare:</b> {rec}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ----------------- TAB 5: QUIZ COD & SEMANTIC -----------------
    with tab5:
        st.markdown("## Quiz Interactiv — Cod & Semantic")
        st.markdown("Alege tipul de quiz, setează timpul și apasă **Start Quiz** pentru a lansa sesiunea în popup.")
        st.write("---")

        q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
        with q_col1:
            quiz_type = st.radio(
                "Tip quiz:",
                ["Quiz din Cod (AST)", "Quiz Semantic (CodeBERT)"],
                horizontal=True
            )
        with q_col2:
            quiz_time = st.selectbox("Timp per întrebare (sec):", [15, 30, 45, 60], index=1)
        with q_col3:
            st.write("")
            st.write("")
            launch_quiz = st.button("🚀 Start Quiz", use_container_width=True, type="primary")

        if launch_quiz:
            if quiz_type == "Quiz din Cod (AST)":
                st.session_state.quiz_ast_questions = generate_ast_questions(st.session_state.chunks)
                st.session_state.quiz_ast_submitted = False
            else:
                st.session_state.quiz_semantic_q = generate_semantic_question(st.session_state.chunks)
                st.session_state.quiz_semantic_answer = None
                st.session_state.quiz_semantic_scores = None
            st.session_state["quiz_open"] = quiz_type
            st.session_state["quiz_time"] = quiz_time
            st.rerun()

        # ---- POPUP QUIZ AST ----
        if st.session_state.get("quiz_open") == "Quiz din Cod (AST)" and st.session_state.quiz_ast_questions:

            @st.dialog("Quiz din Cod — AST", width="large")
            def run_ast_quiz():
                questions = st.session_state.quiz_ast_questions
                seconds = st.session_state.get("quiz_time", 30)
                total_sec = seconds * len(questions)

                # Timer JavaScript
                components.html(f"""
                <div id="timer-bar-wrap" style="margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="color:#38bdf8;font-family:monospace;font-size:1.1em;font-weight:bold;">
                      ⏱ Timp rămas: <span id="countdown">{total_sec}</span>s
                    </span>
                    <span style="color:#64748b;font-size:0.85em;">{len(questions)} întrebări · {seconds}s/întrebare</span>
                  </div>
                  <div style="background:#1e293b;border-radius:6px;height:10px;overflow:hidden;">
                    <div id="timer-bar" style="background:linear-gradient(90deg,#8b5cf6,#38bdf8);height:10px;width:100%;border-radius:6px;transition:width 1s linear;"></div>
                  </div>
                </div>
                <script>
                  let total = {total_sec};
                  let left = total;
                  const cd = document.getElementById('countdown');
                  const bar = document.getElementById('timer-bar');
                  const iv = setInterval(() => {{
                    left--;
                    if (cd) cd.textContent = left;
                    if (bar) bar.style.width = (left / total * 100) + '%';
                    if (bar && left < total * 0.3) bar.style.background = 'linear-gradient(90deg,#ef4444,#f59e0b)';
                    if (left <= 0) {{ clearInterval(iv); if(cd) cd.textContent = '0 — Timp expirat!'; }}
                  }}, 1000);
                </script>
                """, height=80)

                user_answers = {}
                for i, q in enumerate(questions):
                    st.markdown(f"**Î{i+1}.** {q['question']}")
                    with st.expander("Cod analizat", expanded=False):
                        st.code(q["code"][:400], language="python")
                    user_answers[i] = st.radio("", options=q["options"], key=f"dlg_ast_{i}", index=None, label_visibility="collapsed")
                    st.divider()

                if st.button("Verifică Răspunsurile", use_container_width=True, type="primary"):
                    score = sum(1 for i, q in enumerate(questions) if user_answers.get(i) == q["correct"])
                    st.write("---")
                    for i, q in enumerate(questions):
                        ans = user_answers.get(i)
                        ok = ans == q["correct"]
                        color = "#4ade80" if ok else "#ef4444"
                        icon = "✅" if ok else "❌"
                        st.markdown(f'<span style="color:{color}">{icon} Î{i+1}: {q["question"]}<br><small>{q["explanation"]}</small></span>', unsafe_allow_html=True)
                    pct = int(score / len(questions) * 100)
                    medal = "🥇" if pct == 100 else "🥈" if pct >= 60 else "🥉"
                    st.markdown(f"""
                    <div style="text-align:center;background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.4);border-radius:12px;padding:20px;margin-top:16px;">
                        <h2 style="color:#a78bfa;">{medal} Scor: {score}/{len(questions)} ({pct}%)</h2>
                    </div>""", unsafe_allow_html=True)
                    st.session_state["quiz_open"] = None

            run_ast_quiz()

        # ---- POPUP QUIZ SEMANTIC ----
        elif st.session_state.get("quiz_open") == "Quiz Semantic (CodeBERT)" and st.session_state.quiz_semantic_q:

            @st.dialog("Quiz Semantic — CodeBERT", width="large")
            def run_semantic_quiz():
                q = st.session_state.quiz_semantic_q
                seconds = st.session_state.get("quiz_time", 30)

                components.html(f"""
                <div id="timer-bar-wrap" style="margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="color:#38bdf8;font-family:monospace;font-size:1.1em;font-weight:bold;">
                      ⏱ Timp rămas: <span id="countdown">{seconds}</span>s
                    </span>
                  </div>
                  <div style="background:#1e293b;border-radius:6px;height:10px;overflow:hidden;">
                    <div id="timer-bar" style="background:linear-gradient(90deg,#8b5cf6,#38bdf8);height:10px;width:100%;border-radius:6px;transition:width 1s linear;"></div>
                  </div>
                </div>
                <script>
                  let total = {seconds};
                  let left = total;
                  const cd = document.getElementById('countdown');
                  const bar = document.getElementById('timer-bar');
                  const iv = setInterval(() => {{
                    left--;
                    if(cd) cd.textContent = left;
                    if(bar) bar.style.width = (left/total*100)+'%';
                    if(bar && left < total*0.3) bar.style.background='linear-gradient(90deg,#ef4444,#f59e0b)';
                    if(left<=0){{clearInterval(iv);if(cd)cd.textContent='0 — Timp expirat!';}}
                  }}, 1000);
                </script>
                """, height=80)

                st.markdown("**Fragmentul de cod:**")
                st.code(q["code"][:500], language="python")
                st.markdown("**Care descriere se potrivește cel mai bine?**")
                selected = st.radio("", options=q["options"], index=None, label_visibility="collapsed")

                if st.button("Verifică cu CodeBERT", use_container_width=True, type="primary"):
                    if selected:
                        with st.spinner("CodeBERT calculează similaritatea cosinus..."):
                            indexer = st.session_state.indexer
                            code_emb = indexer.get_embeddings([q["code"]])[0]
                            scores = {opt: cosine_sim(code_emb, indexer.get_embeddings([opt])[0]) for opt in q["options"]}
                        max_s = max(scores.values())
                        for opt, score in sorted(scores.items(), key=lambda x: -x[1]):
                            is_correct = opt == q["correct"]
                            is_sel = opt == selected
                            bar_w = int(score / max_s * 100) if max_s else 0
                            label = (" ✅ Corect" if is_correct else "") + (" ← ales" if is_sel else "")
                            bc = "rgba(74,222,128,0.4)" if is_correct else "rgba(255,255,255,0.05)"
                            st.markdown(f"""
                            <div style="border:1px solid {bc};border-radius:8px;padding:10px;margin-bottom:8px;background:rgba(255,255,255,0.02);">
                              <div style="display:flex;justify-content:space-between;">
                                <span style="color:#c9d1d9;font-size:0.88em;">{opt[:100]}{'...' if len(opt)>100 else ''}{label}</span>
                                <span style="color:#38bdf8;font-weight:bold;">cos={score:.4f}</span>
                              </div>
                              <div style="background:#1e293b;border-radius:4px;height:6px;margin-top:6px;">
                                <div style="background:{'#4ade80' if is_correct else '#8b5cf6'};width:{bar_w}%;height:6px;border-radius:4px;"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)
                        is_correct_answer = selected == q["correct"]
                        verdict = "✅ Corect!" if is_correct_answer else f"❌ Greșit — răspunsul cu scorul cosinus maxim era descrierea corectă."
                        st.markdown(f"""
                        <div style="text-align:center;background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.4);border-radius:12px;padding:16px;margin-top:12px;">
                            <h3 style="color:#a78bfa;">{verdict}</h3>
                        </div>""", unsafe_allow_html=True)
                        st.session_state["quiz_open"] = None
                    else:
                        st.warning("Selectează o variantă înainte de verificare.")

            run_semantic_quiz()

    # ----------------- TAB 6: ANALIZĂ & RECOMANDĂRI -----------------
    with tab6:
        dup_tab, smells_tab = st.tabs([
            "Detector Cod Duplicat",
            "Code Smells & Recomandări"
        ])

        # --- SUB-TAB 5.1: DETECTOR COD DUPLICAT ---
        with dup_tab:
            st.markdown("## Detector Cod Duplicat — Similarity Matrix CodeBERT")
            st.markdown("Reconstruiește toți vectorii de embeddings din indexul FAISS și calculează **similaritatea cosinus** între toate perechile de chunk-uri. Perechile cu similaritate > 0.88 sunt marcate ca potențial duplicate — Transformer-ul detectează duplicarea **semantică**, nu doar textuală.")
            st.write("---")

            if st.button("Rulează Detectorul de Cod Duplicat", use_container_width=True):
                with st.spinner("Se reconstruiesc embeddings din FAISS și se calculează matricea de similaritate..."):
                    try:
                        indexer = st.session_state.indexer
                        embeddings = get_all_embeddings(indexer)
                        if embeddings is not None:
                            pairs = find_duplicates(embeddings, st.session_state.chunks)
                            st.session_state.analysis_duplicates = pairs
                        else:
                            st.warning("Indexul FAISS nu conține embeddings. Re-procesează proiectul.")
                    except Exception as e:
                        st.error(f"Eroare: {str(e)}")
                st.rerun()

            if st.session_state.analysis_duplicates is not None:
                pairs = st.session_state.analysis_duplicates
                if not pairs:
                    st.success("Nu s-au detectat fragmente de cod semantice duplicate (threshold > 0.88). Codebase-ul pare bine structurat.")
                else:
                    st.warning(f"S-au detectat **{len(pairs)}** perechi de fragmente semantice similare:")
                    for rank, (i, j, sim) in enumerate(pairs):
                        c1 = st.session_state.chunks[i]
                        c2 = st.session_state.chunks[j]
                        pct = int(sim * 100)
                        st.markdown(f"""
                        <div style="background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.35); border-radius: 10px; padding: 14px; margin-bottom: 14px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <b style="color:#f59e0b;">Pereche #{rank+1} — Similaritate cosinus: {sim:.4f} ({pct}%)</b>
                            </div>
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.9em; color:#94a3b8;">
                                <div>📄 <b style="color:#c9d1d9;">{c1.get('name','?')}</b><br>{c1['file_path']} · liniile {c1['start_line']}–{c1['end_line']}</div>
                                <div>📄 <b style="color:#c9d1d9;">{c2.get('name','?')}</b><br>{c2['file_path']} · liniile {c2['start_line']}–{c2['end_line']}</div>
                            </div>
                            <div style="margin-top:10px; color:#38bdf8; font-size:0.85em;">
                                💡 <b>Recomandare:</b> Extrage logica comună într-o funcție utilitară partajată și înlocuiește ambele instanțe cu apelul la aceasta. Aplică principiul DRY.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"Cod + sugestie refactorizare — Pereche #{rank+1}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"<span style='color:#ef4444; font-weight:bold;'>❌ {c1.get('name','?')}</span> — `{c1['file_path']}`", unsafe_allow_html=True)
                                st.code(c1["content"][:600], language="python")
                            with col_b:
                                st.markdown(f"<span style='color:#ef4444; font-weight:bold;'>❌ {c2.get('name','?')}</span> — `{c2['file_path']}`", unsafe_allow_html=True)
                                st.code(c2["content"][:600], language="python")
                            # Sugestie de refactorizare directă
                            n1 = c1.get('name', 'func_a')
                            n2 = c2.get('name', 'func_b')
                            args1 = ", ".join(c1.get("args", []))
                            args2 = ", ".join(c2.get("args", []))
                            st.markdown("<span style='color:#22c55e; font-weight:bold;'>✅ Cum refactorizezi — extrage logica comună:</span>", unsafe_allow_html=True)
                            st.code(f"""\
# 1. Crează o funcție utilitară care conține logica comună:
def _shared_logic({args1 or args2 or 'data'}):
    \"\"\"Logica comună extrasă din {n1} și {n2}.\"\"\"
    ...   # mută aici blocurile de cod identice

# 2. Înlocuiește {n1} cu apel la utilitar:
def {n1}({args1 or 'data'}):
    return _shared_logic({args1 or 'data'})

# 3. Înlocuiește {n2} cu apel la utilitar:
def {n2}({args2 or 'data'}):
    return _shared_logic({args2 or 'data'})""", language="python")
            else:
                st.info("Apasă butonul de mai sus pentru a rula analiza.")

        # --- SUB-TAB 5.2: CODE SMELLS & RECOMANDĂRI ---
        with smells_tab:
            st.markdown("## Code Smells & Recomandări — Analiză Hibridă (AST + CodeBERT)")
            st.markdown("""
            Fiecare fragment de cod este comparat semantic cu **10 descriptori de cod problematic** folosind CodeBERT.
            Analiza combină două surse:
            - **CodeBERT** — similaritate cosinus între vectorii de 768-D ai codului și ai descrierilor de code smells
            - **AST** — verificări structurale deterministe (nr. argumente, lungime funcție, docstring)
            """)
            st.write("---")

            col_run, col_clear = st.columns([1, 1])
            with col_run:
                if st.button("Rulează Analiza Code Smells", use_container_width=True):
                    prog_bar = st.progress(0)
                    prog_text = st.empty()
                    def smell_progress(pct, text):
                        prog_bar.progress(pct)
                        prog_text.text(text)
                    try:
                        results = analyze_code_smells(
                            st.session_state.chunks,
                            st.session_state.indexer,
                            progress_cb=smell_progress
                        )
                        st.session_state.analysis_smells = results
                    except Exception as e:
                        st.error(f"Eroare analiză: {str(e)}")
                    prog_bar.empty()
                    prog_text.empty()
                    st.rerun()
            with col_clear:
                if st.button("Resetează Rezultatele", use_container_width=True):
                    st.session_state.analysis_smells = None
                    st.rerun()

            if st.session_state.analysis_smells is not None:
                results = st.session_state.analysis_smells
                total_chunks = len(st.session_state.chunks)
                affected = len(results)
                total_issues = sum(len(r["smells"]) for r in results)

                # Metrici sumar
                m1, m2, m3 = st.columns(3)
                m1.metric("Fragmente analizate", total_chunks)
                m2.metric("Fragmente cu probleme", affected)
                m3.metric("Probleme totale detectate", total_issues)
                st.write("---")

                if not results:
                    st.success("Nicio problemă detectată. Codebase-ul pare curat!")
                else:
                    # Statistici pe tip severitate
                    err_count = sum(1 for r in results for s in r["smells"] if s["severity"] == "error")
                    warn_count = sum(1 for r in results for s in r["smells"] if s["severity"] == "warning")
                    info_count = sum(1 for r in results for s in r["smells"] if s["severity"] == "info")

                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:20px;">
                        <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:12px; text-align:center;">
                            <div style="color:#ef4444; font-size:1.8em; font-weight:bold;">{err_count}</div>
                            <div style="color:#94a3b8;">Erori</div>
                        </div>
                        <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); border-radius:8px; padding:12px; text-align:center;">
                            <div style="color:#f59e0b; font-size:1.8em; font-weight:bold;">{warn_count}</div>
                            <div style="color:#94a3b8;">Avertismente</div>
                        </div>
                        <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:8px; padding:12px; text-align:center;">
                            <div style="color:#38bdf8; font-size:1.8em; font-weight:bold;">{info_count}</div>
                            <div style="color:#94a3b8;">Sugestii</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    for r in results:
                        chunk = r["chunk"]
                        smells = r["smells"]
                        has_error = any(s["severity"] == "error" for s in smells)
                        icon = "🔴" if has_error else "🟡"
                        with st.expander(f"{icon} {chunk.get('name','?')} — `{chunk['file_path']}` ({len(smells)} probleme)", expanded=has_error):
                            for smell in sorted(smells, key=lambda x: {"error":0,"warning":1,"info":2}[x["severity"]]):
                                color = SEVERITY_COLOR[smell["severity"]]
                                label = SEVERITY_LABEL[smell["severity"]]
                                score_txt = f"cos={smell['score']:.3f}" if smell["score"] < 1.0 else "AST"
                                st.markdown(f"""
                                <div style="background:rgba(255,255,255,0.03); border-left:3px solid {color}; padding:10px 14px; margin:8px 0 4px 0; border-radius:0 6px 6px 0;">
                                    <span style="color:{color}; font-weight:bold;">[{label}]</span>
                                    <span style="color:#c9d1d9; font-weight:600;"> {smell['name']}</span>
                                    <span style="color:#64748b; font-size:0.75em; margin-left:8px;">{score_txt}</span>
                                    <br><span style="color:#94a3b8; font-size:0.88em;">💡 {smell['rec']}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                fix = generate_restructuring_suggestion(chunk, smell["name"])
                                if fix:
                                    before_code, after_code = fix
                                    col_b, col_a = st.columns(2)
                                    with col_b:
                                        st.markdown("<span style='color:#ef4444; font-weight:bold;'>❌ Cod original</span>", unsafe_allow_html=True)
                                        st.code(before_code, language="python")
                                    with col_a:
                                        st.markdown("<span style='color:#22c55e; font-weight:bold;'>✅ Cod refactorizat</span>", unsafe_allow_html=True)
                                        st.code(after_code, language="python")
                                else:
                                    st.code(chunk["content"][:600], language="python", line_numbers=True)
                                st.write("---")
            else:
                st.info("Apasă **Rulează Analiza Code Smells** pentru a începe.")

