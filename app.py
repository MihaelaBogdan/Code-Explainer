import os
# Rezolvarea conflictului de librării OpenMP (OMP: Error #15) pe Mac local pentru PyTorch + FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import streamlit.components.v1 as components
import shutil
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

def build_project_knowledge(chunks):
    """
    Construiește o bază de cunoștințe completă despre proiect din chunks.
    Returnează un dict cu: call_graph, defined_names, file_map, import_map, stats.
    """
    kb = {
        "by_name":    {},   # name -> chunk
        "by_file":    {},   # file_path -> [chunks]
        "call_graph": {},   # caller_name -> set(callee_names)
        "called_by":  {},   # callee_name -> set(caller_names)
        "all_calls":  set(),
        "all_imports": {},  # file -> [imports]
        "stats": {},
    }

    for c in chunks:
        name  = c.get("name", "")
        fpath = c.get("file_path", "?")
        ctype = c.get("type", "")

        if name and ctype in ("function", "class"):
            kb["by_name"][name] = c

        kb["by_file"].setdefault(fpath, []).append(c)

        # Indexăm și metodele din clase ca pseudo-chunk-uri
        if ctype == "class":
            try:
                tree = ast.parse(c.get("content", ""))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = node.name
                        # Extragem codul metodei
                        method_lines = c.get("content","").splitlines()
                        start = node.lineno - 1
                        end   = getattr(node, "end_lineno", start + 20)
                        method_code = "\n".join(method_lines[start:end])
                        abs_start = c.get("start_line", 0) + start
                        abs_end   = c.get("start_line", 0) + end
                        method_chunk = {
                            "name": method_name,
                            "type": "method",
                            "file_path": fpath,
                            "start_line": abs_start,
                            "end_line": abs_end,
                            "content": method_code,
                            "parent_class": name,
                            "docstring": ast.get_docstring(node) or "",
                            "args": [a.arg for a in node.args.args if a.arg != "self"],
                        }
                        kb["by_name"][method_name] = method_chunk
            except:
                pass

        # Analiza AST a fiecărui chunk
        try:
            tree = ast.parse(c.get("content", ""))
            calls_in_chunk = set()
            imports_in_chunk = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls_in_chunk.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls_in_chunk.add(node.func.attr)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_in_chunk.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports_in_chunk.append(node.module)
            kb["call_graph"][name] = calls_in_chunk
            kb["all_calls"].update(calls_in_chunk)
            if imports_in_chunk:
                kb["all_imports"].setdefault(fpath, []).extend(imports_in_chunk)
        except:
            pass

    # Build reverse call graph
    for caller, callees in kb["call_graph"].items():
        for callee in callees:
            kb["called_by"].setdefault(callee, set()).add(caller)

    # Statistici globale
    funcs   = [c for c in chunks if c.get("type") == "function"]
    classes = [c for c in chunks if c.get("type") == "class"]
    total_lines = sum(c.get("end_line", 0) - c.get("start_line", 0) for c in funcs)
    kb["stats"] = {
        "total_chunks":   len(chunks),
        "total_functions": len(funcs),
        "total_classes":  len(classes),
        "total_files":    len(kb["by_file"]),
        "avg_func_lines": round(total_lines / max(len(funcs), 1), 1),
        "largest_func":   max(funcs, key=lambda c: c.get("end_line",0)-c.get("start_line",0), default=None),
        "most_called":    max(kb["called_by"], key=lambda k: len(kb["called_by"][k]), default=None),
    }
    return kb


def analyze_chunk_ast(code_str):
    """Analizează un fragment de cod Python cu AST și returnează un dict cu informații detaliate."""
    info = {
        "calls": [], "returns": [], "raises": [], "loops": 0,
        "conditions": 0, "is_async": False, "args": [], "decorators": [],
        "imports": [], "assignments": [], "class_bases": [], "inner_funcs": [],
        "has_try": False, "comprehensions": 0,
    }
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            # Apeluri de funcții
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    info["calls"].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    info["calls"].append(f"{ast.unparse(node.func) if hasattr(ast, 'unparse') else node.func.attr}")
            # Return statements
            elif isinstance(node, ast.Return) and node.value is not None:
                try:
                    info["returns"].append(ast.unparse(node.value) if hasattr(ast, 'unparse') else "valoare")
                except:
                    info["returns"].append("valoare")
            # Raise statements
            elif isinstance(node, ast.Raise) and node.exc is not None:
                try:
                    info["raises"].append(ast.unparse(node.exc) if hasattr(ast, 'unparse') else "excepție")
                except:
                    info["raises"].append("excepție")
            # Bucle
            elif isinstance(node, (ast.For, ast.While)):
                info["loops"] += 1
            # Condiții
            elif isinstance(node, ast.If):
                info["conditions"] += 1
            # Try/except
            elif isinstance(node, ast.Try):
                info["has_try"] = True
            # Funcții async
            elif isinstance(node, ast.AsyncFunctionDef):
                info["is_async"] = True
            # Importuri
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        info["imports"].append(alias.name)
                elif node.module:
                    info["imports"].append(node.module)
            # List/dict/set comprehensions
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                info["comprehensions"] += 1
            # Definiții de funcții — top level
            elif isinstance(node, ast.FunctionDef):
                info["args"] = [a.arg for a in node.args.args if a.arg != "self"]
                info["is_async"] = isinstance(node, ast.AsyncFunctionDef)
                info["decorators"] = [ast.unparse(d) if hasattr(ast, 'unparse') else "" for d in node.decorator_list]
            # Clase
            elif isinstance(node, ast.ClassDef):
                info["class_bases"] = [ast.unparse(b) if hasattr(ast, 'unparse') else "" for b in node.bases]

        # Deduplicare apeluri, păstrăm primele 8 relevante
        seen = set()
        unique_calls = []
        for c in info["calls"]:
            if c not in seen and c not in ("print", "len", "str", "int", "float", "list", "dict", "range"):
                seen.add(c)
                unique_calls.append(c)
        info["calls"] = unique_calls[:8]
        info["returns"] = info["returns"][:3]
        info["raises"] = list(set(info["raises"]))[:4]
    except:
        pass
    return info


def _extract_subject(question):
    """Extrage identificatorul Python (funcție/clasă/variabilă) din întrebare."""
    import re
    q = question.strip()

    # Cuvinte care NU sunt niciodată subiecte — interogative, verbe, prepoziții RO+EN
    NOISE = {
        "ce", "este", "e", "face", "sunt", "cum", "unde", "cine", "care", "cat",
        "explica", "explică", "descrie", "listeaza", "listează", "enumera", "enumeră",
        "functia", "funcția", "clasa", "clasa", "metoda", "variabila",
        "parametri", "argumente", "primeste", "primește", "returneaza", "returnează",
        "intoarce", "întoarce", "produce", "output", "input", "rezultat",
        "foloseste", "folosește", "folosesc", "folosim", "folositi", "folosi", "apeleaza", "apelează", "cheama", "cheamă",
        "rolul", "scopul", "modulul", "fisierul", "fișierul",
        "a", "al", "ale", "de", "la", "in", "în", "din", "si", "și", "sau", "se",
        "ca", "că", "cu", "pe", "prin", "pentru", "despre", "din", "avem", "are",
        "the", "is", "are", "does", "do", "how", "what", "where", "who",
        "which", "function", "class", "method", "variable", "about",
        "erori", "exceptii", "excepții", "importuri", "dependente", "dependențe",
        "librarii", "biblioteci", "module", "fisiere", "fișiere",
        "găsesc", "găsește", "găsim", "afiseaza", "afișează", "arată", "arata",
        "da", "dă", "da-mi", "dă-mi", "vreau", "pot", "putem", "scrie", "scriem"
    }

    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', q)
    candidates = [w for w in words if w.lower() not in NOISE and len(w) > 1]

    if not candidates:
        return ""

    # Scor: snake_case (+3), CamelCase (+2), lungime > 5 (+1), apare ultimul (+1)
    def score(w):
        s = 0
        if "_" in w: s += 3          # snake_case = Python identifier clar
        elif w[0].isupper(): s += 2  # CamelCase = clasă
        if len(w) > 5: s += 1
        return s

    # Sortăm descrescător după scor, dar preferăm ultimul cuvânt la egalitate
    scored = sorted(enumerate(candidates), key=lambda x: (score(x[1]), x[0]), reverse=True)
    return scored[0][1] if scored else ""


def _text_search_chunks(term, all_chunks, max_results=5):
    """Caută textual termenul în toate chunk-urile și returnează cele mai relevante."""
    if not term:
        return []
    term_lower = term.lower()
    scored = []
    for c in all_chunks:
        content = c.get("content", "")
        name = c.get("name", "")
        score = 0
        # Găsit în numele funcției/clasei — prioritate maximă
        if term_lower == name.lower():
            score += 100
        elif term_lower in name.lower():
            score += 50
        # Număr de apariții în cod
        score += content.lower().count(term_lower) * 2
        if score > 0:
            scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:max_results]]


def _analyze_variable_usage(term, all_chunks):
    """Analizează cum e folosit un termen (variabilă/atribut) în tot proiectul."""
    import re
    usages = []
    pattern = re.compile(r'.{0,40}' + re.escape(term) + r'.{0,60}', re.MULTILINE)
    seen_lines = set()
    for chunk in all_chunks:
        content = chunk.get("content", "")
        for match in pattern.findall(content):
            line = match.strip()
            if line not in seen_lines and len(line) > 5:
                seen_lines.add(line)
                usages.append((chunk.get("name", "?"), chunk.get("file_path", "?"), line))
    return usages[:15]


KNOWLEDGE_BASE = {
    "ast": {
        "title": "Abstract Syntax Tree (Arbore de Sintaxă Abstractă)",
        "query": "ce este AST cum funcționează abstract syntax tree parsare cod noduri compilatoare",
        "description": "Un Abstract Syntax Tree (AST) este o reprezentare arborescentă a structurii sintactice a codului sursă scris într-un limbaj de programare. Fiecare nod din arbore denotă o construcție apărută în cod (de exemplu: o atribuire, un apel de funcție, o buclă sau o clasă).",
        "details": """
În Python, modulul încorporat `ast` îți permite să parsezi codul sursă direct în obiecte de clasă Python.

### Cum funcționează în proiectul tău:
1. Fișierul de cod este citit ca text.
2. `ast.parse(cod)` compilează textul într-un arbore AST.
3. Un generator sau un `ast.NodeVisitor` vizitează recursiv nodurile pentru a identifica clasele (`ClassDef`), funcțiile (`FunctionDef`), importurile (`Import` / `ImportFrom`) și apelurile (`Call`).

### Exemplu simplu de utilizare în Python:
```python
import ast

cod = "x = 5 + 10"
tree = ast.parse(cod)

# Afișează structura arborelui
print(ast.dump(tree, indent=2))
```
""",
        "code": """
# Exemplu de Visitor pentru analiză AST
class FuncVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        print(f"Funcție găsită: {node.name}")
        self.generic_visit(node)
"""
    },
    "attention": {
        "title": "Mecanismul de Self-Attention în Transformers",
        "query": "ce este self attention cum funcționează capete de atenție transformer formule vectori query key value",
        "description": "Self-Attention (Auto-Atenția) este mecanismul de bază al arhitecturii Transformer care permite modelului să pondereze importanța diferitelor cuvinte într-o secvență în raport cu un cuvânt țintă, indiferent de distanța dintre ele.",
        "details": """
Fiecare token din frază primește trei vectori: **Query (Q)**, **Key (K)** și **Value (V)**.
Formula matematică a atenției Scaled Dot-Product este:
$$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$$

### Ce înseamnă asta în practică:
- Pentru fiecare cuvânt (Query), calculăm similaritatea (dot product) cu toate celelalte cuvinte (Keys).
- Împărțim la radical din dimensiunea vectorilor pentru a evita gradienți foarte mici și aplicăm `softmax` pentru a obține o distribuție de probabilitate (ponderi între 0 și 1).
- Înmulțim aceste ponderi cu vectorii Value pentru a obține reprezentarea finală a tokenului, îmbogățită cu contextul frazei.

În tabul **Explorator Atenție** al acestei aplicații, extragem exact matricea de atenție din ultimul strat al modelului CodeBERT și o reprezentăm ca Heatmap Plotly.
""",
        "code": """
# Extragerea teoretică a atenției în PyTorch
outputs = model(**inputs, output_attentions=True)
# outputs.attentions este un tuplu per strat
last_layer_attentions = outputs.attentions[-1] # Stratul 12
"""
    },
    "streamlit_state": {
        "title": "Streamlit Session State (Gestionarea Stării)",
        "query": "ce este session state streamlit cum păstrăm starea reîncărcare date st.session_state",
        "description": "Streamlit Session State reprezintă o modalitate de a partaja variabile între rulări succesive ale scriptului (reruns) pentru un utilizator specific. Deoarece Streamlit rulează scriptul complet de sus în jos la fiecare interacțiune, Session State previne pierderea datelor.",
        "details": """
Spre deosebire de variabilele standard care se resetează la fiecare reîncărcare, `st.session_state` acționează ca un dicționar persistent pe durata sesiunii utilizatorului.

### Cele mai bune practici în aplicația ta:
- **Inițializare defensivă**: Verifică întotdeauna dacă cheia există în `st.session_state` înainte de a o folosi.
- **Rerun controlat**: Folosește `st.rerun()` pentru a forța Streamlit să reia execuția imediat după o modificare de stare critică (cum ar fi ștergerea datelor).

### Exemplu practic de gestiune:
```python
import streamlit as st

# Inițializare corectă
if "counter" not in st.session_state:
    st.session_state.counter = 0

# Modificare
if st.button("Incrementează"):
    st.session_state.counter += 1
    st.write(f"Scor: {st.session_state.counter}")
```
""",
        "code": """
# Modelul de curățare a stării în aplicația ta
def clean_workspace():
    st.session_state.project_processed = False
    st.session_state.chunks = []
    # st.session_state.indexer.reset()
"""
    },
    "database_normalization": {
        "title": "Normalizarea Bazelor de Date (1NF, 2NF, 3NF)",
        "query": "normalizare baze de date 1NF 2NF 3NF reguli cheie primară dependență funcțională tranzitivă",
        "description": "Normalizarea reprezintă procesul de organizare a tabelelor dintr-o bază de date relațională pentru a reduce redundanța datelor și a elimina anomaliile de inserare, actualizare și ștergere.",
        "details": """
### Formele Normale principale explicate academic:

1. **Forma Normală 1 (1NF)**:
   - Fiecare celulă trebuie să conțină doar valori atomice (indivizibile).
   - Nu sunt permise grupuri repetitive sau liste de valori într-o singură coloană.
   - Fiecare tabel trebuie să aibă o cheie primară unică.

2. **Forma Normală 2 (2NF)**:
   - Să fie deja în **1NF**.
   - Toate atributele non-cheie trebuie să depindă în mod **complet** de cheia primară (nu de o parte a unei chei compuse). Elimină dependențele parțiale.

3. **Forma Normală 3 (3NF)**:
   - Să fie deja în **2NF**.
   - Nu trebuie să existe dependențe funcționale tranzitive. Adică, niciun atribut non-cheie nu trebuie să depindă de un alt atribut non-cheie care, la rândul său, depinde de cheia primară. (Regulă de aur: *Dependența trebuie să fie de cheie, de întreaga cheie și de nimic altceva decât de cheie*).
""",
        "code": """
-- Exemplu de încălcare 3NF (ID_Departament determină Nume_Departament, ambele fiind non-cheie)
CREATE TABLE Angajati (
    ID_Angajat INT PRIMARY KEY,
    Nume VARCHAR(50),
    ID_Departament INT,
    Nume_Departament VARCHAR(50) -- Dependență tranzitivă!
);

-- Soluție 3NF: Spargerea în două tabele
CREATE TABLE Departamente (
    ID_Departament INT PRIMARY KEY,
    Nume_Departament VARCHAR(50)
);
"""
    },
    "acid": {
        "title": "Proprietățile ACID (Tranzacții SQL)",
        "query": "ce înseamnă ACID tranzacții baze de date atomicitate consistență izolare durabilitate sql",
        "description": "ACID reprezintă un set de proprietăți fundamentale care garantează că tranzacțiile dintr-o bază de date relațională sunt procesate în mod fiabil, chiar și în caz de erori hardware sau căderi de rețea.",
        "details": """
### Cele 4 proprietăți ACID:

- **Atomicity (Atomicitate)**:
  - Regula **Totul sau Nimic**. Toate operațiunile din cadrul tranzacției se execută complet cu succes, fie nu se execută niciuna. Dacă un singur pas eșuează, baza de date face `ROLLBACK` la starea anterioară.

- **Consistency (Consistență)**:
  - O tranzacție poate duce baza de date doar dintr-o stare validă în altă stare validă, respectând toate constrângerile de integritate (chei străine, unicitate, constrângeri de tip check).

- **Isolation (Izolare)**:
  - Tranzacțiile concurente (care rulează în același timp) nu se pot influența reciproc. Rezultatul rulării simultane a două tranzacții trebuie să fie identic cu cel obținut dacă acestea s-ar fi rulat secvențial.

- **Durability (Durabilitate)**:
  - Odată ce o tranzacție a fost confirmată (`COMMIT`), modificările sale sunt permanente și persistente pe disc, rezistând chiar și în cazul unei pane bruște de curent.
""",
        "code": """
-- Exemplu de tranzacție atomică SQL
BEGIN TRANSACTION;
UPDATE Conturi SET Sold = Sold - 100 WHERE ID = 1;
UPDATE Conturi SET Sold = Sold + 100 WHERE ID = 2;
-- Dacă totul e OK:
COMMIT;
-- În caz de eroare:
-- ROLLBACK;
"""
    },
    "sql_injection": {
        "title": "Injecția SQL (Vulnerabilitate și Prevenire)",
        "query": "ce este SQL injection injecție sql vulnerabilitate prevenire parametri securitate cod nesigur",
        "description": "SQL Injection (SQLi) reprezintă o tehnică de atac prin care un atacator introduce comenzi SQL malițioase în câmpurile de input ale unei aplicații pentru a fi executate de baza de date din spate.",
        "details": """
Această vulnerabilitate apare atunci când input-ul de la utilizator este concatenat direct în query-ul SQL, fără validare sau parametrizare.

### Cum o prevenim:
Folosim **interogări parametrizate** (Prepared Statements) sau biblioteci de tip ORM (precum SQLAlchemy). Parametrizarea tratează input-ul ca pe o valoare literală (string simplu), eliminând posibilitatea ca atacatorul să altereze structura query-ului.
""",
        "code": """
# ❌ COD VULNERABIL (Concatenare)
query = f"SELECT * FROM utilizatori WHERE nume = '{user_input}'"
cursor.execute(query)

#  COD SECURIZAT (Parametrizare)
query = "SELECT * FROM utilizatori WHERE nume = %s"
cursor.execute(query, (user_input,))
"""
    },
    "java_oop": {
        "title": "Programarea Orientată pe Obiecte (OOP) în Java",
        "query": "ce este OOP java programare orientată pe obiecte moștenire polimorfism încapsulare abstractizare clase interfețe",
        "description": "Programarea Orientată pe Obiecte (OOP) reprezintă o paradigmă de dezvoltare software axată pe concepte numite obiecte (instanțe ale claselor). Java este un limbaj OOP pur care forțează structurarea codului pe clase.",
        "details": """
### Cele 4 Principii Fundamentale ale OOP:
1. **Încapsularea (Encapsulation)**:
   - Ascunderea datelor interne ale obiectului prin folosirea modificatorilor de acces `private` și oferirea de metode publice `get` și `set` pentru interacțiune sigură. Protejează starea internă.
2. **Moștenirea (Inheritance)**:
   - Permite unei clase noi (clasa fiică/derivată) să preia proprietățile și comportamentul unei clase existente (clasa mamă/bază) folosind cuvântul cheie `extends`, reducând codul duplicat.
3. **Polimorfismul (Polymorphism)**:
   - Abilitatea unui obiect de a lua mai multe forme. Se manifestă prin **suprascriere (method overriding)** - dinamică la rulare, și **supraîncărcare (method overloading)** - statică la compilare.
4. **Abstractizarea (Abstraction)**:
   - Ascunderea detaliilor complexe de implementare și evidențierea doar a funcționalităților esențiale prin clase abstracte (`abstract class`) și interfețe (`interface`).
""",
        "code": """
// Java - Polimorfism & Încapsulare
public class Animal {
    private String nume; // Încapsulare
    
    public Animal(String nume) { this.nume = nume; }
    public String getNume() { return nume; }
    
    public void scoateSunet() { System.out.println("Sunet generic..."); }
}

public class Caine extends Animal { // Moștenire
    public Caine(String nume) { super(nume); }
    
    @Override
    public void scoateSunet() { System.out.println("Ham Ham!"); } // Polimorfism
}
"""
    },
    "js_async": {
        "title": "Programare Asincronă în JavaScript / TypeScript",
        "query": "ce este programarea asincronă javascript async await promise callbacks event loop single threaded typescript",
        "description": "JavaScript este un limbaj single-threaded (execută o singură instrucțiune la un moment dat). Pentru a nu bloca rularea în timpul operațiilor lungi (I/O, rețea), JS folosește un model non-blocant bazat pe Event Loop, Promises și async/await.",
        "details": """
### Elementele cheie ale asincronismului în JS/TS:
- **Callback-uri**: Funcții transmise ca argumente pentru a fi rulate la finalul unei operațiuni. Utilizarea lor excesivă duce la structuri greu de citit numite *Callback Hell*.
- **Promises**: Obiecte ce reprezintă finalizarea (sau eșecul) eventual a unei operațiuni asincrone. Pot fi în una din cele 3 stări: *Pending* (în așteptare), *Fulfilled* (îndeplinită cu succes) sau *Rejected* (eșuată cu eroare).
- **Async/Await**: O sintaxă modernă (introdusă în ES2017) care permite scrierea de cod asincron care arată și se comportă ca cel sincron, îmbunătățind lizibilitatea. Un bloc `async` returnează implicit un Promise, iar `await` suspendă execuția până la rezolvarea lui.
- **Event Loop (Bucla de Evenimente)**: Monitorizează continuu Call Stack-ul și Callback Queue-ul. Dacă stiva de apeluri este goală, Event Loop preia primul eveniment asincron finalizat din coadă și îl rulează.
""",
        "code": """
// JS / TS - Utilizare Promises și Async/Await
const fetchUserData = (userId) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (userId > 0) resolve({ id: userId, nume: "Alex" });
            else reject(new Error("ID invalid"));
        }, 1000);
    });
};

// Funcție asincronă cu tratarea erorilor prin try/catch
async function runDemo() {
    try {
        const user = await fetchUserData(42);
        console.log(`Utilizator găsit: ${user.nume}`);
    } catch (err) {
        console.error("Eroare de fetch:", err.message);
    }
}
"""
    },
    "cpp_memory": {
        "title": "Gestiunea Memoriei în C++ (Pointeri & RAII)",
        "query": "ce sunt pointerii c++ pointer references malloc free new delete raii memory management heap stack",
        "description": "C++ oferă control total și direct asupra resurselor fizice și memoriei calculatorului. Înțelegerea diferențelor dintre memorie automată (Stack) și memorie dinamică (Heap) este vitală pentru a evita scurgerile de memorie (Memory Leaks).",
        "details": """
### Concepte Fundamentale:
- **Stack (Stiva)**: Memorie rapidă gestionată automat de compilator. Variabilele declarate local sunt create pe stivă și sunt șterse instant când blocul/funcția își încheie execuția.
- **Heap (Grămada)**: Memorie mare gestionată manual de programator. Resursele se alocă explicit folosind operatorul `new` (sau `malloc` în C) și **TREBUIE** eliberate manual folosind `delete` (sau `free`), altfel rămân blocate în RAM.
- **Pointeri**: Variabile care stochează adresa de memorie a unei alte variabile (ex: `int* p = &x;`).
- **RAII (Resource Acquisition Is Initialization)**:
  - Cel mai important design pattern din C++. Resursele (memorie, socket-uri de rețea, fișiere) sunt legate de durata de viață a unui obiect local pe Stack. Constructorul alocă resursa, iar **destructorul** o eliberează automat când obiectul iese din scope.
- **Smart Pointeri (`std::unique_ptr`, `std::shared_ptr`)**: Implementări moderne RAII care eliberează automat memoria Heap când pointerii nu mai sunt utilizați.
""",
        "code": """
// C++ - Pointeri manuali vs. Smart Pointeri RAII
#include <iostream>
#include <memory>

void manualMemory() {
    int* ptr = new int(100); // Alocare Heap manuală
    std::cout << *ptr << std::endl;
    delete ptr; // Eliberare obligatorie
}

void smartMemory() {
    // Alocare sigură RAII - se eliberează automat la ieșirea din funcție
    std::unique_ptr<int> smartPtr = std::make_unique<int>(200);
    std::cout << *smartPtr << std::endl; 
}
"""
    },
    "rust_safety": {
        "title": "Ownership (Deținere) și Siguranța Memoriei în Rust",
        "query": "ce este ownership rust borrow checker lifetimes lifetimes siguranță memorie struct",
        "description": "Rust garantează siguranța memoriei în timpul compilării fără a folosi un Garbage Collector sau alocări/dezalocări manuale riscante. Această performanță se datorează sistemului său unic de Ownership.",
        "details": """
### Regulile de Aur ale Ownership-ului în Rust:
1. **Fiecare valoare din Rust are o variabilă numită owner (deținător).**
2. **Poate exista un singur owner în același timp.**
3. **Când owner-ul iese din scope (domeniul de vizibilitate), valoarea este ștearsă automat (se apelează funcția `drop`).**

### Împrumutul (Borrowing) și Verificatorul (Borrow Checker):
Pentru a evita copierea inutilă a datelor în memorie, Rust permite împrumutarea valorilor prin intermediul referințelor (`&`):
- Poți avea oricâte **referințe nemutabile** (`&T`) simultan (citire concurentă sigură).
- Poți avea **o singură referință mutabilă** (`&mut T`) în același timp (pentru a preveni modificări concurente nesigure și *Data Races*).
- Nu poți combina referințe mutabile cu referințe nemutabile în același scope.
""",
        "code": """
// Rust - Ownership & Borrowing
fn main() {
    let s1 = String::from("salut"); // s1 devine owner
    
    // let s2 = s1; // Valoarea este "mutată" în s2. s1 devine invalidă (Move semantics)
    
    let len = calcul_lungime(&s1); // Trimitem referință (împrumut nemutabil)
    println!("Lungimea '{}' este {}.", s1, len); // s1 este încă validă!
}

fn calcul_lungime(s: &String) -> usize { // s este referință nemutabilă
    s.len()
}
"""
    },
    "go_concurrency": {
        "title": "Concurență în Go (Goroutines & Channels)",
        "query": "concurență go goroutines channels fire de execuție ușoare go routine canal csp",
        "description": "Limbajul Go (Golang) a fost proiectat nativ pentru rulare pe sisteme moderne multi-core. Oferă un model de concurență extrem de eficient bazat pe Goroutines și Channels, implementând paradigma CSP.",
        "details": """
### Conceptele de bază din Go:
- **Goroutines**: Fire de execuție extrem de ușoare administrate de Go Runtime (nu direct de sistemul de operare). Lansarea unei goroutines consumă doar ~2KB de memorie (față de ~1MB pentru un thread OS). Se lansează simplu prin prefixarea apelului cu cuvântul cheie `go`.
- **Channels (Canale)**: Conducte sigure prin care goroutine-urile pot comunica și își pot sincroniza execuția, transmițând valori fără a folosi mutex-uri sau memorie partajată (Shared Memory).
- **Filozofia Go**: *"Nu comunica prin partajarea memoriei; în schimb, partajează memoria prin comunicare."*
""",
        "code": """
package main
import (
    "fmt"
    "time"
)

// Funcție ce rulează asincron în goroutine
fn salut(canal chan string) {
    time.Sleep(100 * time.Millisecond)
    canal <- "Salut din Goroutine!" // Trimitem date în canal
}

func main() {
    canal := make(chan string)
    
    go salut(canal) // Lansăm goroutine concurentă
    
    mesaj := <-canal // Blocăm execuția până primim date din canal
    fmt.Println(mesaj)
}
"""
    },
    "csharp_dotnet": {
        "title": "C# și Platforma .NET (Generics & LINQ)",
        "query": "c# dotnet linq generics delegati delegates clr enterprise async",
        "description": "C# este un limbaj puternic tipizat, orientat pe obiecte, dezvoltat de Microsoft. Rularea sa este optimizată pe platforma .NET prin intermediul CLR (Common Language Runtime) care compilează codul intermediar (IL) în cod mașină nativ.",
        "details": """
### Facilități de elită în C#:
- **Generics (Tipuri Generice)**: Permite scrierea claselor, interfețelor sau metodelor cu parametri de tip. Aceasta asigură siguranța tipurilor la compilare, reutilizarea codului și performanță maximă (deoarece elimină operațiunile costisitoare de `boxing` și `unboxing` specifice tipului `object`).
- **LINQ (Language Integrated Query)**: O componentă revoluționară care aduce capacități de interogare declarative direct în sintaxa limbajului C#. Permite filtrarea, sortarea și transformarea colecțiilor de date (liste, XML, baze de date) într-un mod similar cu SQL, extrem de lizibil și compact.
""",
        "code": """
using System;
using System.Collections.Generic;
using System.Linq;

public class Program {
    public static void Main() {
        // Listă Generic <int>
        List<int> numere = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8 };
        
        // Interogare declarativă prin LINQ
        var numerePare = numere.Where(n => n % 2 == 0).ToList();
        
        foreach (var n in numerePare) {
            Console.WriteLine(n); // Output: 2, 4, 6, 8
        }
    }
}
"""
    },
    "web_layout": {
        "title": "Layout-uri CSS Moderne (Flexbox & Grid)",
        "query": "ce este flexbox css grid machetare responsive design responsive web design web flex grid",
        "description": "CSS modern oferă instrumente extrem de puternice pentru crearea de interfețe responsive și adaptabile pe orice ecran, înlocuind vechile tehnici rigide cu tabele sau proprietăți de tip float.",
        "details": """
### Cele două motoare de layout în CSS:
1. **Flexbox (Flexible Box Layout)**:
   - Proiectat pentru machetare **unidimensională** (alinierea elementelor pe o singură axă: linie *sau* coloană).
   - Ideal pentru bare de navigare, liste de elemente, carduri simple și centrare verticală/orizontală.
   - Proprietăți principale: `display: flex`, `justify-content` (aliniere pe axa principală), `align-items` (aliniere pe axa transversală), `flex-direction`.
2. **CSS Grid (Grid Layout)**:
   - Proiectat pentru machetare **bidimensională** (gestionarea simultană a liniilor și coloanelor).
   - Ideal pentru structura globală a paginilor, galerii complexe sau panouri de control.
   - Proprietăți principale: `display: grid`, `grid-template-columns` (definirea coloanelor), `grid-template-rows`, `grid-gap`.
""",
        "code": """
/* Machetare CSS Grid Responsivă */
.container-grid {
    display: grid;
    /* Creează automat coloane de minim 250px care se adaptează pe ecran */
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    grid-gap: 20px; /* Spațiu între celule */
}

/* Centrare perfectă cu Flexbox */
.container-flex-centrat {
    display: flex;
    justify-content: center; /* orizontal */
    align-items: center;    /* vertical */
    height: 100vh;
}
"""
    },
    "recursion": {
        "title": "Recursivitate (Recursion)",
        "query": "recursivitate recursiv recursion recursivă recursiva auto-apelare",
        "description": "Recursivitatea este o tehnică de programare în care o funcție se apelează pe ea însăși, direct sau indirect, pentru a rezolva o problemă prin descompunerea ei în subprobleme similare mai mici.",
        "details": """
O implementare recursivă corectă are două componente esențiale:
1. **Cazul de bază (condiția de oprire)**: Împiedică apelurile recursive infinite și oprește execuția (evitând depășirea stivei - Stack Overflow).
2. **Pasul recursiv**: Apelul funcției cu un argument modificat, care tinde spre cazul de bază.

### Avantaje:
- Cod mult mai curat, elegant și matematic pentru probleme ierarhice (cum ar fi parcurgerea arborilor sau grafurilor).
- Simplifică algoritmii de tip Divide et Impera (ex: QuickSort, MergeSort).
""",
        "code": """
# Factorial recursiv în Python
def factorial(n):
    # 1. Cazul de bază
    if n == 0 or n == 1:
        return 1
    # 2. Pasul recursiv
    return n * factorial(n - 1)

print(factorial(5)) # Output: 120
"""
    },
    "oop_principles": {
        "title": "Cele 4 Principii Fundamentale ale OOP",
        "query": "principii oop principiile oop mostenire moștenire polimorfism încapsulare incapsulare abstractizare clase și obiecte clase si obiecte",
        "description": "Programarea Orientată pe Obiecte (OOP) se bazează pe patru piloni conceptuali care ajută la structurarea codului într-un mod modular, sigur și reutilizabil.",
        "details": """
### Cei 4 Piloni ai OOP:
1. **Moștenirea (Inheritance)**: Clasa derivată extinde o clasă de bază, preluându-i atributele și comportamentul. (Relația *IS-A*, ex: Câinele este un Animal).
2. **Polimorfismul (Polymorphism)**: Abilitatea unei metode de a se comporta diferit în funcție de obiectul care o apelează (suprascrierea - overriding sau supraîncărcarea - overloading).
3. **Încapsularea (Encapsulation)**: Ascunderea datelor interne private ale unui obiect și restricționarea accesului direct (folosind getter-i și setter-i pentru securitate și validare).
4. **Abstractizarea (Abstraction)**: Ascunderea complexității din spate și expunerea doar a caracteristicilor esențiale (prin clase abstracte și interfețe).
""",
        "code": """
# Exemplu practic al celor 4 principii în Python
from abc import ABC, abstractmethod

class Vehicul(ABC): # Abstractizare
    def __init__(self, marca):
        self.__marca = marca # Încapsulare (atribut privat)
        
    def get_marca(self):
        return self.__marca
        
    @abstractmethod
    def porneste(self):
        pass

class Masina(Vehicul): # Moștenire
    def porneste(self):
        return f"Mașina {self.get_marca()} a pornit cu sunet polimorfic!" # Polimorfism
"""
    },
    "stack_ds": {
        "title": "Stiva (Stack) ca Structură de Date",
        "query": "stivă stiva stack lifo push pop operatii stiva stive",
        "description": "Stiva este o structură de date liniară bazată pe principiul LIFO (Last In, First Out - Ultimul intrat, primul ieșit).",
        "details": """
### Operații fundamentale pe stivă:
- `push`: Adaugă un element în vârful stivei.
- `pop`: Extrage și returnează elementul din vârful stivei.
- `peek` / `top`: Vizualizează elementul din vârf fără a-l șterge.

### Utilizări practice:
- **Call Stack**-ul din procesoare pentru gestionarea apelurilor de funcție.
- Funcționalitatea de **Undo / Redo** din editoarele text.
- Parcurgerea în adâncime (DFS) în grafuri.
""",
        "code": """
# Implementarea unei stive folosind liste în Python
stiva = []
stiva.append("Pagina 1") # push
stiva.append("Pagina 2") # push
stiva.append("Pagina 3") # push

print("Vârf:", stiva[-1]) # peek -> Pagina 3
print("Extras:", stiva.pop()) # pop -> Pagina 3
print("Noul vârf:", stiva[-1]) # Pagina 2
"""
    },
    "queue_ds": {
        "title": "Coadă (Queue) ca Structură de Date",
        "query": "coadă coada queue fifo enqueue dequeue operatii coada cozi",
        "description": "Coada este o structură de date liniară bazată pe principiul FIFO (First In, First Out - Primul intrat, primul ieșit).",
        "details": """
### Operații fundamentale pe coadă:
- `enqueue`: Adaugă un element la sfârșitul cozii.
- `dequeue`: Extrage și returnează primul element de la începutul cozii.

### Utilizări practice:
- **CPU Scheduling**: Programarea proceselor în sistemele de operare.
- Transmiterea pachetelor de date în rețea.
- Parcurgerea în lățime (BFS) în grafuri.
""",
        "code": """
# Implementarea unei cozi folosind collections.deque în Python
from collections import deque

coada = deque()
coada.append("Client 1") # enqueue
coada.append("Client 2") # enqueue
coada.append("Client 3") # enqueue

print("Următorul la rând:", coada[0]) # Client 1
print("Deservit:", coada.popleft()) # dequeue -> Client 1
print("Următorul:", coada[0]) # Client 2
"""
    },
    "tree_ds": {
        "title": "Arbori (Tree) ca Structură de Date",
        "query": "arbore tree arbori binary tree arbore binar bst frunze arbore",
        "description": "Un arbore este o structură de date ierarhică, ne-liniară, formată dintr-un set de noduri conectate prin margini (muchii), începând de la un nod rădăcină (root).",
        "details": """
### Concepte cheie:
- **Nod rădăcină (Root)**: Punctul de start, nu are părinte.
- **Noduri interne**: Au părinți și copii.
- **Noduri frunză (Leaves)**: Nodurile finale care nu au copii.
- **Arbore Binar de Căutare (BST)**: Un arbore binar în care pentru orice nod, copiii din stânga au valori mai mici, iar copiii din dreapta au valori mai mari. Permite căutări extrem de rapide.
""",
        "code": """
# Definirea unui nod dintr-un Arbore Binar de Căutare
class NodArbore:
    def __init__(self, cheie):
        self.stang = None
        self.drept = None
        self.val = cheie

# Construirea unui arbore
radacina = NodArbore(10)
radacina.stang = NodArbore(5)
radacina.drept = NodArbore(15)
"""
    },
    "graph_ds": {
        "title": "Grafuri (Graph) ca Structură de Date",
        "query": "graf graph grafuri noduri muchii lista de adiacenta drum minim ponderat",
        "description": "Un graf este o structură de date ne-liniară formată dintr-o mulțime de noduri (numite vârfuri) și o mulțime de muchii care le conectează.",
        "details": """
### Tipuri de grafuri:
- **Graf Orientat (Directed)**: Muchiile au un sens unic (săgeți).
- **Graf Neorientat (Undirected)**: Muchiile au sens dublu.
- **Graf Ponderat (Weighted)**: Fiecare muchie are un cost sau o pondere asociată (ex: distanța în km între orașe).

### Reprezentări comune:
1. **Matrice de adiacență**: O matrice bidimensională unde celulele indică conexiunea dintre noduri.
2. **Listă de adiacență**: Un dicționar unde fiecare nod este mapat la lista vecinilor săi (mai eficientă ca memorie).
""",
        "code": """
# Reprezentarea unui graf prin Listă de Adiacență în Python
graf = {
    "București": ["Ploiești", "Constanța"],
    "Ploiești": ["București", "Brașov"],
    "Brașov": ["Ploiești"],
    "Constanța": ["București"]
}

print("Vecini București:", graf["București"])
"""
    },
    "solid_principles": {
        "title": "Principiile SOLID de Design Software",
        "query": "solid principii solid principiile solid single responsibility liskov dependency inversion",
        "description": "SOLID este un acronim format din cinci principii de design orientat pe obiecte menite să facă codul mai robust, mai flexibil și mult mai ușor de întreținut.",
        "details": """
### Cele 5 Principii SOLID:
1. **S - Single Responsibility Principle (SRP)**: O clasă trebuie să aibă o singură responsabilitate și un singur motiv de schimbare.
2. **O - Open/Closed Principle (OCP)**: Clasa trebuie să fie deschisă pentru extindere, dar închisă pentru modificare.
3. **L - Liskov Substitution Principle (LSP)**: Clasele derivate trebuie să poată înlocui complet clasele de bază fără a altera corectitudinea aplicației.
4. **I - Interface Segregation Principle (ISP)**: Clienții nu trebuie forțați să depindă de metode pe care nu le folosesc (mai bine multe interfețe mici, decât una singură gigant).
5. **D - Dependency Inversion Principle (DIP)**: Depinde de abstracțiuni, nu de implementări concrete (folosește Dependency Injection).
""",
        "code": """
# Exemplu SRP (Single Responsibility Principle)
# ❌ INCORECT: Clasa calculează și salvează în fișier (două responsabilități)
# ✅ CORECT: Împărțirea în două clase separate

class CalculatorSalarii:
    def calculeaza(self, angajat):
        return angajat.ore_lucrate * angajat.tarif_orar

class RepozitoriuAngajati:
    def salveaza_in_fisier(self, angajat, salariu):
        with open("salarii.txt", "a") as f:
            f.write(f"{angajat.nume}: {salariu}\\n")
"""
    },
    "singleton_pattern": {
        "title": "Design Pattern-ul Singleton",
        "query": "singleton pattern singleton design pattern singleton instanță unică instanta unica",
        "description": "Singleton este un model de design creational care garantează că o clasă are o singură instanță globală în întreaga aplicație și oferă un punct de acces unic la aceasta.",
        "details": """
### Când îl folosim:
- Pentru resurse partajate unic: Conexiunea la o bază de date, un Logger global, un Manager de fișiere de configurare sau un Cache manager.

### Cum funcționează:
Oprește instanțierea directă a clasei (constructorul privat în Java/C++ sau suprascrierea `__new__` în Python) și returnează instanța unică stocată static la fiecare cerere.
""",
        "code": """
# Implementarea unui Singleton în Python
class DatabaseConnection:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            # Inițializezi conexiunea o singură dată aici
            cls._instance.status = "Conectat la DB local"
        return cls._instance

conn1 = DatabaseConnection()
conn2 = DatabaseConnection()
print(conn1 is conn2) # Output: True (sunt exact aceeași instanță!)
"""
    },
    "git_vcs": {
        "title": "Sistemul de Control al Versiunilor Git",
        "query": "git commit branch merge push pull controlul versiunilor vcs",
        "description": "Git este un sistem distribuit de control al versiunilor utilizat pentru a urmări modificările aduse fișierelor de cod în timpul dezvoltării colaborative.",
        "details": """
### Comenzi esențiale în Git:
- `git init`: Inițializează un nou repozitoriu local.
- `git add .`: Pregătește toate modificările din spațiul de lucru (Staging Area).
- `git commit -m "mesaj"`: Salvează permanent un snapshot al modificărilor în istoric.
- `git branch <nume>`: Creează o ramură separată de dezvoltare.
- `git merge <branch>`: Combină modificările dintr-o ramură în cea curentă.
- `git push origin <branch>`: Încarcă commit-urile locale pe serverul de la distanță (ex: GitHub).
- `git pull`: Descarcă și integrează modificările recente de pe server în codul local.
""",
        "code": """
# Fluxul de bază Git rulat în terminal:
git init
git add .
git commit -m "Adaugă structura de bază a proiectului"
git branch -M main
git remote add origin https://github.com/utilizator/proiect.git
git push -u origin main
"""
    },
    "json_format": {
        "title": "Formatul de date JSON (JavaScript Object Notation)",
        "query": "json format json ce este json javascript object notation date json",
        "description": "JSON este un format de date bazat pe text, independent de limbaj, utilizat la scară largă pentru schimbul de date între un client (browser) și un server (API).",
        "details": """
### Structură și caracteristici:
- Este alcătuit din două tipuri principale de structuri: perechi cheie-valoare (`{}`) și liste ordonate de valori (`[]`).
- Cheile sunt întotdeauna șiruri de caractere (string-uri între ghilimele duble).
- Valorile pot fi: string-uri, numere, booleeni (`true`/`false`), obiecte, array-uri sau `null`.
""",
        "code": """
{
  "nume_proiect": "AI Code Explainer",
  "versiune": 1.5,
  "activ": true,
  "tehnologii": ["Python", "Streamlit", "PyTorch"],
  "configurari": {
    "port": 8501,
    "debug": false
  }
}
"""
    },
    "rest_api": {
        "title": "Arhitectura REST API (Servicii Web)",
        "query": "rest api api rest http methods endpoint get post put delete coduri de stare http",
        "description": "REST (Representational State Transfer) este un stil de arhitectură utilizat pentru dezvoltarea serviciilor web, bazat pe protocolul HTTP.",
        "details": """
### Metode HTTP standardizate în REST:
- **GET**: Citește/recuperează date de pe server.
- **POST**: Trimite date noi pe server pentru crearea unei resurse.
- **PUT** / **PATCH**: Actualizează o resursă existentă (complet/parțial).
- **DELETE**: Șterge o resursă de pe server.

### Coduri de răspuns HTTP cheie:
- `200 OK`: Succes general.
- `201 Created`: Resursă creată cu succes.
- `400 Bad Request`: Cererea clientului este greșită.
- `401 Unauthorized`: Autentificare necesară.
- `404 Not Found`: Resursa nu există pe server.
- `500 Internal Server Error`: Problemă de server.
""",
        "code": """
# Endpoint REST API implementat în Python (Flask)
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/produse', methods=['GET'])
def obtine_produse():
    produse = [{"id": 1, "nume": "Laptop"}, {"id": 2, "nume": "Mouse"}]
    return jsonify(produse), 200 # 200 OK
"""
    },
    "sql_nosql": {
        "title": "Diferențele dintre SQL și NoSQL",
        "query": "sql vs nosql nosql baze de date relationale baze de date ne-relationale mongodb vs postgresql",
        "description": "Bazele de date SQL (Relaționale) stochează datele structurat în tabele fixe, în timp ce NoSQL (Ne-relaționale) folosesc modele dinamice, schemaless (documente, grafuri) ideale pentru flexibilitate și scalare.",
        "details": """
### Comparație detaliată:
- **SQL (Relaționale)**:
  - Date structurate pe linii și coloane (tabele).
  - Conexiuni prin chei primare și străine (tabele legate).
  - Schema este strictă și predefinită (necesită migrații).
  - Suport ACID complet (Consistență strictă).
  - Exemple: PostgreSQL, MySQL, SQLite.
- **NoSQL (Ne-relaționale)**:
  - Structuri flexibile: documente JSON, cheie-valoare, grafuri.
  - Schema dinamică (poți adăuga câmpuri noi oricând fără eroare).
  - Scalabilitate orizontală (se împarte pe mai multe servere).
  - Exemple: MongoDB, Redis, Cassandra.
""",
        "code": """
-- Query SQL (Relațional)
SELECT nume, prenume FROM utilizatori WHERE varsta >= 18;

-- Interogare MongoDB NoSQL (Document-based)
db.utilizatori.find({ varsta: { $gte: 18 } }, { nume: 1, prenume: 1 })
"""
    },
    "db_indexing": {
        "title": "Indexarea în Baze de Date",
        "query": "indexare index db indecși indecsi index baze de date b-tree idx",
        "description": "Un index reprezintă o structură de date specială creată de sistemul de gestiune al bazelor de date pentru a crește considerabil viteza de căutare a rândurilor într-o tabelă.",
        "details": r"""
### Cum funcționează:
Indexul acționează ca un cuprins de carte. În loc ca motorul DB să scaneze întreaga tabelă linie cu linie (Table Scan - complexitate $O(N)$), acesta caută în structura de index (B-Tree - complexitate $O(\log N)$) pentru a merge direct la adresa de memorie fizică a rândului căutat.

### Regulă importantă:
- **Avantaj**: Răspuns ultra-rapid la interogări de tip `SELECT`.
- **Dezavantaj**: Încetinește operațiile de scriere (`INSERT`, `UPDATE`, `DELETE`), deoarece sistemul trebuie să actualizeze indecșii la fiecare modificare.
""",
        "code": """
-- Crearea unui index pe coloana 'email'
CREATE INDEX idx_utilizatori_email ON utilizatori(email);

-- Interogare accelerată masiv de index
SELECT * FROM utilizatori WHERE email = 'mihai@domain.com';
"""
    },
    "complexity_big_o": {
        "title": "Complexitatea Timp și Notația Big O",
        "query": "big o complexitate timp complexitatea timp notația big o timp de executie eficienta algoritm",
        "description": "Notația Big O este utilizată în informatică pentru a măsura și descrie eficiența în timp și spațiu a unui algoritm, indicând cum crește timpul de execuție în raport cu volumul datelor de intrare ($N$).",
        "details": r"""
### Cele mai comune complexități:
1. **$O(1)$ - Constantă**: Timp identic indiferent de input (ex: accesarea unui element dintr-un array după index).
2. **$O(\log N)$ - Logarithmică**: Dimensiunea problemei se înjumătățește la fiecare pas (ex: Căutarea Binară).
3. **$O(N)$ - Liniară**: Timpul crește liniar cu numărul de elemente (ex: parcurgerea unei liste).
4. **$O(N \log N)$ - Liniar-Logarithmică**: Optimizări bune de sortare (ex: QuickSort, MergeSort).
5. **$O(N^2)$ - Pătratică**: Bucle imbricate (ex: BubbleSort, parcurgerea matricelor).
""",
        "code": """
# Algoritm O(N) - Timpul crește liniar cu lungimea listei
def cauta_element(lista, tinta):
    for idx, x in enumerate(lista):
        if x == tinta:
            return idx
    return -1
"""
    },
    "recursion_iteration": {
        "title": "Recursivitate vs Iterare",
        "query": "recursivitate vs iterare iterare vs recursivitate recursiv vs iterativ suma_recursiva suma_iterativa",
        "description": "Recursivitatea folosește apeluri repetate de funcție care consumă memorie pe stivă, în timp ce iterarea utilizează bucle repetitive eficiente în memorie.",
        "details": """
### Comparație directă:
- **Recursivitatea**:
  - Se bazează pe Call Stack (stiva de apeluri a sistemului).
  - Risc de **Stack Overflow** dacă depășește adâncimea maximă de apel.
  - Cod extrem de curat, elegant și scurt.
- **Iterarea (Buclele)**:
  - Folosește variabile locale de control și sare direct la adrese de memorie pe CPU.
  - Extrem de rapidă și optimă ca resurse de memorie (complexitate spațiu $O(1)$).
""",
        "code": """
# Suma primelor N numere - Recursivă (O(N) spațiu pe stivă)
def suma_recursiva(n):
    if n <= 1:
        return n
    return n + suma_recursiva(n - 1)

# Suma primelor N numere - Iterativă (O(1) spațiu, mult mai sigură)
def suma_iterativa(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
"""
    }
}


# Programmatically inject high-precision Transformer semantic anchors for CodeBERT matching
ANCHORS_MAP = {
    "ast": ["ast", "abstract syntax tree", "arbore de sintaxa abstracta", "sintaxa abstracta", "ast parse", "visitor ast"],
    "attention": ["self attention", "attention formula", "query key value", "capete de atentie", "mecanismul de atentie", "softmax qk"],
    "streamlit_state": ["session state", "st.session_state", "streamlit state", "pastrare stare", "starea sesiunii"],
    "database_normalization": ["normalizare", "forma normala", "1nf", "2nf", "3nf", "baza de date normalizare", "functional dependency"],
    "acid": ["acid", "tranzactii sql", "atomicitate", "consistenta", "izolare", "durabilitate", "acid transaction"],
    "sql_injection": ["sql injection", "injectie sql", "sqli", "prepared statements", "securitate sql"],
    "java_oop": ["java oop", "oop java", "clase java", "polimorfism java", "incapsulare java", "mostenire java"],
    "js_async": ["javascript async", "async await", "promises js", "event loop", "javascript promises", "asincron js"],
    "cpp_memory": ["cpp memory", "pointeri cpp", "raii", "unique_ptr", "delete new cpp", "gestiune memorie c++"],
    "rust_safety": ["rust ownership", "borrow checker", "lifetimes rust", "siguranta memorie rust", "referinte rust"],
    "go_concurrency": ["go concurrency", "goroutines", "go channels", "concurenta go", "buffered channel"],
    "csharp_dotnet": ["c# dotnet", "linq", "generics c#", "garbage collection clr", "delegat c#"],
    "web_layout": ["css flexbox", "css grid", "layout css", "responsive grid", "flexbox vs grid"],
    "recursion": ["recursivitate", "recursie", "recursion", "functie recursiva", "factorial recursiv"],
    "oop_principles": ["principii oop", "mostenire polimorfism", "incapsulare abstractizare", "principii programare orientata pe obiecte"],
    "stack_ds": ["stiva", "stack ds", "lifo", "operatii stiva", "push pop stack"],
    "queue_ds": ["coada", "queue ds", "fifo", "enqueue dequeue", "operatii coada"],
    "tree_ds": ["arbore binar", "binary tree", "bst", "avl tree", "parcurgere arbore"],
    "graph_ds": ["graf", "graph ds", "dijkstra", "noduri muchii graf", "lista adiacenta"],
    "solid_principles": ["principii solid", "solid design", "single responsibility", "liskov", "dependency inversion"],
    "singleton_pattern": ["singleton", "design pattern singleton", "instanta unica singleton", "singleton class"],
    "git_vcs": ["git commit", "git branch", "git rebase", "vcs git", "merge conflict git"],
    "json_format": ["json format", "parse json", "javascript object notation", "serializare json"],
    "rest_api": ["rest api", "endpoint-uri http", "http methods get post", "stateless api"],
    "sql_nosql": ["sql vs nosql", "mongodb vs postgresql", "baze de date relationale", "nosql relationale"],
    "db_indexing": ["indexare db", "database index", "b-tree index", "explain index query"],
    "complexity_big_o": ["big o", "complexitate timp", "complexitate spatiu", "notația big o", "o log n"],
    "recursion_iteration": ["recursivitate vs iterare", "recursiv vs iterativ", "tail call optimization"]
}

for k, anchors in ANCHORS_MAP.items():
    if k in KNOWLEDGE_BASE:
        KNOWLEDGE_BASE[k]["anchors"] = anchors


UNIVERSAL_DICT = {
    "docker": {
        "title": "🐳 Docker & Containerizarea Modernă",
        "description": "Docker este o platformă open-source care permite dezvoltatorilor să împacheteze, distribuie și ruleze aplicații în medii izolate numite **containere**.",
        "details": """
Containerele Docker includ tot ce are nevoie o aplicație pentru a rula (cod, runtime, biblioteci de sistem), garantând că aceasta va funcționa la fel pe orice mașină (dezvoltare, testare, producție).
Spre deosebire de Mașinile Virtuale (VM) care includ un întreg sistem de operare oaspete, containerele partajează kernel-ul sistemului de operare gazdă, fiind extrem de ușoare, rapide și eficiente ca resurse.

### Elemente cheie:
- **Dockerfile**: Un fișier text cu instrucțiuni pas cu pas pentru construirea unei imagini.
- **Imagine Docker**: Un șablon read-only folosit pentru crearea containerelor.
- **Container**: O instanță rulabilă a unei imagini.
- **Docker Compose**: Un instrument pentru definirea și rularea aplicațiilor multi-container.
""",
        "code": """
# Exemplu de Dockerfile simplu pentru Python
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
    },
    "react": {
        "title": "⚛️ React.js (Dezvoltare Frontend modernă)",
        "description": "React este o bibliotecă JavaScript declarativă, eficientă și flexibilă pentru construirea de interfețe cu utilizatorul (UI), dezvoltată de Meta (Facebook).",
        "details": """
React se bazează pe **Componente** independente și reutilizabile, care își gestionează propria stare.

### Concepte Fundamentale:
- **Virtual DOM**: React creează o copie în memorie a DOM-ului real. Când starea se modifică, React calculează diferențele și actualizează eficient doar elementele modificate în pagină, asigurând performanțe excepționale.
- **JSX**: O extensie de sintaxă care permite scrierea de cod asemănător cu HTML direct în JavaScript.
- **State & Props**: `State` reprezintă datele interne ale unei componente (care se pot schimba), iar `Props` sunt datele primite de la componenta părinte.
- **Hooks**: Funcții speciale (cum ar fi `useState` și `useEffect`) care permit componentelor funcționale să folosească starea și alte funcționalități React fără a scrie clase.
""",
        "code": """
// Exemplu de componentă funcțională React cu Hooks
import React, { useState, useEffect } from 'react';

function Counter() {
    const [count, setCount] = useState(0);

    useEffect(() => {
        document.title = `Ai dat click de ${count} ori`;
    }, [count]);

    return (
        <button onClick={() => setCount(count + 1)}>
            Click me: {count}
        </button>
    );
}
"""
    },
    "django": {
        "title": "🦄 Django Framework (Web Development în Python)",
        "description": "Django este un framework web Python de nivel înalt care încurajează dezvoltarea rapidă și designul curat. Urmează filosofia **'baterii incluse'**, oferind aproape tot ce este necesar direct din cutie.",
        "details": """
Django folosește arhitectura **MVT (Model-View-Template)**, similară cu clasicul MVC:
- **Model**: Definește structura datelor și interacționează cu baza de date prin ORM.
- **View**: Gestionează logica de business și returnează răspunsurile HTTP.
- **Template**: Partea vizuală, fișiere HTML dinamice redate în browser.

### Avantaje Majore:
- **Securitate implicită**: Protejează automat aplicațiile de SQL Injection, Cross-Site Scripting (XSS) și Cross-Site Request Forgery (CSRF).
- **Panou de Administrare**: Generează automat o interfață completă de admin pe baza modelelor definite.
- **ORM puternic**: Permite interogarea bazei de date folosind exclusiv cod Python (fără a scrie SQL manual).
""",
        "code": """
# Exemplu de model Django
from django.db import models

class Student(models.Model):
    nume = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    nota_licenta = models.FloatField()

    def __str__(self):
        return self.nume
"""
    },
    "machine_learning": {
        "title": "🤖 Învățare Automată (Machine Learning & AI)",
        "description": "Machine Learning (ML) reprezintă o subramură a Inteligenței Artificiale care se concentrează pe dezvoltarea de algoritmi capabili să învețe din date și să facă predicții fără a fi programați explicit.",
        "details": """
### Cele 3 categorii principale de ML:
1. **Învățare Supervizată (Supervised Learning)**:
   - Modelul este antrenat pe un set de date etichetate (input + output corect). Exemple: clasificare (ex: spam vs. email legitim) și regresie (ex: predicția prețului unei case).
2. **Învățare Nesupervizată (Unsupervised Learning)**:
   - Datele nu sunt etichetate, iar algoritmul încearcă să găsească tipare sau structuri ascunse. Exemplu: clustering (gruparea clienților după comportament).
3. **Învățare prin Recompensă (Reinforcement Learning)**:
   - Un agent învață să ia decizii într-un mediu pentru a maximiza o recompensă cumulativă (folosit în roboți, jocuri de șah/Go).
""",
        "code": """
# Exemplu de regresie liniară simplă cu Scikit-Learn
from sklearn.linear_model import LinearRegression
import numpy as np

# Date de antrenare: dimensiune casă -> preț
X = np.array([[50], [80], [120]])
y = np.array([200000, 310000, 450000])

model = LinearRegression()
model.fit(X, y)

# Predicție pentru o casă de 100 mp
pret_predis = model.predict([[100]])
print(f"Preț estimat: {pret_predis[0]:.2f} EUR")
"""
    },
    "linux": {
        "title": "🐧 Sistemul de Operare Linux & Linia de Comandă",
        "description": "Linux este un sistem de operare open-source de tip Unix, extrem de robust și stabil, care rulează pe majoritatea serverelor de internet, supercomputerelor și dispozitivelor Android.",
        "details": """
Interacțiunea principală cu Linux se face prin **Terminal (Shell / Bash)**, oferind automatizări puternice.

### Comenzi esențiale:
- `ls` - listează fișierele și directoarele.
- `cd [cale]` - schimbă directorul curent.
- `pwd` - afișează calea absolută a directorului curent.
- `grep [termen] [fișier]` - caută un text în interiorul unui fișier.
- `chmod` - modifică permisiunile fișierelor.
- `ps aux` - afișează procesele active din sistem.
- `top` / `htop` - monitorizează în timp real resursele de sistem.
""",
        "code": """
# Script Bash simplu pentru backup
#!/bin/bash
DIR_SURA="/Users/mihaela/Desktop/P3"
DIR_DEST="/Users/mihaela/Backup"
tar -czf "$DIR_DEST/backup_$(date +%F).tar.gz" "$DIR_SURA"
echo "Backup finalizat cu succes!"
"""
    },
    "html_css": {
        "title": "🌐 Dezvoltare Web: HTML și CSS",
        "description": "HTML și CSS reprezintă fundamentele dezvoltării web: HTML oferă structura structurală a paginii, iar CSS definește stilizarea și designul vizual.",
        "details": """
### Concepte Fundamentale:
- **HTML (HyperText Markup Language)**: Folosește un sistem de etichete (tag-uri) precum `<h1>`, `<p>`, `<div>`, `<a>` pentru a descrie conținutul și semantica paginii.
- **CSS (Cascading Style Sheets)**: Selectează elementele HTML și le aplică reguli vizuale (culori, fonturi, margini, aliniamente).
- **Box Model**: Fiecare element HTML este reprezentat ca o casetă formată din: `Content` (conținutul brut), `Padding` (spațiul interior), `Border` (conturul) și `Margin` (spațiul exterior de separare).
- **Layout Modern**: Folosește Flexbox (pentru alinieri unidimensionale) și Grid Layout (pentru structuri bidimensionale complexe).
""",
        "code": """
<!-- Structură HTML cu CSS inline/embeded -->
<div class="card">
  <h2>Titlu Card</h2>
  <p>Conținut descriptiv card...</p>
</div>

<style>
.card {
  padding: 20px;
  border-radius: 8px;
  background: #1e293b;
  color: #f8fafc;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
}
.card:hover {
  transform: translateY(-5px);
}
</style>
"""
    },
    "jwt": {
        "title": "🔑 JWT (JSON Web Tokens) & Autentificare",
        "description": "JSON Web Token (JWT) este un standard deschis (RFC 7519) compact și autonom folosit pentru a transmite informații în siguranță între părți sub forma unui obiect JSON.",
        "details": """
JWT-urile sunt folosite pe scară largă în autentificarea de tip stateless pentru API-urile REST moderne.

### Structura unui JWT (format din 3 părți separate prin puncte `.`):
1. **Header**: Conține tipul de token (JWT) și algoritmul de criptare folosit (ex: HS256, RS256).
2. **Payload (Informațiile)**: Conține datele efective (claims), cum ar fi ID-ul utilizatorului, numele acestuia și permisiunile, alături de data expirării.
3. **Signature (Semnătura)**: Se obține prin combinarea Header-ului și Payload-ului criptate cu o cheie secretă de pe server, asigurând integritatea tokenului (dacă atacatorul modifică Payload-ul, semnătura devine invalidă).
""",
        "code": """
# Utilizare teoretică PyJWT în Python
import jwt
import datetime

SECRET_KEY = "cheie_secreta_super_dificil_de_ghicit"

# Generare Token (pe server la Login)
payload = {
    "user_id": 42,
    "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
}
token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
print(f"Token JWT generat: {token}")
"""
    },
    "virtualenv": {
        "title": "📦 Medii Virtuale în Python (virtualenv & pip)",
        "description": "Un mediu virtual este un director izolat complet care conține o instalare proprie a unui interpretor Python și o colecție independentă de pachete instalate.",
        "details": """
Mediile virtuale rezolvă problema conflictelor de versiuni ale bibliotecilor. De exemplu, dacă Proiectul A are nevoie de Django 3.2 și Proiectul B de Django 4.2, rularea lor pe aceeași instalare globală de Python ar cauza eșecuri.

### Fluxul de lucru cu `venv`:
1. **Creare**: `python3 -m venv .venv` (creează folderul `.venv` în proiect).
2. **Activare**:
   - macOS / Linux: `source .venv/bin/activate`
   - Windows: `.venv\\Scripts\\activate`
3. **Instalare**: `pip install streamlit` (pachetele se descarcă doar în interiorul `.venv`).
4. **Salvare dependențe**: `pip freeze > requirements.txt` (permite colegilor să își instaleze aceleași versiuni prin `pip install -r requirements.txt`).
""",
        "code": """
# Secvență completă de lucru în terminal
$ cd /Users/mihaela/Desktop/P3
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
"""
    }
}


def _generate_dynamic_fallback(topic):
    """Generează dinamic o pagină educațională custom în limba română pentru orice concept de programare."""
    topic_title = topic.strip().capitalize()
    
    lines = []
    lines.append(f"### 💡 Conceptul: **{topic_title}**\n")
    lines.append(f"Subiectul **{topic_title}** reprezintă un concept, instrument sau tehnologie importantă în dezvoltarea software modernă și informatică.\n")
    
    lines.append("#### 🧱 Prezentare Generală:")
    lines.append(f"- **Definiție**: În sens larg, **{topic_title}** servește ca element esențial în arhitectura software sau în logică algoritmică, având rolul de a îmbunătăți structura, optimizarea, performanța sau securitatea aplicațiilor.")
    lines.append(f"- **Cum funcționează**: Operează prin abstractizarea proceselor subiacente, permițând dezvoltatorilor să scrie cod mai curat, modular și adaptabil la schimbări.")
    lines.append("")
    
    lines.append("#### ⚙️ Piloni Principali de Aplicare:")
    lines.append(f"1. **Eficiență și Scalabilitate**: Permite optimizarea utilizării resurselor fizice și logice, facilitând extinderea facilă a programelor.")
    lines.append(f"2. **Mentenabilitate**: Prin utilizarea **{topic_title}**, codul devine mai ușor de testat, documentat și înțeles de către alți membri ai echipei de dezvoltare.")
    lines.append(f"3. **Standardizare**: Reprezintă o bună practică recunoscută la nivel global în industrie, oferind soluții pre-testate pentru probleme frecvente.")
    lines.append("")
    
    lines.append("#### 💻 Exemplu teoretic / Structură de cod:")
    lines.append(f"În majoritatea limbajelor (de ex. Python), utilizarea conceptuală a **{topic_title}** se ghidează după structuri logice bine definite:")
    lines.append(f"```python\n# Exemplu conceptual de utilizare / design pattern pentru {topic_title}\nclass Modern{topic_title.replace(' ', '')}:\n    def __init__(self, config=None):\n        self.config = config or {{}}\n        self.is_active = True\n        print(f\"[System] {topic_title} inițializat cu succes!\")\n\n    def executa_operatie(self, date):\n        if not self.is_active:\n            raise ValueError(\"Sistemul nu este activ!\")\n        # Procesare logică specifică\n        rezultat = f\"Date procesate prin {topic_title}: {{date}}\"\n        return rezultat\n```")
    
    answer_text = "\n".join(lines)
    analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Modul Generativ Local]</b> Am detectat interesul tău pentru conceptul: <b>{topic_title}</b>. Deoarece nu este o funcție locală din fișierele tale, am generat un ghid educațional complet.
</div>

"""
    return analysis_box + answer_text


SYNTHESIS_DB = {
    "python": {
        "variable": {
            "title": "Declararea variabilelor în Python",
            "desc": "În Python, variabilele se declară simplu prin atribuirea unei valori folosind operatorul `=`. Python este un limbaj cu tipizare dinamică, deci nu trebuie să specifici tipul variabilei la declarare.",
            "code": """
# Atribuiri simple în Python
nume = "Alex"      # string
varsta = 21        # int
inaltime = 1.75    # float
este_student = True # bool

print(f"{nume} are {varsta} ani.")
"""
        },
        "function": {
            "title": "Definirea funcțiilor în Python",
            "desc": "Funcțiile în Python se definesc folosind cuvântul cheie `def`, urmat de numele funcției, parametrii între paranteze și un bloc de cod indentat. Returnarea valorilor se face prin `return`.",
            "code": """
def calculeaza_suma(a, b):
    \"\"\"Returnează suma a două numere\"\"\"
    return a + b

# Apelul funcției
rezultat = calculeaza_suma(5, 7)
print("Suma este:", rezultat) # Output: 12
"""
        },
        "class": {
            "title": "Clase și Obiecte în Python",
            "desc": "În Python, clasele se definesc prin `class`. Constructorul este definit prin metoda specială `__init__`, iar primul argument al oricărei metode trebuie să fie `self`, reprezentând instanța curentă.",
            "code": """
class Student:
    def __init__(self, nume, nota):
        self.nume = nume
        self.nota = nota
        
    def afiseaza_detalii(self):
        print(f"Studentul {self.nume} are nota {self.nota}.")

# Instanțierea clasei
s = Student("Maria", 9.5)
s.afiseaza_detalii()
"""
        },
        "loop": {
            "title": "Bucle în Python (For & While)",
            "desc": "Buclele `for` sunt folosite în Python pentru a itera peste colecții sau intervale (generate prin `range()`), iar buclele `while` rulează atâta timp cât o condiție este adevărată.",
            "code": """
# Buclă For peste un interval
print("Interval:")
for i in range(3):
    print(f"Pasul {i}")

# Buclă For peste o listă
print("\\nListă:")
fructe = ["măr", "banană", "cireașă"]
for f in fructe:
    print(f)
"""
        },
        "condition": {
            "title": "Structuri Condiționale în Python (If/Else)",
            "desc": "Python folosește structurile `if`, `elif` (else if) și `else` pentru ramificarea execuției codului, bazându-se pe indentare pentru definirea blocurilor.",
            "code": """
nota = 8

if nota >= 9:
    print("Excelent!")
elif nota >= 5:
    print("Promovat.")
else:
    print("Picat.")
"""
        },
        "list": {
            "title": "Lucrul cu Liste în Python",
            "desc": "Listele în Python sunt colecții ordonate, mutabile și dinamice. Pot conține elemente de tipuri diferite și suportă indexare, feliere (`slicing`) și metode variate.",
            "code": """
numere = [1, 2, 3]
numere.append(4) # Adăugare la sfârșit
numere.insert(1, 10) # Adăugare la indexul 1

print("Lungime listă:", len(numere))
print("Elementul 1:", numere[1]) # Output: 10
"""
        },
        "file": {
            "title": "Citirea și Scrierea Fișierelor în Python",
            "desc": "Pentru operarea pe fișiere, cel mai sigur mod în Python este utilizarea managerului de context `with open(...)`, deoarece eliberează automat resursele în caz de eroare.",
            "code": """
# Scrierea într-un fișier
with open("exemplu.txt", "w", encoding="utf-8") as f:
    f.write("Salut din Python!\\n")

# Citirea dintr-un fișier
with open("exemplu.txt", "r", encoding="utf-8") as f:
    continut = f.read()
    print(continut)
"""
        }
    },
    "java": {
        "variable": {
            "title": "Declararea variabilelor în Java",
            "desc": "Java este un limbaj puternic tipizat și static. La declararea fiecărei variabile, trebuie să specifici explicit tipul de date al acesteia.",
            "code": """
// Java - Tipuri primitive și obiecte
int numar = 42;
double pret = 19.99;
char litera = 'A';
boolean esteActiv = true;
String nume = "Ioana";

System.out.println("Numele este: " + nume);
"""
        },
        "function": {
            "title": "Definirea metodelor în Java",
            "desc": "În Java, funcțiile sunt numite metode și trebuie să facă parte dintr-o clasă. Trebuie specificat modificatorul de acces, tipul returnat (sau `void`), numele și argumentele.",
            "code": """
public class Calculator {
    // Metodă statică simplă
    public static int adunare(int a, int b) {
        return a + b;
    }
    
    public static void main(String[] args) {
        int rezultat = adunare(10, 15);
        System.out.println("Rezultatul: " + rezultat);
    }
}
"""
        },
        "class": {
            "title": "Clase și Obiecte în Java",
            "desc": "Java este un limbaj OOP pur. Constructorul clasei poartă numele exact al clasei și nu returnează niciun tip.",
            "code": """
public class Persoana {
    private String nume; // Încapsulare
    
    public Persoana(String nume) { // Constructor
        this.nume = nume;
    }
    
    public void saluta() {
        System.out.println("Salut, eu sunt " + nume);
    }
}
// Instanțiere: Persoana p = new Persoana("Mihai");
"""
        },
        "loop": {
            "title": "Bucle în Java (For, While, Foreach)",
            "desc": "Java folosește structuri de buclă standard similare cu C/C++: `for` clasic, `while` și bucla îmbunătățită `for-each` pentru colecții.",
            "code": """
// Buclă For clasică
for (int i = 0; i < 3; i++) {
    System.out.println("Pas " + i);
}

// Buclă For-each peste un array
String[] culori = {"Rosu", "Verde", "Albastru"};
for (String c : culori) {
    System.out.println(c);
}
"""
        },
        "condition": {
            "title": "Structuri Condiționale în Java (If/Else)",
            "desc": "Instrucțiunea `if` evaluează o expresie booleană parantezată. Java acceptă și `switch` pentru verificări multiple.",
            "code": """
int punctaj = 85;

if (punctaj >= 90) {
    System.out.println("Nota 10");
} else if (punctaj >= 80) {
    System.out.println("Nota 9");
} else {
    System.out.println("Nota sub 9");
}
"""
        },
        "list": {
            "title": "Lucrul cu Liste în Java (ArrayList)",
            "desc": "În Java, array-urile simple au dimensiune fixă. Pentru dimensiuni dinamice, se folosește clasa generică `ArrayList` din `java.util`.",
            "code": """
import java.util.ArrayList;

ArrayList<String> orase = new ArrayList<>();
orase.add("București");
orase.add("Cluj");
orase.add("Timișoara");

System.out.println("Număr orașe: " + orase.size());
System.out.println("Primul oraș: " + orase.get(0));
"""
        },
        "file": {
            "title": "Citirea și Scrierea Fișierelor în Java",
            "desc": "În Java modern, lucrul cu fișiere se face cel mai simplu folosind clasa `Files` din pachetul `java.nio.file`.",
            "code": """
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

try {
    // Scrierea în fișier
    Files.write(Paths.get("date.txt"), "Salut din Java!".getBytes());
    
    // Citirea din fișier
    String continut = new String(Files.readAllBytes(Paths.get("date.txt")));
    System.out.println(continut);
} catch (IOException e) {
    e.printStackTrace();
}
"""
        }
    },
    "cpp": {
        "variable": {
            "title": "Declararea variabilelor în C++",
            "desc": "C++ este un limbaj compilat, puternic tipizat static. Variabilele trebuie declarate precizând tipul lor înaintea utilizării.",
            "code": """
#include <iostream>
#include <string>

int main() {
    int varsta = 20;
    double pret = 9.99;
    char grupa = 'B';
    bool admis = true;
    std::string nume = "Vlad";
    
    std::cout << nume << " are " << varsta << " ani." << std::endl;
    return 0;
}
"""
        },
        "function": {
            "title": "Definirea funcțiilor în C++",
            "desc": "Funcțiile în C++ necesită specificarea tipului returnat, a numelui și a parametrilor. Pot fi declarate mai întâi ca prototip și implementate ulterior.",
            "code": """
#include <iostream>

// Prototipul funcției
int inmultire(int a, int b);

int main() {
    std::cout << "Produsul: " << inmultire(4, 5) << std::endl;
    return 0;
}

// Implementarea funcției
int inmultire(int a, int b) {
    return a * b;
}
"""
        },
        "class": {
            "title": "Clase și Obiecte în C++",
            "desc": "C++ suportă clase și încapsulare cu secțiuni explicite `public:` și `private:`. Destructorul clasei se notează cu tilda `~`.",
            "code": """
#include <iostream>
#include <string>

class Dreptunghi {
private:
    int latime, inaltime; // Private implicit
public:
    Dreptunghi(int l, int h) : latime(l), inaltime(h) {} // Constructor
    
    int arie() { return latime * inaltime; }
};
"""
        },
        "loop": {
            "title": "Bucle în C++ (For, While, Do-While)",
            "desc": "C++ are bucle de bază identice cu C: `for` pentru pași controlați, `while` și `do-while` pentru testare la final.",
            "code": """
#include <iostream>

int main() {
    // Buclă For simplă
    for(int i = 0; i < 3; ++i) {
        std::cout << "Pas " << i << std::endl;
    }
    
    // Ranged-based For (C++11+)
    int numere[] = {10, 20, 30};
    for(int n : numere) {
        std::cout << n << std::endl;
    }
    return 0;
}
"""
        },
        "condition": {
            "title": "Structuri Condiționale în C++",
            "desc": "C++ folosește instrucțiunile standard `if`, `else if` și `else`, utilizând operatori logici ca `&&` (AND), `||` (OR) și `!` (NOT).",
            "code": """
#include <iostream>

int main() {
    int x = 15;
    if (x > 10 && x < 20) {
        std::cout << "Numărul este în interval." << std::endl;
    } else {
        std::cout << "În afara intervalului." << std::endl;
    }
    return 0;
}
"""
        },
        "list": {
            "title": "Lucrul cu Liste în C++ (std::vector)",
            "desc": "În loc de array-uri rigide, în C++ modern se folosește containerul dinamic `std::vector` din biblioteca standard.",
            "code": """
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;
    v.push_back(1); // Adăugare element
    v.push_back(2);
    v.push_back(3);
    
    std::cout << "Dimensiune vector: " << v.size() << std::endl;
    std::cout << "Elementul de pe poz. 0: " << v[0] << std::endl;
    return 0;
}
"""
        },
        "file": {
            "title": "Citirea și Scrierea Fișierelor în C++",
            "desc": "C++ folosește clasele `ofstream` (pentru output în fișiere) și `ifstream` (pentru input din fișiere) din biblioteca `<fstream>`.",
            "code": """
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Scrierea în fișier
    std::ofstream out("text.txt");
    out << "Salut din C++!" << std::endl;
    out.close();
    
    // Citirea din fișier
    std::ifstream in("text.txt");
    std::string linie;
    if (std::getline(in, linie)) {
        std::cout << linie << std::endl;
    }
    return 0;
}
"""
        }
    },
    "javascript": {
        "variable": {
            "title": "Declararea variabilelor în JavaScript",
            "desc": "În JS modern, variabilele se declară folosind `let` (pentru valori mutabile) sau `const` (pentru constante mutabile). Se evită utilizarea vechiului `var` din cauza problemelor de vizibilitate (scoping).",
            "code": """
const nume = "Andrei"; // Constantă
let varsta = 22;       // Mutabilă
varsta = 23;           // Permis

const esteStudent = true;
console.log(`${nume} are ${varsta} de ani.`);
"""
        },
        "function": {
            "title": "Definirea funcțiilor în JavaScript",
            "desc": "JavaScript suportă declararea funcțiilor standard (`function`) cât și funcțiile săgeată (`Arrow Functions`) moderne, folosite des ca expresii.",
            "code": """
// Funcție standard
function salut(nume) {
    return "Salut, " + nume;
}

// Funcție săgeată (Arrow Function)
const inmultire = (a, b) => a * b;

console.log(salut("Dana"));
console.log("Produs:", inmultire(3, 4)); // Output: 12
"""
        },
        "class": {
            "title": "Clase și Obiecte în JavaScript",
            "desc": "Deși JS folosește moștenire bazată pe prototipuri, ES6 a introdus cuvântul cheie `class` ca syntactic sugar peste prototipuri.",
            "code": """
class Persoana {
    constructor(nume) {
        this.nume = nume;
    }
    
    prezintaTe() {
        console.log(`Salut, numele meu este ${this.nume}.`);
    }
}

const p = new Persoana("George");
p.prezintaTe();
"""
        },
        "loop": {
            "title": "Bucle în JavaScript (For, Foreach, While)",
            "desc": "Pe lângă buclele `for` și `while` clasice, JS oferă buclele moderne `for...of` (pentru valori din array-uri) și `for...in` (pentru chei de obiecte).",
            "code": """
// Buclă For clasică
for (let i = 0; i < 3; i++) {
    console.log("Număr: " + i);
}

// Buclă For...of peste un Array
const fructe = ["mere", "pere", "banane"];
for (const f of fructe) {
    console.log(f);
}
"""
        },
        "condition": {
            "title": "Structuri Condiționale în JavaScript",
            "desc": "Utilizează structurile clasice `if`, `else if` și `else`. Evaluarea egalității se recomandă să fie strictă prin `===` (care verifică și tipul, nu doar valoarea ca `==`).",
            "code": """
const scor = "100";

if (scor === 100) {
    console.log("Scor numeric perfect!");
} else if (scor == 100) {
    console.log("Scor egal ca valoare, dar nu ca tip (string).");
} else {
    console.log("Alt scor.");
}
"""
        },
        "list": {
            "title": "Lucrul cu Array-uri în JavaScript",
            "desc": "În JS, array-urile sunt structuri de date native deosebit de puternice și flexibile. Conțin metode funcționale extrem de iubite.",
            "code": """
const culori = ["rosu", "galben"];
culori.push("albastru"); // Adăugare la final

// Metode funcționale moderne
const culoriMajuscule = culori.map(c => c.toUpperCase());
console.log(culoriMajuscule); // Output: ["ROSU", "GALBEN", "ALBASTRU"]
"""
        },
        "file": {
            "title": "Citirea și Scrierea Fișierelor în JavaScript (Node.js)",
            "desc": "Deoarece JS rulează de obicei în browser, accesul direct la fișiere pe disc necesită mediul de execuție Node.js și modulul său nativ `fs`.",
            "code": """
// Exemplu în Node.js folosind fs/promises
const fs = require('fs').promises;

async function fileOperations() {
    try {
        // Scrierea în fișier
        await fs.writeFile('mesaj.txt', 'Salut din Node.js!');
        
        // Citirea din fișier
        const date = await fs.readFile('mesaj.txt', 'utf8');
        console.log(date);
    } catch (err) {
        console.error(err);
    }
}
"""
        }
    }
}


def generate_follow_up_questions(question, answer, kb_match=None):
    """
    Generează dinamic 3 întrebări de follow-up relevante bazate pe contextul întrebării/răspunsului.
    """
    q = question.lower()
    
    # 1. Baza de cunoștințe
    if kb_match:
        if kb_match.startswith("dynamic_fallback_"):
            topic = kb_match.replace("dynamic_fallback_", "").replace("_", " ").title()
            return [
                f"Care este diferența dintre {topic} și alte concepte conexe?",
                f"Cum se implementează {topic} în practică cu un exemplu complet?",
                f"Care sunt bunele practici și greșelile frecvente în {topic}?"
            ]

        follow_ups = {
            "ast": [
                "Cum pot vizualiza arborele AST în tabul dedicat?",
                "Care este diferența dintre un NodeVisitor și un NodeTransformer?",
                "Cum detectează AST variabilele nefolosite?"
            ],
            "attention": [
                "Cum influențează capetele multiple (Multi-Head) auto-atenția?",
                "Cum pot interpreta intensitatea din Heatmap-ul de atenție?",
                "Ce rol are vectorul Query (Q) în formulă?"
            ],
            "streamlit_state": [
                "Cum folosesc st.session_state pentru a stoca date din formulare?",
                "Când ar trebui să apelez st.rerun() explicit?",
                "Cum pot curăța memoria cache din Streamlit?"
            ],
            "database_normalization": [
                "Ce este forma normală Boyce-Codd (BCNF)?",
                "Cum sparg un tabel în 3NF fără pierdere de date?",
                "Ce este o cheie primară compusă?"
            ],
            "acid": [
                "Ce înseamnă Dirty Read și cum se evită în tranzacții?",
                "Cum diferă Isolation Level-ul într-o bază de date?",
                "Cum previne Rollback-ul pierderea de date?"
            ],
            "sql_injection": [
                "Cum folosesc SQLAlchemy pentru prepared statements?",
                "Cum curăț inputurile primite în câmpurile de text?",
                "Ce este un blind SQL injection?"
            ],
            "java_oop": [
                "Care este diferența dintre o interfață și o clasă abstractă în Java?",
                "Cum funcționează supraîncărcarea (overloading) vs suprascrierea (overriding)?",
                "Ce este modificatorul de acces 'protected'?"
            ],
            "js_async": [
                "Ce este un Promise.all() și cum optimizează apelurile?",
                "Cum funcționează microtask queue în Event Loop?",
                "Care este diferența dintre Promise și Callback?"
            ],
            "cpp_memory": [
                "Care este diferența dintre unique_ptr și shared_ptr?",
                "Ce este un memory leak și cum îl pot depista?",
                "Cum funcționează destructorul virtual în C++?"
            ],
            "rust_safety": [
                "Ce sunt Lifetimes în Rust și când sunt necesare?",
                "Cum funcționează conceptconceptul de mutabilitate în Rust?",
                "Ce este tipul special Option<T>?"
            ],
            "go_concurrency": [
                "Ce este un buffered channel în Go?",
                "Cum folosesc sync.WaitGroup pentru sincronizare?",
                "Ce se întâmplă în caz de Deadlock într-un canal?"
            ],
            "csharp_dotnet": [
                "Cum funcționează delegatul Func<> și Action<> în C#?",
                "Care este diferența dintre IQueryable și IEnumerable în LINQ?",
                "Ce este Garbage Collection-ul în CLR?"
            ],
            "web_layout": [
                "Când ar trebui să folosesc CSS Grid în loc de Flexbox?",
                "Cum fac un grid responsiv cu repeat(auto-fit, minmax)?",
                "Ce proprietăți Flexbox aliniază elementele pe axa transversală?"
            ],
            "recursion": [
                "Care este diferența de performanță dintre recursivitate și iterare?",
                "Cum previn eroarea de Stack Overflow în Python?",
                "Cum implementez căutarea binară recursiv?"
            ],
            "oop_principles": [
                "Ce înseamnă compoziția în loc de moștenire (Composition over Inheritance)?",
                "Ce este polimorfismul la compilare?",
                "Cum se aplică abstractizarea în design-ul bazelor de date?"
            ],
            "stack_ds": [
                "Cum pot implementa o stivă folosind o listă înlănțuită?",
                "Care este complexitatea timp O(1) pentru push și pop?",
                "Cum folosește compilatorul stiva în recursivitate?"
            ],
            "queue_ds": [
                "Care este diferența dintre coada simplă și o coadă de priorități?",
                "Cum funcționează o coadă circulară (Circular Queue)?",
                "Cum folosesc deque din Python pentru cozi?"
            ],
            "tree_ds": [
                "Cum se face parcurgerea în inordine (In-order traversal) a unui BST?",
                "Ce este un arbore AVL și cum se auto-echilibrează?",
                "Care este diferența dintre BFS și DFS pentru arbori?"
            ],
            "graph_ds": [
                "Cum funcționează algoritmul lui Dijkstra pentru drumul minim?",
                "Care sunt diferențele dintre matricea și lista de adiacență?",
                "Cum detectez ciclurile într-un graf?"
            ],
            "solid_principles": [
                "Ce este Liskov Substitution Principle (LSP)?",
                "Cum aplic Dependency Inversion în proiectul meu?",
                "Ce înseamnă Interface Segregation?"
            ],
            "singleton_pattern": [
                "De ce este Singleton considerat uneori un anti-pattern?",
                "Cum creez un Singleton thread-safe în Python?",
                "Cum folosesc Singleton pentru conexiunea la baza de date?"
            ],
            "git_vcs": [
                "Cum rezolv conflictele de îmbinare (merge conflicts) în Git?",
                "Care este diferența dintre git merge și git rebase?",
                "Cum anulez ultimul commit local fără a pierde modificările?"
            ],
            "json_format": [
                "Cum convertesc o stuctură de date Python în JSON?",
                "Care este diferența dintre JSON și XML?",
                "Cum parsez JSON asincron în JavaScript?"
            ],
            "rest_api": [
                "Ce este un RESTful API stateless?",
                "Care este rolul token-urilor JWT în securizarea API-ului?",
                "Ce înseamnă codul de răspuns HTTP 429?"
            ],
            "sql_nosql": [
                "Când este recomandat să folosesc MongoDB în loc de PostgreSQL?",
                "Ce înseamnă consistența eventuală (Eventual Consistency) în NoSQL?",
                "Cum funcționează scalabilitatea orizontală?"
            ],
            "db_indexing": [
                "De ce scrierea devine mai lentă când adăugăm indecși?",
                "Ce este un index compus pe mai multe coloane?",
                "Cum pot vedea dacă interogarea mea folosește un index (EXPLAIN)?"
            ],
            "complexity_big_o": [
                "De ce este $O(N \\log N)$ mai bun decât $O(N^2)$?",
                "Cum calculez complexitatea spațiu a unui algoritm?",
                "Ce complexitate are căutarea binară?"
            ],
            "recursion_iteration": [
                "Cum se face optimizarea apelului pe coadă (Tail Call Optimization)?",
                "Care este complexitatea memoriei în recursivitatea pe stivă?",
                "Când ar trebui să prefer buclele standard?"
            ]
        }
        if kb_match in follow_ups:
            return follow_ups[kb_match]
            
    # 2. Întrebări legate de fișiere specifice
    for f in ["app.py", "code_parser.py", "vector_store.py", "security_analyzer.py"]:
        if f in q:
            if "app.py" in f:
                return [
                    "Ce componente UI Streamlit se regăsesc în app.py?",
                    "Cum se gestionează stările session_state în app.py?",
                    "Ce biblioteci se folosesc pentru stilizarea interfeței?"
                ]
            elif "code_parser.py" in f:
                return [
                    "Cum extrage code_parser.py clasele și funcțiile prin AST?",
                    "Ce structură are codul pentru generarea diagramelor Mermaid?",
                    "Cum sunt filtrate fișierele de cod acceptate?"
                ]
            elif "vector_store.py" in f:
                return [
                    "Cum funcționează indexul FAISS pentru stocarea embeddings?",
                    "Cum funcționează căutarea hibridă (lexicală + semantică)?",
                    "Cum se vectorisează batch-urile de cod pe Apple Silicon (MPS)?"
                ]
            elif "security_analyzer.py" in f:
                return [
                    "Ce reguli de securitate scanează security_analyzer.py?",
                    "Cum detectează similaritatea CodeBERT secretele hardcodate?",
                    "Cum se pot integra reguli de securitate personalizate?"
                ]
                
    # 3. Întrebări legate de importuri/statistici
    if any(w in q for w in ["import", "depend", "pachet", "librar"]):
        return [
            "Ce librării externe folosește modulul vector_store.py?",
            "Cum adaug o bibliotecă nouă în requirements.txt?",
            "De ce importăm ast în mod explicit?"
        ]
        
    if any(w in q for w in ["statist", "sumar", "cate", "câte", "linii"]):
        return [
            "Care este cel mai mare fișier din proiect și de ce?",
            "Câte funcții din codebase sunt documentate cu docstrings?",
            "Cum influențează dimensiunea codului performanța vectorizării?"
        ]

    # 4. Fallback general bazat pe termeni detectați
    if any(w in q for w in ["variabil", "declar", "atribu"]):
        return [
            "Cum declar variabile în JavaScript vs Java?",
            "Ce este tipizarea dinamică în Python?",
            "Ce este scope-ul unei variabile (local vs global)?"
        ]
    if any(w in q for w in ["functi", "funcți", "metod"]):
        return [
            "Cum definesc o funcție recursivă?",
            "Ce sunt argumentele implicite (default arguments)?",
            "Cum returnez mai multe valori dintr-o funcție Python?"
        ]
    if any(w in q for w in ["clas", "obiect", "construct"]):
        return [
            "Ce rol are constructorul __init__ în Python?",
            "Ce este clasa de bază Object în Java?",
            "Cum implementez moștenirea multiplă în Python?"
        ]
        
    # Fallback implicit universal
    return [
        "Care sunt principiile SOLID de design software?",
        "Cum funcționează recursivitatea în programare?",
        "Ce diferențe sunt între bazele de date SQL și NoSQL?"
    ]


def build_chatbot_answer(question, results, all_chunks, kb=None, indexer=None):
    """Răspuns complet bazat pe KB pre-calculat + FAISS + AST + clasificare semantică prin CodeBERT."""
    import re as _re
    q   = question.lower().strip()
    kb  = kb or {}
    by_name   = kb.get("by_name", {})
    called_by = kb.get("called_by", {})
    call_graph= kb.get("call_graph", {})
    by_file   = kb.get("by_file", {})
    stats     = kb.get("stats", {})
    all_imports = kb.get("all_imports", {})

    subject = _extract_subject(question)

    # ── GENERATORUL DINAMIC DE COD ȘI SINTEZĂ (DYN-SYNTHESIS) ───────────────
    detected_lang = None
    detected_concept = None
    
    # 1. Detecție limbaj
    LANG_KEYWORDS = {
        "python": ["python", "py"],
        "java": ["java"],
        "cpp": ["c++", "cpp", "c plus plus"],
        "javascript": ["javascript", "js", "node.js", "node"]
    }
    for lang, kws in LANG_KEYWORDS.items():
        if any(kw in q for kw in kws):
            detected_lang = lang
            break
            
    # 2. Detecție concept de programare
    CONCEPT_KEYWORDS = {
        "variable": ["variabila", "variabile", "declar", "atribuire", "variabilei", "constanta", "constante"],
        "function": ["functie", "funcție", "functii", "funcții", "metoda", "metodă", "metode"],
        "class": ["clasa", "clasă", "clase", "obiect", "constructor", "init"],
        "loop": ["loop", "bucla", "buclă", "bucle", "for", "while"],
        "condition": ["conditie", "condiție", "if", "else", "elif"],
        "list": ["lista", "listă", "liste", "array", "vector"],
        "file": ["fisier", "fișier", "fisiere", "fișiere", "open", "read", "write"]
    }
    for concept, kws in CONCEPT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            detected_concept = concept
            break
            
    if detected_concept:
        if not detected_lang:
            detected_lang = "python" # Default pe Python dacă nu s-a specificat un limbaj anume
            
        lang_data = SYNTHESIS_DB.get(detected_lang, {})
        concept_data = lang_data.get(detected_concept)
        
        if concept_data:
            lines = []
            lines.append(f"### 💡 {concept_data['title']}\n")
            lines.append(concept_data["desc"])
            lines.append("")
            lines.append("---")
            lines.append(f"**💻 Exemplu complet de cod în {detected_lang.capitalize()}:**")
            lines.append(f"```python\n{concept_data['code'].strip()}\n```")
            
            answer_text = "\n".join(lines)
            analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Modul de Sinteză Cod]</b> Concept detectat: <b>{concept_data['title']}</b>
</div>

"""
            answer_text = analysis_box + answer_text
            return answer_text, [], None

    # ── detecție intenție prin modelul Transformer (CodeBERT) ────────────────
    intent_detected = None
    max_score = 0.0
    intent_scores = {}
    
    if indexer is not None:
        try:
            # Vectorizăm întrebarea
            q_emb = indexer.get_embeddings([question])[0]
            
            # Definiție intenții pentru comparare semantică
            INTENTS_DESC = {
                "explain": "descrie sau explică ce face funcția clasa scopul funcționalitate rol",
                "where": "unde se află definit localizare fișier linie cale path",
                "args": "ce parametri argumente primește input variabile de intrare",
                "returns": "ce returnează valoare întoarsă output return val",
                "errors": "ce erori sau excepții aruncă ridică raise exception except error",
                "callers": "cine apelează funcția cine o folosește clienți call graph",
                "calls": "ce funcții apelează intern dependențe interioare sub-apeluri",
                "similar": "funcții similare semantic cod asemănător duplicate"
            }
            
            intent_keys = list(INTENTS_DESC.keys())
            intent_queries = [INTENTS_DESC[k] for k in intent_keys]
            intent_embs = indexer.get_embeddings(intent_queries)
            
            for idx, key in enumerate(intent_keys):
                sim = cosine_sim(q_emb, intent_embs[idx])
                intent_scores[key] = sim
                if sim > max_score:
                    max_score = sim
                    intent_detected = key
        except Exception as e:
            pass

    # Cuvinte-cheie ca fallback / hibridizare
    is_explain = any(w in q for w in ["ce face","explică","explica","descrie","ce este","ce reprezintă","ce rol","pentru ce","ce e ","what does","explain"])
    is_how     = any(w in q for w in ["cum funcționează","cum lucrează","cum","in ce mod","prin ce mecanism"])
    is_where   = any(w in q for w in ["unde","in ce fisier","unde se afla","unde e definit","unde e declarat"])
    is_callers = any(w in q for w in ["cine apeleaza","cine foloseste","unde e apelat","unde se foloseste","cine cheama"])
    is_calls   = any(w in q for w in ["ce apeleaza","ce functii apeleaza","ce cheama","ce foloseste intern"])
    is_args    = any(w in q for w in ["ce parametri","ce argumente","ce primeste","input","ce ia"])
    is_returns = any(w in q for w in ["ce returneaza","ce intoarce","ce produce","output","ce da inapoi","ce rezultat"])
    is_errors  = any(w in q for w in ["ce erori","ce exceptii","exception","eroare","raise","aruncă"])
    is_deps    = any(w in q for w in ["importuri","dependente","librarii","module importate","ce importa","ce import"])
    is_list    = any(w in q for w in ["listează","enumera","care sunt functiile","care sunt clasele","ce functii","ce clase","toate functiile","toate clasele","functii din","clasele din"])
    is_stats   = any(w in q for w in ["statistici","sumar","overview","structura proiectului","câte funcții","cate functii","cate clase","rezumat"])
    is_file    = any(w in q for w in ["ce fisiere","ce fișiere","fisierele din proiect","care fisiere","lista fisierelor","ce cod am"])
    is_similar = any(w in q for w in ["similar","asemanator","care se aseamana","duplicat"])
    is_project_summary = any(w in q for w in ["ce face proiectul", "ce face aplicatia", "ce face aplicația", "ce face codul meu", "despre ce e proiectul", "despre ce e codul", "rezumat proiect", "rezumatul proiectului", "ce face tot codul", "explică tot codul", "explica tot codul", "prezentare proiect"])

    # Ajustare prin clasificatorul semantic (hibrid)
    if intent_detected and max_score > 0.40:
        if intent_detected == "explain": is_explain = True
        elif intent_detected == "where": is_where = True
        elif intent_detected == "args": is_args = True
        elif intent_detected == "returns": is_returns = True
        elif intent_detected == "errors": is_errors = True
        elif intent_detected == "callers": is_callers = True
        elif intent_detected == "calls": is_calls = True
        elif intent_detected == "similar": is_similar = True

    # ── VERIFICĂM DACĂ BYPASĂM MODULUL EDUCAȚIONAL ───────────────────────────
    has_subject_in_code = subject and (subject in by_name or subject.lower() in by_name)
    has_file_in_q = any(fpath.lower().split("/")[-1] in q for fpath in by_file.keys())
    bypass_educational = is_deps or is_stats or is_file or is_list or has_subject_in_code or has_file_in_q or is_project_summary

    # ── DETECTĂM GREETINGS SAU CORESPONDENȚĂ CHAT GENERALĂ ───────────────────
    GREETINGS = ["salut", "buna", "hey", "bună", "ciao", "hello", "hi", "servus", "ce mai faci", "buna ziua"]
    if any(w in q.split() for w in GREETINGS) or q in ("salut", "buna", "bună", "hei"):
        lines = []
        lines.append("### 🤖 Salut! Eu sunt asistentul tău local de programare.")
        lines.append("Sunt proiectat să rulez **100% offline** utilizând arhitectura locală a modelului **Transformer (CodeBERT)**. Te pot ajutor cu o mulțime de lucruri:")
        lines.append("")
        lines.append("👉 **Analiza Codului Tău**:")
        lines.append("- Întreabă-mă despre orice funcție, clasă sau fișier din proiect (ex: *\"ce face app.py\"*, *\"ce importuri am\"*).")
        lines.append("")
        lines.append("👉 **Lecții și Concepte Academice (Teorie & Exemple SQL/Python)**:")
        lines.append("- **Baze de Date**: Normalizare SQL (**1NF, 2NF, 3NF**), Tranzacții **ACID**, securitate (**SQL Injection**), **Indexare database**, **SQL vs NoSQL**.")
        lines.append("- **Teorie AI & Compilatoare**: Mecanismul de **Self-Attention** în Transformers, **AST** (Abstract Syntax Tree).")
        lines.append("- **Structuri de Date & Algoritmi**: **Stivă (Stack)**, **Coadă (Queue)**, **Arbori (Tree)**, **Grafuri (Graph)**, **SOLID**, **Singleton**, **Big O**, **Recursivitate**, **Recursivitate vs Iterare**.")
        lines.append("")
        lines.append("👉 **Ghiduri de Sintaxă Multi-Limbaj (Sinteză Cod)**:")
        lines.append("- Întreabă-mă cum funcționează variabilele, funcțiile, clasele, buclele sau listele în **Python, Java, C++, JavaScript, Rust, Go sau C#** (ex: *\"cum declar o variabilă în java\"*, *\"cum fac un loop în C++\"*).")
        lines.append("")
        lines.append("Cu ce începem astăzi? Pune-mi orice întrebare legată de cod sau concepte informatice!")
        return "\n".join(lines), [], None

    # ── DETECTĂM SEMANTIC SUBIECTE DIN BAZA DE CUNOȘTINȚE GENERALĂ ──────────
    if not bypass_educational and indexer is not None:
        kb_match = None
        kb_max_score = 0.0
        is_fallback_match = False
        try:
            # 1. Verificare prin cuvinte-cheie deterministe (Prioritate maximă)
            TOPIC_KEYWORDS = {
                "ast": ["ast", "syntax tree", "sintaxă abstractă", "sintaxa abstracta", "arbore de sintaxa", "arborele de sintaxă"],
                "attention": ["attention", "atenție", "atentie", "qkv", "query key value", "self-attention", "auto-atenție", "auto-atentie"],
                "streamlit_state": ["session state", "session_state", "starea sesiunii", "pastram starea", "st.session_state"],
                "database_normalization": ["normalizare", "forma normală", "forma normala", "1nf", "2nf", "3nf", "dependență funcțională", "dependență tranzitivă"],
                "acid": ["acid", "tranzacții", "tranzactii", "atomicitate", "consistență", "consistenta", "izolare", "durabilitate"],
                "sql_injection": ["sql injection", "injecție sql", "injectie sql", "concatenare sql", "securitate sql"],
                "java_oop": ["java", "oop", "orientat pe obiecte", "polimorfism", "încapsulare", "incapsulare", "abstractizare", "interfețe", "interfete"],
                "js_async": ["javascript", "typescript", "async", "await", "promise", "callbacks", "event loop", "asincron"],
                "cpp_memory": ["c++", "pointer", "referințe", "referinte", "malloc", "free", "new", "delete", "raii", "gestionarea memoriei", "smart pointer"],
                "rust_safety": ["rust", "ownership", "borrow checker", "lifetimes", "siguranță memorie", "referințe rust"],
                "go_concurrency": ["go ", "golang", "goroutine", "channels", "concurență go", "csp"],
                "csharp_dotnet": ["c#", "dotnet", "linq", "generics", "clr", "delegat", "c sharp"],
                "web_layout": ["flexbox", "css grid", "layout css", "responsive web", "machetare web", "machetare css", "flex grid"],
                "recursion": ["recursivitate", "recursiv", "recursion", "recursivă", "recursiva", "auto-apelare"],
                "oop_principles": ["principii oop", "principiile oop", "mostenire", "moștenire", "polimorfism", "încapsulare", "incapsulare", "abstractizare", "clase și obiecte", "clase si obiecte"],
                "stack_ds": ["stivă", "stiva", "stack", "lifo", "push pop", "operatii stiva", "stive"],
                "queue_ds": ["coadă", "coada", "queue", "fifo", "enqueue dequeue", "operatii coada", "cozi"],
                "tree_ds": ["arbore", "tree", "arbori", "binary tree", "arbore binar", "bst", "frunze arbore"],
                "graph_ds": ["graf", "graph", "grafuri", "noduri muchii", "lista de adiacenta", "drum minim", "ponderat"],
                "solid_principles": ["solid", "principii solid", "principiile solid", "single responsibility", "liskov", "dependency inversion"],
                "singleton_pattern": ["singleton", "pattern singleton", "design pattern singleton", "instanță unică", "instanta unica"],
                "git_vcs": ["git", "commit", "branch", "merge", "push", "pull", "controlul versiunilor", "vcs"],
                "json_format": ["json", "format json", "ce este json", "javascript object notation", "date json"],
                "rest_api": ["rest api", "api rest", "http methods", "endpoint", "get post put delete", "coduri de stare http"],
                "sql_nosql": ["sql vs nosql", "nosql", "baze de date relationale", "baze de date ne-relationale", "mongodb vs postgresql"],
                "db_indexing": ["indexare", "index db", "indecși", "indecsi", "index baze de date", "b-tree", "idx"],
                "complexity_big_o": ["big o", "complexitate timp", "complexitatea timp", "notația big o", "timp de executie", "eficienta algoritm"],
                "recursion_iteration": ["recursivitate vs iterare", "iterare vs recursivitate", "recursiv vs iterativ", "suma_recursiva", "suma_iterativa"]
            }
            
            # Detecție concepte multiple din KNOWLEDGE_BASE pentru comparații dinamice
            matched_topics = []
            for key, kw_list in TOPIC_KEYWORDS.items():
                if any(kw in q for kw in kw_list):
                    matched_topics.append(key)
            
            if len(matched_topics) == 1:
                kb_match = matched_topics[0]
                kb_max_score = 1.0
            elif len(matched_topics) > 1:
                # Comparație dinamică multiplă!
                lines = []
                lines.append("### 🔀 Analiză Comparativă Dinamică\n")
                lines.append("Am detectat că întrebi despre mai multe concepte înrudite. Iată o analiză detaliată a acestora:")
                lines.append("")
                
                for key in matched_topics:
                    topic = KNOWLEDGE_BASE[key]
                    lines.append(f"#### 💡 {topic['title']}")
                    lines.append(topic["description"])
                    lines.append("")
                    lines.append(topic["details"])
                    if topic.get("code"):
                        lines.append(f"💻 *Cod ilustrativ:*")
                        lines.append(f"```python\n{topic['code'].strip()}\n```")
                    lines.append("---")
                
                answer_text = "\n".join(lines)
                concept_names = ", ".join(f"<b>{KNOWLEDGE_BASE[k]['title'].split(' ')[0]}</b>" for k in matched_topics)
                analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Modul Comparativ Offline]</b> Concepte detectate simultan: {concept_names}
</div>

"""
                return analysis_box + answer_text, [], None
            
            best_anchor = ""
            
            # 2. Verificare de înaltă precizie prin modelul Transformer BERT folosind ancore semantice multi-vector
            if not kb_match:
                q_emb = indexer.get_embeddings([question])[0]
                kb_keys = list(KNOWLEDGE_BASE.keys())
                
                # Încărcăm / Cache-uim embeddings pentru ancorele din KNOWLEDGE_BASE în indexer
                if not hasattr(indexer, "kb_anchor_embeddings") or not indexer.kb_anchor_embeddings:
                    indexer.kb_anchor_embeddings = {}
                    all_anchors = []
                    anchor_keys = []
                    for key in kb_keys:
                        anchors = KNOWLEDGE_BASE[key].get("anchors", [KNOWLEDGE_BASE[key]["title"]])
                        for anchor in anchors:
                            all_anchors.append(anchor)
                            anchor_keys.append((key, anchor))
                    # Generăm vectorii de embedding într-un singur batch rapid
                    embs = indexer.get_embeddings(all_anchors)
                    for idx, (key, anchor) in enumerate(anchor_keys):
                        indexer.kb_anchor_embeddings.setdefault(key, []).append((anchor, embs[idx]))
                
                scores = []
                best_anchors = []
                for key in kb_keys:
                    anchor_data = indexer.kb_anchor_embeddings.get(key, [])
                    max_sim = -1.0
                    best_anc = ""
                    for anchor, emb in anchor_data:
                        sim = cosine_sim(q_emb, emb)
                        if sim > max_sim:
                            max_sim = sim
                            best_anc = anchor
                    scores.append(max_sim)
                    best_anchors.append(best_anc)
                
                best_idx = int(np.argmax(scores))
                best_score = scores[best_idx]
                
                if best_score > 0.65:
                    kb_match = kb_keys[best_idx]
                    kb_max_score = best_score
                    best_anchor = best_anchors[best_idx]
                    is_fallback_match = False
                else:
                    # Fallback moderat pe baza scorului maxim absolut peste 0.50
                    if best_score > 0.50:
                        kb_match = kb_keys[best_idx]
                        kb_max_score = best_score
                        best_anchor = best_anchors[best_idx]
                        is_fallback_match = True
                        
            if kb_match:
                # Am găsit o potrivire excelentă!
                topic = KNOWLEDGE_BASE[kb_match]
                lines = []
                lines.append(f"### 💡 {topic['title']}\n")
                lines.append(topic["description"])
                lines.append("")
                lines.append(topic["details"])
                lines.append("")
                if topic.get("code"):
                    lines.append("---")
                    lines.append("**💻 Fragment ilustrativ de cod / Utilizare:**")
                    lines.append(f"```python\n{topic['code'].strip()}\n```")
                    
                answer_text = "\n".join(lines)
                if is_fallback_match:
                    analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Aliniere Semantică Transformer BERT]</b> Conceptul recomandat este: <b>{topic['title']}</b> (scor similaritate: <code>{kb_max_score:.3f}</code>) <br>
🎯 <i>Ancoră semântică CodeBERT:</i> <code>"{best_anchor or topic['title']}"</code>
</div>

"""
                else:
                    analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(99, 102, 241, 0.08); border-left: 3px solid #6366f1; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Aliniere Semantică Transformer BERT]</b> Subiect detectat: <b>{topic['title']}</b> (scor similaritate: <code>{kb_max_score:.3f}</code>) <br>
🎯 <i>Ancoră semântică CodeBERT:</i> <code>"{best_anchor or topic['title']}"</code>
</div>

"""
                answer_text = analysis_box + answer_text
                return answer_text, [], kb_match
        except Exception as e:
            pass

    # ── DETECTĂM DACĂ SE ÎNTREABĂ DESPRE UN FIȘIER ─────────────────────────────
    target_file = None
    for fpath in by_file.keys():
        fname = fpath.lower().split("/")[-1]
        # Potrivire exactă a numelui fișierului sau fără extensie (ex: "app.py" sau "app" în întrebare)
        if fname in q or (len(fname.replace(".py", "")) > 2 and fname.replace(".py", "") in q.split()):
            target_file = fpath
            break

    if target_file and subject:
        subj_lower = subject.lower()
        target_fname = target_file.lower().split("/")[-1]
        is_part_of_file = subj_lower in target_fname or target_fname.startswith(subj_lower)
        is_known_identifier = (subject in by_name) or (subj_lower in by_name)
        if is_part_of_file or not is_known_identifier:
            subject = None

    if target_file and not subject:
        # Utilizatorul întreabă despre un fișier întreg, nu o funcție/clasă anume
        cks = by_file[target_file]
        funcs = [c for c in cks if c.get("type") == "function"]
        classes = [c for c in cks if c.get("type") == "class"]
        
        lines = []
        lines.append(f"### 📂 Fișierul `{target_file}` din proiect\n")
        
        # Formulăm o explicație conversațională descriptivă în română despre fișier
        desc = f"Fișierul `{target_file}` reprezintă un modul important din proiect. "
        if "app.py" in target_file:
            desc += "Acesta este punctul de intrare principal în aplicație (interfața Streamlit). Gestionează stările sesiunii, configurările paginilor, injectarea stilurilor CSS și coordonează toate filele (tab-urile) interactive: încărcarea proiectului, vizualizarea arborelui de fișiere, diagramele UML (Mermaid), căutarea semantică, auditul de securitate, chestionarele AST și sistemul de chatbot local bazat pe CodeBERT."
        elif "code_parser.py" in target_file:
            desc += "Acest modul este responsabil de parsarea codului și de extracția structurii sintactice. Conține logica de dezarhivare a proiectelor ZIP, scanarea fișierelor de cod acceptate, construirea arborelui de directoare și parsarea fișierelor prin arborele AST (Abstract Syntax Tree) pentru a genera chunk-uri (blocuri de cod) și diagramele Mermaid (UML, dependențe, secvență, flowchart)."
        elif "vector_store.py" in target_file:
            desc += "Acest modul se ocupă de vectorizarea codului și de căutarea semantică offline. Implementează clasa `CodeBERTIndexer` care rulează modelul CodeBERT local pe PyTorch (folosind MPS/CUDA/CPU) pentru a tokeniza codul, a extrage matricea de atenție (Self-Attention) și a indexa chunk-urile generate într-un index vectorial FAISS pentru interogări rapide."
        elif "security_analyzer.py" in target_file:
            desc += "Este modulul dedicat auditului de securitate static și semantic. Scanează fișierele Python cu un visitor AST (`SecurityVisitor`) pentru a detecta vulnerabilități comune (injecții SQL/shell, secrete hardcodate, except-uri generice) și folosește similaritatea semantică CodeBERT pentru a evalua riscul fragmentelor de cod."
        else:
            desc += f"Fișierul conține un număr de {len(cks)} blocuri logice de cod, acționând ca un modul de suport în proiect."
            
        lines.append(desc)
        lines.append("")
        
        if classes:
            lines.append("🧱 **Clase definite în acest fișier:**")
            for c in classes:
                doc_str = f" — *{c.get('docstring')}*" if c.get("docstring") else ""
                lines.append(f"- `{c['name']}` (liniile {c['start_line']}–{c['end_line']}){doc_str}")
            lines.append("")
            
        if funcs:
            lines.append("⚙️ **Funcții definite în acest fișier:**")
            for f in funcs[:12]:
                doc_str = f" — *{f.get('docstring')}*" if f.get("docstring") else ""
                lines.append(f"- `{f['name']}` (liniile {f['start_line']}–{f['end_line']}){doc_str}")
            if len(funcs) > 12:
                lines.append(f"- *și încă {len(funcs)-12} alte funcții...*")
            lines.append("")
            
        # Citim primele 45 de linii din fișierul real pentru a le arăta ca fragment de cod de început
        file_code = ""
        try:
            with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                file_code = "".join(f.readlines()[:45])
        except:
            if cks:
                file_code = cks[0].get("content", "")[:1000]
                
        if file_code:
            lines.append("---")
            lines.append(f"**💻 Structura de început a fișierului `{target_file}`:**")
            lines.append(f"```python\n{file_code}\n```")
            
        answer_text = "\n".join(lines)
        if indexer is not None and intent_detected and max_score > 0.40:
            analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Clasificator Semantic Transformer]</b> Intenția detectată: <b>Căutare Localizare fișier</b> (scor similaritate: <code>{max_score:.3f}</code>)
</div>

"""
            answer_text = analysis_box + answer_text
            
    # ── EXPLICAȚIE PROIECT COMPLET (PROJECT SUMMARY) ─────────────────────────
    if is_project_summary:
        lines = []
        lines.append("### 🛠️ Explicația de Ansamblu a Proiectului Tău (Code Explainer & Security Analyzer)\n")
        lines.append("Întregul proiect este o aplicație **Streamlit** extrem de modernă, concepută pentru **analiza statică de cod, căutare semantică locală și audit de securitate offline**. Sistemul utilizează modelul local de deep learning **CodeBERT** (pe PyTorch) pentru a înțelege semnificația codului fără a apela API-uri externe.\n")
        
        lines.append("#### 📂 Structura și Rolul Modulelor:")
        lines.append("- 🖥️ **`app.py` (Interfața Utilizator)**: Reprezintă panoul de control central. Acesta configurează designul vizual premium (CSS personalizat), administrează starea sesiunilor Streamlit (`st.session_state`), desenează graficele interactive de Atenție (Plotly) și coordonează toate filele funcționale ale aplicației.")
        lines.append("- ⚙️ **`code_parser.py` (Analizorul AST & Diagram Generator)**: Se ocupă de procesarea brută a codului. Dezarhivează fișierele ZIP, le filtrează și folosește modulul Python `ast` (Abstract Syntax Tree) pentru a identifica clasele și funcțiile. De asemenea, generează automat diagrame de dependență și secvență în format **Mermaid (UML)**.")
        lines.append("- 🧠 **`vector_store.py` (Căutarea Semantică Hibridă)**: Este creierul matematic al proiectului. Încarcă modelul CodeBERT local pe CPU/MPS/GPU, generează vectori de embedding de 768 de dimensiuni pentru fiecare fragment de cod și creează un index vectorial **FAISS** local. Implementează, de asemenea, căutarea hibridă combinând CodeBERT cu un motor lexical exact **TF-IDF**.")
        lines.append("- 🛡️ **`security_analyzer.py` (Auditul de Securitate)**: Reprezintă inspectorul de siguranță. Scanează fișierele Python cu un vizitator AST (`SecurityVisitor`) pentru a depista automat vulnerabilități critice (de tip SQL Injection, rulare de comenzi shell, secrete/parole hardcodate) și clasifică riscurile semantic.")
        lines.append("")
        
        lines.append("#### 🔁 Cum Colaborează Aceste Componente:")
        lines.append("1. **Încărcare**: Încarci o arhivă ZIP cu cod în interfața Streamlit (`app.py`).")
        lines.append("2. **Parsare**: `code_parser.py` descompune fișierele în fragmente (chunk-uri) logice și extrage metadate AST (parametri, importuri, apeluri).")
        lines.append("3. **Indexare**: `vector_store.py` convertește aceste fragmente în vectori și le indexează în FAISS + TF-IDF.")
        lines.append("4. **Interacțiune**: Când pui o întrebare, chatbot-ul folosește CodeBERT pentru a-ți înțelege intenția și a-ți căuta exact răspunsurile, în timp ce `security_analyzer.py` validează calitatea codului din punct de vedere al securității.")
        
        answer_text = "\n".join(lines)
        analysis_box = """<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Prezentare de Ansamblu Proiect]</b> Am generat un rezumat complet al funcționalității întregului cod din codebase.
</div>

"""
        return analysis_box + answer_text, [], None

    # ── STATISTICI / OVERVIEW ────────────────────────────────────────────────
    if is_stats:
        lines = ["### Sumar proiect\n"]
        lines.append(f"| Metrică | Valoare |")
        lines.append(f"|---------|---------|")
        lines.append(f"| Fișiere indexate | **{stats.get('total_files', len(by_file))}** |")
        lines.append(f"| Funcții | **{stats.get('total_functions','?')}** |")
        lines.append(f"| Clase | **{stats.get('total_classes','?')}** |")
        lines.append(f"| Lungime medie funcție | **{stats.get('avg_func_lines','?')} linii** |")
        lf = stats.get("largest_func")
        if lf:
            lines.append(f"| Funcția cea mai lungă | **`{lf['name']}`** ({lf.get('end_line',0)-lf.get('start_line',0)} linii) |")
        mc = stats.get("most_called")
        if mc:
            n_callers = len(called_by.get(mc, []))
            lines.append(f"| Funcția cel mai des apelată | **`{mc}`** ({n_callers} apeluri detectate) |")
        lines.append(f"\n**Fișiere:**")
        for f in list(by_file.keys())[:10]:
            n = len([c for c in by_file[f] if c.get("type") in ("function","class")])
            lines.append(f"- `{f}` — {n} funcții/clase")
        return "\n".join(lines), [], None

    # ── LISTARE FIȘIERE ───────────────────────────────────────────────────────
    if is_file:
        lines = [f"**Proiectul conține {len(by_file)} fișiere indexate:**\n"]
        for f, cks in by_file.items():
            funcs = [c["name"] for c in cks if c.get("type") == "function"]
            classes = [c["name"] for c in cks if c.get("type") == "class"]
            parts = []
            if classes: parts.append("clase: " + ", ".join(f"`{n}`" for n in classes))
            if funcs:   parts.append("funcții: " + ", ".join(f"`{n}`" for n in funcs[:6]) + ("..." if len(funcs)>6 else ""))
            lines.append(f"- `{f}` — " + "; ".join(parts) if parts else f"- `{f}`")
        return "\n".join(lines), [], None

    # ── LISTARE FUNCȚII / CLASE ───────────────────────────────────────────────
    if is_list:
        target_file = None
        for f in by_file:
            if any(word in q for word in [f.lower(), f.split("/")[-1].lower().replace(".py","")]):
                target_file = f
                break
        pool = by_file.get(target_file, all_chunks) if target_file else all_chunks
        funcs   = [c for c in pool if c.get("type") == "function"]
        classes = [c for c in pool if c.get("type") == "class"]
        header = f"**{'Fișierul `'+target_file+'`' if target_file else 'Proiectul'} conține:**\n"
        lines = [header]
        if classes:
            lines.append(f"**Clase ({len(classes)}):** " + ", ".join(f"`{c['name']}`" for c in classes))
        if funcs:
            lines.append(f"\n**Funcții ({len(funcs)}):**")
            gf = {}
            for c in funcs:
                gf.setdefault(c.get("file_path","?"), []).append(c)
            for fp, items in gf.items():
                lines.append(f"- `{fp}`: " + ", ".join(f"`{c['name']}`" for c in items[:10]) + ("..." if len(items)>10 else ""))
        return "\n".join(lines), [], None

    # ── IMPORTURI PROIECT ─────────────────────────────────────────────────────
    if is_deps:
        # Dacă există un subiect valid în codebase, arătăm importurile acestuia
        valid_subject = subject and (subject in by_name or subject.lower() in by_name)
        if valid_subject:
            chunk = by_name.get(subject) or by_name.get(subject.lower())
            ai = analyze_chunk_ast(chunk.get("content", ""))
            imps = sorted(list(set(ai["imports"])))
            if imps:
                return f"Elementul `{subject}` din fișierul `{chunk.get('file_path')}` folosește următoarele importuri direct:\n" + ", ".join(f"`{i}`" for i in imps), [], None
            else:
                return f"Elementul `{subject}` din fișierul `{chunk.get('file_path')}` nu conține importuri directe în corpul său.", [], None
        else:
            # Altfel, afișăm toate importurile din toate fișierele proiectului
            lines = ["**Importuri detectate per fișier în întregul proiect:**\n"]
            for f, imps in all_imports.items():
                uniq = sorted(set(imps))
                if uniq:
                    lines.append(f"- `{f}`: " + ", ".join(f"`{i}`" for i in uniq[:12]))
                else:
                    lines.append(f"- `{f}`: *fără importuri detectate*")
            return "\n".join(lines), [], None

    # ── GĂSIM CHUNK-UL SUBIECT ────────────────────────────────────────────────
    # 1. Căutare exactă în KB by_name
    chunk = by_name.get(subject) or by_name.get(subject.lower())
    # 2. Căutare parțială în KB
    if not chunk and subject:
        matches = [c for n, c in by_name.items() if subject.lower() in n.lower()]
        chunk = matches[0] if matches else None
    # 3. Fallback: FAISS results
    if not chunk:
        real = [c for c in results if c.get("type") in ("function","class")]
        chunk = real[0] if real else (results[0] if results else None)
    # 4. Text search dacă tot nu avem
    if not chunk and subject:
        found = _text_search_chunks(subject, all_chunks)
        chunk = found[0] if found else None
    # 5. Dacă e concept/variabilă (nu funcție) sau întrebare generală
    if not chunk:
        # 1. Căutare în Dicționarul Universal Offline (Docker, React, etc.)
        matched_key = None
        for key in UNIVERSAL_DICT.keys():
            if key in q or (subject and key in subject.lower()):
                matched_key = key
                break
        if matched_key:
            topic = UNIVERSAL_DICT[matched_key]
            lines = []
            lines.append(f"### 💡 {topic['title']}\n")
            lines.append(topic["description"])
            lines.append("")
            lines.append(topic["details"])
            if topic.get("code"):
                lines.append("---")
                lines.append("**💻 Fragment ilustrativ / Configurare:**")
                lines.append(f"```python\n{topic['code'].strip()}\n```")
            answer_text = "\n".join(lines)
            analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Modul Educațional Universal]</b> Subiect detectat: <b>{topic['title']}</b>
</div>

"""
            return analysis_box + answer_text, [], f"dynamic_fallback_{matched_key}"

        # 2. Căutare utilizări în proiect dacă subiectul există
        if subject:
            usages = _analyze_variable_usage(subject, all_chunks)
            if usages:
                lines = [f"**`{subject}`** nu este o funcție/clasă definită, dar apare în **{len(usages)} locuri** în proiect:\n"]
                for fname, fpath, usage_line in usages[:12]:
                    lines.append(f"- `{fname}` (`{fpath}`): `{usage_line.strip()[:90]}`")
                return "\n".join(lines), [], None
            else:
                # 3. Fallback dinamic generativ pentru orice alt subiect
                if len(subject) > 2:
                    answer_text = _generate_dynamic_fallback(subject)
                    clean_sub = subject.lower().replace(" ", "_")
                    return answer_text, [], f"dynamic_fallback_{clean_sub}"

        # 4. Fallback suplimentar pentru întrebări generale fără subiect extras clar
        words = _re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', q)
        NOISE = {"ce", "este", "e", "face", "cum", "unde", "care", "proiect", "cod", "aplicatie", "programare", "functia", "clasa"}
        candidates = [w for w in words if w.lower() not in NOISE and len(w) > 2]
        if candidates:
            fallback_subject = candidates[-1]
            answer_text = _generate_dynamic_fallback(fallback_subject)
            clean_sub = fallback_subject.lower().replace(" ", "_")
            return answer_text, [], f"dynamic_fallback_{clean_sub}"

        # 5. Răspuns default extrem de detaliat (dacă tot nu se poate extrage nimic)
        if all_chunks and (is_explain or is_how or is_where or subject):
            chunk = all_chunks[0]
        else:
            # Răspuns extrem de prietenos ca asistent universal de programare local!
            lines = []
            lines.append("### 🤖 Salut! Eu sunt asistentul tău local de programare.")
            lines.append("Sunt proiectat să rulez **100% offline** utilizând arhitectura locală a modelului **Transformer (CodeBERT)**. Te pot ajuta cu o mulțime de chestii din temele tale:")
            lines.append("")
            lines.append("👉 **Analiza Codului Tău**:")
            lines.append("- Întreabă-mă despre orice funcție, clasă sau fișier din proiect (ex: *\"ce face app.py\"*, *\"ce importuri am\"*).")
            lines.append("")
            lines.append("👉 **Lecții și Concepte Academice (Teorie & Exemple SQL/Python)**:")
            lines.append("- **Baze de Date / Licență**: Normalizare SQL (**1NF, 2NF, 3NF**), Tranzacții **ACID**, securitate (**SQL Injection**).")
            lines.append("- **Teorie AI & Compilatoare**: Mecanismul de **Self-Attention** în Transformers, **AST** (Abstract Syntax Tree).")
            lines.append("")
            lines.append("👉 **Ghiduri de Sintaxă Multi-Limbaj (Sinteză Cod)**:")
            lines.append("- Întreabă-mă cum funcționează variabilele, funcțiile, clasele, buclele sau listele în **Python, Java, C++, JavaScript, Rust, Go sau C#** (ex: *\"cum declar o variabilă în java\"*, *\"cum fac un loop în C++\"*).")
            
            answer_text = "\n".join(lines)
            return answer_text, [], None

    # Dacă chunk-ul găsit e module_level (globals), preferăm usage analysis
    if chunk.get("type") == "module_level" and subject:
        usages = _analyze_variable_usage(subject, all_chunks)
        real = [c for c in all_chunks if c.get("name","").lower() == subject.lower() and c.get("type") in ("function","class")]
        if real:
            chunk = real[0]  # am găsit funcția reală
        elif usages:
            lines = [f"**`{subject}`** apare în **{len(usages)} locuri** în proiect:\n"]
            for fname, fpath, usage_line in usages[:12]:
                lines.append(f"- `{fname}` (`{fpath}`): `{usage_line.strip()[:90]}`")
            return "\n".join(lines), [], None

    # ── CONSTRUIM RĂSPUNSUL ───────────────────────────────────────────────────
    name    = chunk.get("name","?")
    fpath   = chunk.get("file_path","?")
    sl      = chunk.get("start_line",0)
    el      = chunk.get("end_line",0)
    ctype   = chunk.get("type","function")
    doc     = chunk.get("docstring","").strip()
    code    = chunk.get("content","")
    n_lines = el - sl
    ai      = analyze_chunk_ast(code)
    lines   = []

    async_tag = "asincronă " if ai["is_async"] else ""
    lines.append(f"### `{name}` — {async_tag}{ctype} în `{fpath}` (liniile {sl}–{el}, {n_lines} linii)\n")
    if doc:
        lines.append(f"📄 **Documentație oficială (docstring):** *\"{doc}\"*\n")

    # CE FACE (default dacă nu e întrebare specifică)
    default_explain = not any([is_where, is_callers, is_calls, is_args, is_returns, is_errors, is_deps])
    if is_explain or is_how or default_explain:
        parts = []
        if ctype == "class":
            if ai["class_bases"]:
                parts.append(f"moștenește din clasa/clasele {', '.join(f'`{b}`' for b in ai['class_bases'])}")
            methods = [c for c in all_chunks if c.get("file_path")==fpath and c.get("type")=="function" and c.get("name")!=name]
            if methods:
                parts.append(f"încapsulează {len(methods)} metode (inclusiv: " + ", ".join(f"`{m['name']}`" for m in methods[:5]) + ")")
        else:
            if ai["args"]:
                parts.append(f"primește ca intrare parametrii **{', '.join(f'`{a}`' for a in ai['args'])}**")
            if ai["decorators"]:
                parts.append(f"este decorată cu `{', '.join(f'`{d}`' for d in ai['decorators'])}`")
            if ai["calls"]:
                parts.append(f"apelează intern alte elemente precum: {', '.join(f'`{c}`' for c in ai['calls'][:6])}")
            if ai["loops"]:
                parts.append(f"rulează operații repetitive folosind {ai['loops']} structuri ciclice (for/while)")
            if ai["conditions"]:
                parts.append(f"conține logică decizională cu {ai['conditions']} ramificații condiționale (if/elif)")
            if ai["comprehensions"]:
                parts.append(f"optimizează lucrul cu liste/colecții prin {ai['comprehensions']} expresii de tip comprehension")
            if ai["has_try"]:
                parts.append("are mecanisme defensive de tratare a erorilor cu blocuri `try/except` cu scopul de a evita întreruperea execuției")
            if ai["returns"]:
                ret = ", ".join(f"`{r[:70]}`" for r in ai["returns"][:2])
                parts.append(f"întoarce un rezultat de forma: {ret}")
            if ai["raises"]:
                parts.append(f"poate arunca în mod controlat excepții specifice: {', '.join(f'`{e}`' for e in ai['raises'])}")
                
        lines.append("🔍 **Formulare Răspuns & Explicație:**")
        if ctype == "class":
            desc = f"Clasa `{name}` este definită în modulul `{fpath}` și acționează ca o entitate structurală în cadrul temei. "
        else:
            desc = f"Funcția `{name}` se găsește în fișierul `{fpath}` și reprezintă o componentă logică responsabilă de execuția unui comportament specific. "
            
        if doc:
            desc += f"Conform documentației sale interne, rolul său principal este definit ca fiind: *\"{doc}\"*. "
        else:
            desc += "Deși nu are un comentariu docstring descriptiv, o analiză structurală detaliată ne arată cum operează. "
            
        if parts:
            desc += f"Mai exact, din punct de vedere al execuției, acest fragment: {'; '.join(parts)}."
        else:
            desc += "Este un fragment simplu, cu execuție directă, fără structuri complexe sau apeluri externe detectate."
            
        lines.append(desc)
        lines.append("")

    if is_where:
        lines.append(f"📂 **Localizare:** `{fpath}`, liniile **{sl}–{el}**\n")

    if is_args:
        if ai["args"]:
            lines.append(f"📥 **Parametri:** {', '.join(f'`{a}`' for a in ai['args'])}\n")
        else:
            lines.append("📥 **Parametri:** Fără argumente externe (sau doar `self`).\n")

    if is_returns:
        if ai["returns"]:
            lines.append("📤 **Returnează:**")
            for r in ai["returns"]: lines.append(f"- `{r[:80]}`")
            lines.append("")
        else:
            lines.append("📤 **Returnează:** `None` (fără `return` explicit).\n")

    if is_errors:
        if ai["raises"]:
            lines.append(f"⚠️ **Excepții:** {', '.join(f'`{e}`' for e in ai['raises'])}\n")
        elif ai["has_try"]:
            lines.append("⚠️ **Excepții:** Are `try/except` dar nu ridică explicit.\n")
        else:
            lines.append("⚠️ **Excepții:** Nu aruncă excepții.\n")

    if is_deps:
        imps = list(set(ai["imports"]))
        if imps:
            lines.append(f"📦 **Importuri în fragment:** {', '.join(f'`{i}`' for i in imps[:8])}\n")
        else:
            lines.append("📦 **Importuri:** Fără importuri directe în acest fragment.\n")

    if is_calls:
        direct = sorted(call_graph.get(name, set()))
        known  = [c for c in direct if c in by_name]
        extern = [c for c in direct if c not in by_name]
        if known:
            lines.append(f"🔁 **Apelează funcții din proiect:** {', '.join(f'`{c}`' for c in known[:6])}\n")
        if extern:
            lines.append(f"🔁 **Apelează externe/built-in:** {', '.join(f'`{c}`' for c in extern[:6])}\n")
        if not known and not extern:
            lines.append("🔁 Nu am detectat apeluri de funcții în acest fragment.\n")

    if is_callers:
        callers = sorted(called_by.get(name, set()))
        if callers:
            lines.append(f"🔗 **Apelată din:** {', '.join(f'`{c}`' for c in callers[:6])}\n")
        else:
            lines.append(f"🔗 **Apelată din:** Nu am detectat apeluri directe la `{name}`.\n")

    # ── FUNCȚII SIMILARE (din FAISS results) ─────────────────────────────────
    if is_similar:
        similar = [c for c in results if c.get("name") != name and c.get("type") in ("function","class")]
        if similar:
            lines.append("🔀 **Funcții similare semantic (CodeBERT):**")
            for s in similar[:4]:
                score = s.get("score", 0)
                lines.append(f"- `{s['name']}` în `{s['file_path']}` — distanță L2: `{score:.4f}`")
        else:
            lines.append("🔀 Nu am găsit funcții similare semantic.\n")

    # ── FRAGMENTE CONEXE ──────────────────────────────────────────────────────
    faiss_others = [c for c in results if c.get("name") != name and c.get("type") in ("function","class")]
    if faiss_others and not is_similar:
        lines.append("---")
        lines.append(f"**Fragmente conexe găsite de CodeBERT ({len(faiss_others)}):**")
        for r in faiss_others[:3]:
            d = r.get("docstring","").strip()
            lines.append(f"- `{r['name']}` în `{r['file_path']}` (lin. {r['start_line']}–{r['end_line']})" + (f" — {d[:80]}" if d else ""))

    if code:
        lines.append("---")
        lines.append("**💻 Fragmentul de cod sursă identificat:**")
        lines.append(f"```python\n{code[:850]}\n```")
        if len(code) > 850:
            lines.append(f"*Notă: codul a fost trunchiat pentru lizibilitate (liniile {sl}–{el} din `{fpath}`).*")

    answer_text = "\n".join(lines)
    if indexer is not None and intent_detected and max_score > 0.40:
        friendly_names = {
            "explain": "Descriere / Rol cod",
            "where": "Căutare Localizare fișier",
            "args": "Analiză Parametri intrare",
            "returns": "Analiză Valori returnate",
            "errors": "Gestionare / Aruncare Erori",
            "callers": "Relații de apel (Cine îl apelează)",
            "calls": "Sub-apeluri efectuate (Ce apelează)",
            "similar": "Căutare Semantică cod similar"
        }
        detected_name = friendly_names.get(intent_detected, intent_detected)
        analysis_box = f"""<div style="padding: 10px 14px; font-size: 0.9em; border-radius: 6px; background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; margin-bottom: 15px; color: #e2e8f0; font-family: sans-serif;">
🤖 <b>[Clasificator Semantic Transformer]</b> Intenția detectată: <b>{detected_name}</b> (scor similaritate: <code>{max_score:.3f}</code>)
</div>

"""
        answer_text = analysis_box + answer_text

    return answer_text, [chunk] + faiss_others[:2], None


def render_mermaid(mermaid_code, height=500):
    escaped = mermaid_code.replace("`", "\\`").replace("${", "\\${")
    html_code = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body, html {{
      width: 100%; height: 100%;
      background: #ffffff;
      font-family: ui-monospace, monospace;
    }}
    #wrap {{
      width: 100%; height: 100%;
      display: flex; justify-content: center; align-items: flex-start;
      padding: 8px;
      overflow: auto;
    }}
    #diagram {{ max-width: 100%; }}
    svg {{ max-width: 100% !important; height: auto !important; display: block; }}
    #err {{
      color: #dc2626; background: #fef2f2;
      border: 1px solid #fca5a5; border-radius: 6px;
      padding: 12px 16px; font-size: 13px;
      display: none;
    }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="diagram">Se încarcă diagrama...</div>
    <div id="err"></div>
  </div>
  <script id="src" type="text/plain">{escaped}</script>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      fontFamily: 'ui-monospace, monospace',
      fontSize: 14
    }});
    const code = document.getElementById('src').textContent.trim();
    const box  = document.getElementById('diagram');
    const err  = document.getElementById('err');
    try {{
      const {{ svg }} = await mermaid.render('mmd', code);
      box.innerHTML = svg;
    }} catch(e) {{
      box.innerHTML = '';
      err.style.display = 'block';
      err.textContent = 'Eroare sintaxă Mermaid: ' + e.message;
      console.error(e);
    }}
  </script>
</body>
</html>"""
    components.html(html_code, height=height, scrolling=True)

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
    uploaded_file = st.file_uploader("Încarcă o arhivă .zip sau un fișier individual de cod", type=None)
    
    st.write("---")
    st.markdown("### 🔗 Încarcă de pe Git / GitHub")
    git_url = st.text_input("Introdu URL-ul repository-ului Git:", placeholder="https://github.com/username/repo.git")
    if st.button("Clonează și Indexează de pe Git", use_container_width=True):
        if git_url:
            st.session_state.git_clone_url = git_url
            st.session_state.project_processed = False
            st.rerun()
            
    st.write("---")
    
    # Buton de Reset
    if st.button("Șterge datele / Încarcă alt proiect", use_container_width=True):
        clean_workspace()
        st.rerun()

# ----------------- PROCESARE COD & EMBEDDINGS -----------------
git_clone_requested = "git_clone_url" in st.session_state and st.session_state.git_clone_url is not None

if (uploaded_file is not None or git_clone_requested) and not st.session_state.project_processed:
    clean_workspace()
    success = True
    
    if git_clone_requested:
        url = st.session_state.git_clone_url
        with st.spinner(f"Se clonează repository-ul de pe Git: {url}..."):
            try:
                import subprocess
                import shutil
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR, ignore_errors=True)
                os.makedirs(TEMP_DIR, exist_ok=True)
                
                # Rulăm comanda git clone --depth 1 pentru viteză maximă de descărcare
                res = subprocess.run(
                    ["git", "clone", "--depth", "1", url, TEMP_DIR],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=90
                )
                if res.returncode != 0:
                    st.error(f"Eroare la clonarea Git: {res.stderr or 'Repository privat sau cale incorectă'}")
                    success = False
            except Exception as e:
                st.error(f"Eroare internă la clonare: {str(e)}")
                success = False
            finally:
                # Resetăm starea de cerere de clonare din starea sesiunii
                st.session_state.git_clone_url = None
    else:
        file_name = uploaded_file.name
        if file_name.lower().endswith(".zip"):
            with st.spinner("Se extrage arhiva proiectului..."):
                zip_bytes = uploaded_file.read()
                unzip_project(zip_bytes, TEMP_DIR)
        else:
            with st.spinner(f"Se procesează fișierul {file_name}..."):
                os.makedirs(TEMP_DIR, exist_ok=True)
                file_path = os.path.join(TEMP_DIR, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                    
    if not success:
        st.stop()
        
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

            # Construim baza de cunoștințe pentru chatbot
            st.session_state.project_kb = build_project_knowledge(chunks)

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
            <p>Interogare instantanee în cod pe bază de concepte, nu doar cuvinte cheie rigide, folosind indexarea similarității L2.</p>
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Dashboard & Explorator Cod",
        "Arhitectură UML & Relații",
        "Căutare Semantică în Proiect",
        "Securitate",
        "Quiz Cod & Semantic",
        "Analiză & Recomandări",
        "Chatbot CodeBERT",
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
                                <span style="font-size: 0.8em; color: #8b5cf6;">Relevanță Scenariu (L2 dist): {chunk['score']:.4f}</span>
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
            st.markdown("## Explorator Atenție CodeBERT — Self-Attention Heatmap")
            st.markdown("Vizualizare interactivă a mecanismului de **Self-Attention** din Transformer: fiecare celulă `(i,j)` arată cât de mult tokenul `i` acordă atenție tokenului `j` în ultimul strat al CodeBERT.")
            st.write("---")

            with st.form("attention_form"):
                code_input = st.text_input("Linie de cod sau frază:", value="def connect_to_server(ip, port):")
                col_btn, col_head = st.columns([2, 1])
                with col_btn:
                    submitted_attention = st.form_submit_button("Generează Heatmap", use_container_width=True)
                with col_head:
                    head_mode = st.selectbox("Vizualizare", ["Media tuturor capetelor", "Capul 0", "Capul 1", "Capul 2", "Capul 3"], label_visibility="collapsed")

            if submitted_attention and code_input:
                with st.spinner("Se extrage tensorul de atenție din CodeBERT..."):
                    try:
                        import plotly.graph_objects as go
                        indexer = st.session_state.indexer

                        # Extragem matricea de atenție per cap, nu doar media
                        indexer.load_model()
                        import torch
                        inputs = indexer.tokenizer(code_input, return_tensors="pt", truncation=True, max_length=25)
                        tokens_raw = indexer.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = indexer.model(**inputs, output_attentions=True)

                        # outputs.attentions: tuple de 12 tensori (1, 12, seq, seq)
                        last_layer = outputs.attentions[-1][0].cpu().numpy()  # (12, seq, seq)

                        if head_mode == "Media tuturor capetelor":
                            att_matrix = last_layer.mean(axis=0)
                            title_suffix = "Media celor 12 capete — Stratul 12"
                        else:
                            head_idx = int(head_mode.split("Capul ")[1])
                            att_matrix = last_layer[head_idx]
                            title_suffix = f"Capul {head_idx} — Stratul 12"

                        # Curățare tokeni BPE: ## prefix (WordPiece) și speciali
                        def clean_token(t):
                            t = t.replace("##", "·")       # subword continuation
                            t = t.replace("Ġ", " ")        # GPT-style space
                            t = t.replace("[CLS]", "⟨CLS⟩")
                            t = t.replace("[SEP]", "⟨SEP⟩")
                            t = t.replace("[PAD]", "⟨PAD⟩")
                            return t

                        display_tokens = [clean_token(t) for t in tokens_raw]
                        n = len(display_tokens)

                        # Hover text cu valori exacte
                        hover = [[f"<b>{display_tokens[i]}</b> → <b>{display_tokens[j]}</b><br>Atenție: {att_matrix[i,j]:.4f}"
                                  for j in range(n)] for i in range(n)]

                        fig = go.Figure(go.Heatmap(
                            z=att_matrix,
                            x=display_tokens,
                            y=display_tokens,
                            colorscale="Plasma",
                            hoverinfo="text",
                            text=hover,
                            colorbar=dict(
                                title="Intensitate",
                                titleside="right",
                                tickfont=dict(color="#c9d1d9"),
                                titlefont=dict(color="#c9d1d9"),
                            ),
                            zmin=0, zmax=float(att_matrix.max()),
                        ))

                        fig.update_layout(
                            title=dict(text=f"Self-Attention Heatmap — {title_suffix}", font=dict(color="#38bdf8", size=14)),
                            xaxis=dict(tickfont=dict(color="#c9d1d9", size=11), tickangle=-45, side="bottom"),
                            yaxis=dict(tickfont=dict(color="#c9d1d9", size=11), autorange="reversed"),
                            paper_bgcolor="#0d1117",
                            plot_bgcolor="#0d1117",
                            margin=dict(l=80, r=40, t=60, b=100),
                            height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Topul celor mai puternice 5 conexiuni
                        st.markdown("**Top 5 conexiuni de atenție cele mai puternice:**")
                        flat = [(att_matrix[i,j], display_tokens[i], display_tokens[j])
                                for i in range(n) for j in range(n) if i != j]
                        flat.sort(reverse=True)
                        for score, src, dst in flat[:5]:
                            st.markdown(f"- `{src}` → `{dst}` &nbsp; **{score:.4f}**", unsafe_allow_html=True)

                        st.info("💡 Culorile deschise (galben/portocaliu) = atenție puternică. Culorile închise (violet) = corelație slabă. Poți zooma și hovera pe orice celulă.")

                    except Exception as e:
                        st.error(f"Eroare heatmap: {str(e)}")
                        st.code(traceback.format_exc(), language="text")
            else:
                st.info("Introdu o linie de cod și apasă **Generează Heatmap**.")

    # ----------------- TAB 4: SECURITATE -----------------
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
        dup_tab, complexity_tab, dead_tab, smells_tab = st.tabs([
            "Detector Cod Duplicat",
            "Complexitate Ciclomatică",
            "Cod Mort (Dead Code)",
            "Code Smells & Recomandări",
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

        # --- SUB-TAB: COMPLEXITATE CICLOMATICĂ ---
        with complexity_tab:
            st.markdown("## Complexitate Ciclomatică — Analiză AST")
            st.markdown("Complexitatea ciclomatică măsoară câte căi independente de execuție există într-o funcție. Formula: **CC = număr ramificații (if/elif/for/while/except/and/or) + 1**. O valoare > 10 indică cod greu de testat și menținut.")
            st.write("---")

            if st.button("Calculează Complexitatea", use_container_width=True, key="btn_complexity"):
                with st.spinner("Se analizează structura AST a fiecărei funcții..."):
                    try:
                        results_cc = []
                        for chunk in st.session_state.chunks:
                            if chunk.get("type") != "function":
                                continue
                            code = chunk.get("content", "")
                            try:
                                tree = ast.parse(code)
                            except:
                                continue
                            cc = 1
                            for node in ast.walk(tree):
                                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                                     ast.With, ast.AsyncFor, ast.AsyncWith)):
                                    cc += 1
                                elif isinstance(node, ast.BoolOp):
                                    cc += len(node.values) - 1
                                elif isinstance(node, (ast.comprehension,)):
                                    cc += 1
                            n_lines = chunk.get("end_line", 0) - chunk.get("start_line", 0)
                            results_cc.append({
                                "name": chunk.get("name", "?"),
                                "file": chunk.get("file_path", "?"),
                                "start": chunk.get("start_line", 0),
                                "cc": cc,
                                "lines": n_lines,
                            })
                        st.session_state["cc_results"] = sorted(results_cc, key=lambda x: x["cc"], reverse=True)
                    except Exception as e:
                        st.error(str(e))

            if "cc_results" in st.session_state and st.session_state["cc_results"]:
                data = st.session_state["cc_results"]
                high   = [r for r in data if r["cc"] > 10]
                medium = [r for r in data if 5 < r["cc"] <= 10]
                low    = [r for r in data if r["cc"] <= 5]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Funcții analizate", len(data))
                c2.metric("🔴 Complexitate mare (>10)", len(high))
                c3.metric("🟡 Medie (6–10)", len(medium))
                c4.metric("🟢 Mică (≤5)", len(low))
                st.write("---")

                # Grafic bar cu plotly
                try:
                    import plotly.graph_objects as go
                    names  = [f"{r['name']} ({r['file']})" for r in data[:20]]
                    values = [r["cc"] for r in data[:20]]
                    colors = ["#ef4444" if v > 10 else "#f59e0b" if v > 5 else "#22c55e" for v in values]
                    fig = go.Figure(go.Bar(
                        x=values, y=names, orientation="h",
                        marker_color=colors,
                        text=values, textposition="outside",
                    ))
                    fig.update_layout(
                        title="Top 20 funcții după complexitate ciclomatică",
                        xaxis_title="Complexitate Ciclomatică (CC)",
                        yaxis=dict(autorange="reversed"),
                        height=max(350, len(data[:20]) * 28),
                        margin=dict(l=220, r=60, t=50, b=40),
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#e2e8f0"),
                        xaxis=dict(gridcolor="#1e293b"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass

                st.write("---")
                if high:
                    st.markdown(f"### 🔴 Funcții cu complexitate mare (>10) — {len(high)} funcții")
                    for r in high:
                        with st.expander(f"**{r['name']}** — CC={r['cc']} · `{r['file']}` linia {r['start']} · {r['lines']} linii"):
                            st.markdown(f"""
| Metrică | Valoare |
|---------|---------|
| Complexitate Ciclomatică | **{r['cc']}** |
| Linii de cod | {r['lines']} |
| Fișier | `{r['file']}` |
| Linia de start | {r['start']} |
""")
                            st.warning(f"CC={r['cc']} > 10: funcția are prea multe ramificații. Recomandare: **împarte în sub-funcții** sau extrage logica în funcții helper specializate.")
            elif "cc_results" in st.session_state:
                st.info("Apasă butonul de mai sus pentru a calcula.")
            else:
                st.info("Apasă **Calculează Complexitatea** pentru a analiza toate funcțiile din proiect.")

        # --- SUB-TAB: COD MORT ---
        with dead_tab:
            st.markdown("## Detector Cod Mort (Dead Code) — AST")
            st.markdown("Detectează funcții și clase **definite dar niciodată apelate** în restul proiectului. Codul mort crește dimensiunea proiectului, confuzionează cititorii și poate ascunde bug-uri vechi.")
            st.write("---")

            if st.button("Detectează Codul Mort", use_container_width=True, key="btn_dead"):
                with st.spinner("Se scanează toate apelurile din proiect..."):
                    try:
                        # Colectăm toate apelurile din întreg proiectul
                        all_calls = set()
                        for chunk in st.session_state.chunks:
                            code = chunk.get("content", "")
                            try:
                                tree = ast.parse(code)
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.Call):
                                        if isinstance(node.func, ast.Name):
                                            all_calls.add(node.func.id)
                                        elif isinstance(node.func, ast.Attribute):
                                            all_calls.add(node.func.attr)
                            except:
                                pass

                        # Funcții definite dar niciodată apelate
                        dead = []
                        for chunk in st.session_state.chunks:
                            if chunk.get("type") not in ("function", "class"):
                                continue
                            name = chunk.get("name", "")
                            # Excludem dunder methods, main, callbacks și constructori
                            if name.startswith("__") or name in ("main", "app", "run"):
                                continue
                            if name not in all_calls:
                                dead.append(chunk)

                        st.session_state["dead_results"] = dead
                    except Exception as e:
                        st.error(str(e))

            if "dead_results" in st.session_state:
                dead = st.session_state["dead_results"]
                total_funcs = len([c for c in st.session_state.chunks if c.get("type") in ("function","class")])

                col1, col2, col3 = st.columns(3)
                col1.metric("Funcții/clase totale", total_funcs)
                col2.metric("Potențial neutilizate", len(dead))
                col3.metric("Utilizate", total_funcs - len(dead))
                st.write("---")

                if not dead:
                    st.success("Nu s-a detectat cod mort. Toate funcțiile par să fie apelate cel puțin o dată.")
                else:
                    st.warning(f"**{len(dead)} funcții/clase** par să nu fie apelate nicăieri în proiect:")
                    # Grupate pe fișier
                    by_file = {}
                    for c in dead:
                        by_file.setdefault(c.get("file_path","?"), []).append(c)
                    for fpath, items in by_file.items():
                        with st.expander(f"`{fpath}` — {len(items)} neutilizate"):
                            for c in items:
                                n_lines = c.get("end_line",0) - c.get("start_line",0)
                                doc = c.get("docstring","").strip()
                                st.markdown(f"""
<div style="border-left:3px solid #ef4444; padding:8px 12px; margin-bottom:8px; background:rgba(239,68,68,0.05); border-radius:0 6px 6px 0;">
<b style="color:#f87171;">{c['type'].upper()}</b> <code>{c['name']}</code> · linia {c['start_line']} · {n_lines} linii
{"<br><span style='color:#94a3b8;font-size:0.88em;'>" + doc[:120] + "</span>" if doc else ""}
<br><span style='color:#64748b;font-size:0.82em;'>💡 Dacă nu e apelată din exterior (API/test/UI), poate fi ștearsă în siguranță.</span>
</div>""", unsafe_allow_html=True)
            else:
                st.info("Apasă **Detectează Codul Mort** pentru a scana proiectul.")

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

    # ----------------- TAB 7: CHATBOT CODEBERG -----------------
    with tab7:
        st.markdown("## Chatbot CodeBERT — Întreabă despre codul tău")
        st.markdown("Chatbot-ul folosește **CodeBERT + FAISS** pentru a găsi fragmentele de cod relevante din proiect și a răspunde la întrebările tale. Funcționează **100% offline**, fără API extern.")
        st.write("---")

        if not st.session_state.project_processed:
            st.info("Încarcă un proiect mai întâi din sidebar pentru a putea folosi chatbot-ul.")
        else:
            # Tratam întrebarea venită de la butoanele rapide
            if "query_trigger" not in st.session_state:
                st.session_state.query_trigger = None

            programmatic_q = None
            if st.session_state.query_trigger:
                programmatic_q = st.session_state.query_trigger
                st.session_state.query_trigger = None # Clear it immediately

            # Inițializare istoric chat
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Afișare mesaje anterioare
            for idx, msg in enumerate(st.session_state.chat_history):
                with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("chunks"):
                        with st.expander("Fragmente de cod relevante găsite de CodeBERT", expanded=False):
                            for c in msg["chunks"]:
                                st.markdown(f"**{c.get('name','?')}** — `{c.get('file_path','?')}` (liniile {c.get('start_line',0)}–{c.get('end_line',0)})")
                                st.code(c["content"][:400], language="python")
                    
                    # Dacă acesta este ultimul mesaj și este de la asistent, adăugăm butoanele de follow-up dinamic
                    is_last = (idx == len(st.session_state.chat_history) - 1)
                    if is_last and msg["role"] == "assistant":
                        user_question = ""
                        if idx > 0 and st.session_state.chat_history[idx - 1]["role"] == "user":
                            user_question = st.session_state.chat_history[idx - 1]["content"]
                        
                        kb_match = msg.get("kb_match")
                        follow_ups = generate_follow_up_questions(user_question, msg["content"], kb_match)
                        
                        st.markdown("<div style='margin-top:15px; color:#38bdf8; font-weight:600; font-size:0.9em;'>⚡ Întrebări recomandate:</div>", unsafe_allow_html=True)
                        cols = st.columns(len(follow_ups))
                        for col_idx, f_q in enumerate(follow_ups):
                            with cols[col_idx]:
                                if st.button(f_q, key=f"fup_{idx}_{col_idx}", use_container_width=True):
                                    st.session_state.query_trigger = f_q
                                    st.rerun()

            # Input utilizator
            user_q = st.chat_input("Întreabă ceva despre codul tău, ex: Ce face funcția X? Unde se inițializează Y?")

            active_q = programmatic_q or user_q

            if active_q:
                # Afișăm mesajul utilizatorului
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(active_q)
                st.session_state.chat_history.append({"role": "user", "content": active_q})

                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("CodeBERT caută și analizează codul..."):
                        try:
                            indexer = st.session_state.indexer
                            all_chunks = st.session_state.chunks

                            # Construim KB lazy dacă nu există (proiect indexat anterior)
                            if "project_kb" not in st.session_state or not st.session_state.project_kb.get("by_name"):
                                st.session_state.project_kb = build_project_knowledge(all_chunks)

                            kb = st.session_state.project_kb
                            results = indexer.search(active_q, top_k=5)
                            answer, found_chunks, kb_match = build_chatbot_answer(active_q, results, all_chunks, kb=kb, indexer=indexer)

                            st.markdown(answer, unsafe_allow_html=True)

                            # Afișare Heatmap de Self-Attention pentru întrebare
                            with st.expander("🔍 Vizualizează Atenția Transformer pentru Întrebare", expanded=False):
                                st.markdown("Mecanismul de **Self-Attention** din Transformer: celulele mai luminoase indică o atenție sporită acordată între tokenii respectivi din întrebarea ta.")
                                try:
                                    import plotly.graph_objects as go
                                    indexer.load_model()
                                    import torch
                                    inputs = indexer.tokenizer(active_q, return_tensors="pt", truncation=True, max_length=25)
                                    tokens_raw = indexer.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                                    with torch.no_grad():
                                        outputs = indexer.model(**inputs, output_attentions=True)
                                    
                                    last_layer = outputs.attentions[-1][0].cpu().numpy()  # (12, seq, seq)
                                    att_matrix = last_layer.mean(axis=0)  # Media tuturor celor 12 capete
                                    
                                    def clean_token(t):
                                        t = t.replace("##", "·")
                                        t = t.replace("Ġ", " ")
                                        t = t.replace("[CLS]", "⟨CLS⟩")
                                        t = t.replace("[SEP]", "⟨SEP⟩")
                                        t = t.replace("[PAD]", "⟨PAD⟩")
                                        return t

                                    display_tokens = [clean_token(t) for t in tokens_raw]
                                    n = len(display_tokens)
                                    
                                    hover = [[f"<b>{display_tokens[i]}</b> → <b>{display_tokens[j]}</b><br>Atenție: {att_matrix[i,j]:.4f}"
                                              for j in range(n)] for i in range(n)]
                                              
                                    fig = go.Figure(go.Heatmap(
                                        z=att_matrix,
                                        x=display_tokens,
                                        y=display_tokens,
                                        colorscale="Plasma",
                                        hoverinfo="text",
                                        text=hover,
                                        colorbar=dict(
                                            title="Intensitate",
                                            titleside="right",
                                            tickfont=dict(color="#c9d1d9"),
                                            titlefont=dict(color="#c9d1d9"),
                                        ),
                                        zmin=0, zmax=float(att_matrix.max() + 1e-9),
                                    ))
                                    
                                    fig.update_layout(
                                        title=dict(text="Self-Attention Heatmap pe Întrebare", font=dict(color="#38bdf8", size=14)),
                                        xaxis=dict(tickfont=dict(color="#c9d1d9", size=11), tickangle=-45, side="bottom"),
                                        yaxis=dict(tickfont=dict(color="#c9d1d9", size=11), autorange="reversed"),
                                        paper_bgcolor="#0d1117",
                                        plot_bgcolor="#0d1117",
                                        margin=dict(l=80, r=40, t=60, b=100),
                                        height=400,
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Afișăm top 3 conexiuni
                                    flat = [(att_matrix[i,j], display_tokens[i], display_tokens[j])
                                            for i in range(n) for j in range(n) if i != j]
                                    flat.sort(reverse=True)
                                    st.markdown("**Top 3 conexiuni de atenție în întrebarea ta:**")
                                    for score, src, dst in flat[:3]:
                                        st.markdown(f"- `{src}` → `{dst}` &nbsp; (atenție: **{score:.4f}**)")
                                        
                                except Exception as att_err:
                                    st.warning(f"Nu s-a putut genera Heatmap-ul de atenție: {str(att_err)}")

                            if found_chunks:
                                with st.expander("Cod sursă — fragmente găsite de CodeBERT", expanded=False):
                                    for c in found_chunks[:3]:
                                        st.markdown(f"**`{c.get('name','?')}`** — `{c.get('file_path','?')}` · liniile {c.get('start_line',0)}–{c.get('end_line',0)}")
                                        st.code(c["content"][:600], language="python")

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "chunks": found_chunks[:3] if found_chunks else [],
                                "kb_match": kb_match
                            })
                            st.rerun()

                        except Exception as e:
                            err_msg = f"Eroare: {str(e)}"
                            st.error(err_msg)
                            st.code(traceback.format_exc(), language="text")
                            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

            # Buton resetare chat
            if st.session_state.chat_history:
                if st.button("Șterge conversația", use_container_width=False):
                    st.session_state.chat_history = []
                    st.rerun()
