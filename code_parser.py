import os
import zipfile
import shutil
import ast
import re
from pathlib import Path

def unzip_project(zip_file_bytes, extract_to_dir):
    """
    Dezarhivează un fișier ZIP primit ca bytes într-un director specificat.
    Șterge directorul dacă există deja pentru a evita suprapunerea proiectelor.
    """
    extract_path = Path(extract_to_dir)
    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    # Scriem bytes într-un fișier temporar zip și îl dezarhivăm
    temp_zip = extract_path / "temp_project.zip"
    with open(temp_zip, "wb") as f:
        f.write(zip_file_bytes)
        
    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        
    # Ștergem arhiva temporară
    os.remove(temp_zip)
    
    # Dacă dezarhivarea a creat un singur subdirector care conține totul,
    # îl aducem la rădăcină pentru o mai bună organizare.
    subdirs = [d for d in extract_path.iterdir() if d.is_dir()]
    files = [f for f in extract_path.iterdir() if f.is_file()]
    if len(subdirs) == 1 and len(files) == 0:
        single_subdir = subdirs[0]
        for item in single_subdir.iterdir():
            shutil.move(str(item), str(extract_path))
        single_subdir.rmdir()

# Seturi de directoare și fișiere de ignorat pentru a evita "zgomotul" în embeddings
IGNORED_DIRS = {
    '.git', '__pycache__', 'node_modules', 'venv', '.venv', 'env', 
    'dist', 'build', '.idea', '.vscode', '.gemini', 'eggs', 
    '.mypy_cache', '.pytest_cache', '.sass-cache', 'target', 'out'
}

IGNORED_FILES = {
    '.DS_Store', 'Thumbs.db', '.gitignore', 'package-lock.json', 
    'yarn.lock', 'pnpm-lock.yaml', 'poetry.lock', 'pip-log.txt'
}

ALLOWED_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json', 
    'yaml', '.yml', '.md', '.java', '.cpp', '.h', '.c', '.cs', 
    '.go', '.rs', '.kt', '.php', '.rb', '.sh', '.sql', '.xml'
}

def is_allowed_file(file_path):
    """
    Verifică dacă fișierul are o extensie acceptată și nu este într-un folder ignorat.
    """
    path = Path(file_path)
    
    # Verificăm dacă vreunul dintre părinți este în lista de directoare ignorate
    for part in path.parts:
        if part in IGNORED_DIRS:
            return False
            
    if path.name in IGNORED_FILES:
        return False
        
    return path.suffix.lower() in ALLOWED_EXTENSIONS

def scan_project_files(root_dir):
    """
    Scanează recursiv directorul proiectului și returnează o listă de căi de fișiere valide.
    """
    valid_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for file in files:
            full_path = Path(root) / file
            if is_allowed_file(full_path):
                valid_files.append(full_path)
    return valid_files

def build_file_tree(root_dir, current_dir=None):
    """
    Construiește o structură arborescentă recursivă (dicționar) a fișierelor valide.
    """
    if current_dir is None:
        current_dir = Path(root_dir)
    
    tree = {"name": current_dir.name or "Root", "type": "directory", "children": []}
    
    try:
        items = sorted(list(current_dir.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        for item in items:
            if item.is_dir():
                if item.name in IGNORED_DIRS:
                    continue
                subtree = build_file_tree(root_dir, item)
                if subtree["children"]:
                    tree["children"].append(subtree)
            else:
                if is_allowed_file(item):
                    rel_path = str(item.relative_to(root_dir))
                    tree["children"].append({
                        "name": item.name,
                        "type": "file",
                        "path": rel_path,
                        "size": item.stat().st_size
                    })
    except Exception as e:
        pass
        
    return tree

def chunk_python_file(file_path, code_content, rel_path):
    """
    Folosește AST pentru a descompune fișierul Python în clase și funcții independente.
    Păstrează metadate bogate despre semnături, docstrings și linii.
    """
    chunks = []
    lines = code_content.splitlines()
    
    try:
        tree = ast.parse(code_content)
    except SyntaxError:
        return chunk_text_fallback(code_content, rel_path)
        
    module_level_nodes = []
    
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Extragem funcția globală
            start = node.lineno
            end = getattr(node, 'end_lineno', len(lines))
            func_code = "\n".join(lines[start-1:end])
            docstring = ast.get_docstring(node) or "Fără descriere disponibilă în docstring."
            args = [arg.arg for arg in node.args.args]
            
            chunks.append({
                "file_path": rel_path,
                "type": "function",
                "name": node.name,
                "start_line": start,
                "end_line": end,
                "docstring": docstring,
                "args": args,
                "summary": f"Funcția `{node.name}({', '.join(args)})` definită în `{rel_path}` (liniile {start}-{end}).",
                "content": f"# File: {rel_path} | Function: {node.name} (Lines {start}-{end})\n{func_code}"
            })
        elif isinstance(node, ast.ClassDef):
            # Extragem clasa întreagă
            start = node.lineno
            end = getattr(node, 'end_lineno', len(lines))
            class_code = "\n".join(lines[start-1:end])
            docstring = ast.get_docstring(node) or "Fără descriere disponibilă în docstring."
            parents = [base.id for base in node.bases if isinstance(base, ast.Name)]
            
            # Extragem metodele clasei
            methods = []
            for sub_node in node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sub_args = [arg.arg for arg in sub_node.args.args]
                    methods.append(f"{sub_node.name}({', '.join(sub_args)})")
            
            chunks.append({
                "file_path": rel_path,
                "type": "class",
                "name": node.name,
                "start_line": start,
                "end_line": end,
                "docstring": docstring,
                "parents": parents,
                "methods": methods,
                "summary": f"Clasa `{node.name}` (moștenește {', '.join(parents) if parents else 'obiectul de bază'}) definită în `{rel_path}` (liniile {start}-{end}) cu metodele: {', '.join(methods)}.",
                "content": f"# File: {rel_path} | Class: {node.name} (Lines {start}-{end})\n{class_code}"
            })
        else:
            module_level_nodes.append(node)
            
    # Grupăm codul rămas de la nivelul modulului
    if module_level_nodes:
        chunk_size = 80
        for i in range(0, len(lines), chunk_size):
            end_idx = min(i + chunk_size, len(lines))
            sub_lines = lines[i:end_idx]
            sub_code = "\n".join(sub_lines)
            if sub_code.strip():
                chunks.append({
                    "file_path": rel_path,
                    "type": "module_level",
                    "name": "globals",
                    "start_line": i + 1,
                    "end_line": end_idx,
                    "docstring": "Cod la nivel de modul (importuri, variabile globale sau execuție principală).",
                    "summary": f"Bloc general de cod la nivel de modul în `{rel_path}` (liniile {i+1}-{end_idx}).",
                    "content": f"# File: {rel_path} | Global Code (Lines {i+1}-{end_idx})\n{sub_code}"
                })
                
    return chunks

def chunk_generic_code(code_content, rel_path, suffix):
    """
    Chunker inteligent pentru limbaje non-Python (C++, Java, JS, TS).
    """
    lines = code_content.splitlines()
    if len(lines) <= 40:
        return [{
            "file_path": rel_path,
            "type": "full_file",
            "name": "all",
            "start_line": 1,
            "end_line": len(lines),
            "docstring": f"Fișier complet de tip {suffix}.",
            "summary": f"Fișierul complet `{rel_path}` ({len(lines)} linii).",
            "content": f"# File: {rel_path} | Full file\n{code_content}"
        }]
        
    chunks = []
    patterns = [
        r'(class\s+\w+)', 
        r'(function\s+\w+)', 
        r'(\w+\s+\w+\(.*?\)\s*\{)', 
        r'(const\s+\w+\s*=\s*\(.*?\)\s*=>)' 
    ]
    combined_pattern = re.compile("|".join(patterns))
    
    current_chunk = []
    chunk_start = 1
    
    for idx, line in enumerate(lines):
        line_num = idx + 1
        if combined_pattern.search(line) and len(current_chunk) >= 20:
            chunk_code = "\n".join(current_chunk)
            chunks.append({
                "file_path": rel_path,
                "type": "code_block",
                "name": f"block_{chunk_start}",
                "start_line": chunk_start,
                "end_line": line_num - 1,
                "docstring": "Bloc de cod extras structural din semnături.",
                "summary": f"Bloc structural în `{rel_path}` (liniile {chunk_start}-{line_num-1}).",
                "content": f"// File: {rel_path} | Block (Lines {chunk_start}-{line_num-1})\n{chunk_code}"
            })
            current_chunk = []
            chunk_start = line_num
            
        current_chunk.append(line)
        
        if len(current_chunk) >= 80:
            chunk_code = "\n".join(current_chunk)
            chunks.append({
                "file_path": rel_path,
                "type": "code_block",
                "name": f"block_{chunk_start}",
                "start_line": chunk_start,
                "end_line": line_num,
                "docstring": "Secțiune de cod din fișier.",
                "summary": f"Segment de cod lung în `{rel_path}` (liniile {chunk_start}-{line_num}).",
                "content": f"// File: {rel_path} | Block (Lines {chunk_start}-{line_num})\n{chunk_code}"
            })
            current_chunk = []
            chunk_start = line_num + 1
            
    if current_chunk:
        chunk_code = "\n".join(current_chunk)
        chunks.append({
            "file_path": rel_path,
            "type": "code_block",
            "name": f"block_{chunk_start}",
            "start_line": chunk_start,
            "end_line": len(lines),
            "docstring": "Ultimul bloc de cod al fișierului.",
            "summary": f"Partea finală a fișierului `{rel_path}` (liniile {chunk_start}-{len(lines)}).",
            "content": f"// File: {rel_path} | Block (Lines {chunk_start}-{len(lines)})\n{chunk_code}"
        })
        
    return chunks

def chunk_text_fallback(content, rel_path):
    """
    Fallback pentru fișiere text simple, Markdown, YAML sau cod care a eșuat la parsare.
    """
    lines = content.splitlines()
    chunks = []
    chunk_size = 60
    overlap = 10
    
    i = 0
    while i < len(lines):
        end_idx = min(i + chunk_size, len(lines))
        chunk_lines = lines[i:end_idx]
        chunk_code = "\n".join(chunk_lines)
        if chunk_code.strip():
            chunks.append({
                "file_path": rel_path,
                "type": "text_block",
                "name": f"text_{i+1}",
                "start_line": i + 1,
                "end_line": end_idx,
                "docstring": "Fragment de text din document.",
                "summary": f"Fragment text în `{rel_path}` (liniile {i+1}-{end_idx}).",
                "content": f"# File: {rel_path} | Text (Lines {i+1}-{end_idx})\n{chunk_code}"
            })
        if end_idx == len(lines):
            break
        i += (chunk_size - overlap)
        
    return chunks

def parse_and_chunk_file(file_path, root_dir):
    """
    Primește un fișier fizic, decide tipul lui și returnează o listă de chunk-uri indexabile.
    """
    path = Path(file_path)
    rel_path = str(path.relative_to(root_dir))
    suffix = path.suffix.lower()
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return []
        
    if not content.strip():
        return []
        
    if suffix == '.py':
        return chunk_python_file(file_path, content, rel_path)
    elif suffix in {'.js', '.jsx', '.ts', '.tsx', '.cpp', '.h', '.c', '.cs', '.java', '.go', '.rs', '.kt'}:
        return chunk_generic_code(content, rel_path, suffix)
    else:
        return chunk_text_fallback(content, rel_path)

# ==================== DETERMINISTIC MERMAID GENERATORS (AST BASED) ====================

def generate_uml_class_diagram(files_list, root_dir):
    """
    Analizează AST-ul tuturor fișierelor Python din listă și generează deterministic
    o diagramă de clase Mermaid.js complet validă și extrem de detaliată.
    """
    classes = []
    
    for file_path in files_list:
        if file_path.suffix.lower() != '.py':
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    parents = [base.id for base in node.bases if isinstance(base, ast.Name)]
                    methods = []
                    fields = []
                    
                    # Inspectăm corpul clasei pentru metode și atribute
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [arg.arg for arg in item.args.args if arg.arg != 'self']
                            methods.append(f"{item.name}({', '.join(args)})")
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    fields.append(target.id)
                                    
                    classes.append({
                        "name": node.name,
                        "parents": parents,
                        "methods": methods[:8],  # Limită pentru lizibilitate grafică
                        "fields": fields[:8]
                    })
        except:
            pass

    if not classes:
        return "classDiagram\n    class FărăClase {\n        +descriere: Nu s-au detectat clase Python\n    }"

    mermaid_lines = ["classDiagram"]
    
    # Adăugăm definițiile claselor
    for cls in classes:
        cls_name = cls["name"]
        mermaid_lines.append(f"    class {cls_name} {{")
        for field in cls["fields"]:
            mermaid_lines.append(f"        +{field}")
        for method in cls["methods"]:
            mermaid_lines.append(f"        +{method}")
        mermaid_lines.append("    }")
        
        # Adăugăm legăturile de moștenire (Parent <|-- Child)
        for parent in cls["parents"]:
            # Relație de moștenire în Mermaid.js
            mermaid_lines.append(f"    {parent} <|-- {cls_name}")
            
    return "\n".join(mermaid_lines)

def generate_dependency_diagram(files_list, root_dir):
    """
    Analizează importurile (`import` și `from X import Y`) din fișierele Python și Javascript
    și desenează diagrama de dependențe / apeluri de module a proiectului.
    """
    dependencies = []
    file_basenames = {f.stem: str(f.relative_to(root_dir)) for f in files_list}
    
    for file_path in files_list:
        rel_path = str(file_path.relative_to(root_dir))
        suffix = file_path.suffix.lower()
        
        if suffix != '.py':
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                imported_module = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name.split('.')[0]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_module = node.module.split('.')[0]
                        
                if imported_module and imported_module in file_basenames:
                    target_file = file_basenames[imported_module]
                    dependencies.append((rel_path, target_file))
        except:
            pass

    if not dependencies:
        # Dacă nu sunt dependențe detectate prin AST, mapăm fișierele pur ierarhic la rădăcină
        mermaid_lines = ["graph TD"]
        for f in list(file_basenames.values())[:10]:
            name_clean = f.replace("/", "_").replace(".", "_").replace("-", "_")
            mermaid_lines.append(f"    Root --> {name_clean}[\"{f}\"]")
        return "\n".join(mermaid_lines)

    # Convertim dependințele în grafic Mermaid
    mermaid_lines = ["graph TD"]
    visited_edges = set()
    
    for src, dest in dependencies:
        if src == dest:
            continue
        edge = (src, dest)
        if edge not in visited_edges:
            visited_edges.add(edge)
            src_clean = src.replace("/", "_").replace(".", "_").replace("-", "_")
            dest_clean = dest.replace("/", "_").replace(".", "_").replace("-", "_")
            mermaid_lines.append(f"    {src_clean}[\"{src}\"] --> {dest_clean}[\"{dest}\"]")
            
    return "\n".join(mermaid_lines)
