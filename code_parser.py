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
 Verifică dacă fișierul nu este într-un folder ignorat și nu este un fișier binar cunoscut.
 """
 path = Path(file_path)
 
 # Verificăm dacă vreunul dintre părinți este în lista de directoare ignorate
 for part in path.parts:
 if part in IGNORED_DIRS:
 return False
 
 if path.name in IGNORED_FILES:
 return False
 
 # Excludem formatele binare cunoscute pentru a permite orice text/cod
 BINARY_EXTENSIONS = {
 '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz',
 '.mp3', '.mp4', '.avi', '.mov', '.exe', '.dll', '.so', '.bin', '.pkl',
 '.db', '.sqlite', '.class', '.jar', '.woff', '.woff2', '.ttf', '.eot',
 '.7z', '.rar'
 }
 
 suffix = path.suffix.lower()
 if suffix in BINARY_EXTENSIONS:
 return False
 
 return True

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

def _safe_id(name):
 """Transformă orice string într-un identificator valid pentru Mermaid classDiagram."""
 import re
 return re.sub(r'[^a-zA-Z0-9_]', '_', name).strip('_') or 'Node'

def _safe_label(text):
 """Elimină orice caractere care ar putea sparge sintaxa Mermaid din labels."""
 return text.replace('"', '').replace("'", '').replace('{', '').replace('}', '').replace(':', ' ').replace('<', '').replace('>', '').replace('(', '_').replace(')', '').replace(',', ' ').replace('=', '_')

def generate_uml_class_diagram(files_list, root_dir):
 classes = []

 for file_path in files_list:
 suffix = file_path.suffix.lower()
 if suffix != '.py':
 # Extracție universală pentru non-Python (Java, C++, JS, TS, Rust, C#)
 try:
 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 # Căutăm clase în Java, C++, C#, JS, TS, Rust
 class_matches = re.finditer(r'(?:class|struct|interface|trait)\s+([a-zA-Z0-9_]+)(?:\s*(?:extends|:)\s*([a-zA-Z0-9_]+))?', content)
 for match in class_matches:
 c_name = match.group(1)
 parent = match.group(2)
 parents = [parent] if parent else []
 
 # Extragem contextul din jurul clasei pentru a căuta metode/câmpuri brute
 class_start = match.start()
 context = content[class_start:class_start + 1500]
 
 # Căutăm metode brute: e.g. nume(argumente)
 methods = []
 methods_found = re.findall(r'(?:fn|public|private|protected|void|int|string|async|function)?\s+([a-zA-Z0-9_]+)\s*\(', context)
 for m in methods_found:
 if m not in {c_name, 'if', 'for', 'while', 'switch', 'catch', 'init', 'class', 'struct', 'fn', 'void'}:
 methods.append(f"{m}()")
 
 # Căutăm câmpuri brute: e.g. int nume; sau let nume =
 fields = []
 fields_found = re.findall(r'(?:private|public|protected|let|const|var)?\s*(?:int|string|double|float|bool|boolean)?\s+([a-zA-Z0-9_]+)\s*(?:;|=)', context)
 for fd in fields_found:
 if fd not in {c_name, 'if', 'for', 'while', 'return', 'class', 'struct', 'fn', 'void'} and len(fd) > 1:
 fields.append(fd)
 
 classes.append({
 "name": c_name,
 "safe_name": _safe_id(c_name),
 "parents": parents,
 "methods": list(dict.fromkeys(methods))[:6],
 "fields": list(dict.fromkeys(fields))[:5],
 })
 except:
 pass
 else:
 # Extracție precisă AST pentru Python
 try:
 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 tree = ast.parse(content)

 for node in ast.walk(tree):
 if isinstance(node, ast.ClassDef):
 parents = [base.id for base in node.bases if isinstance(base, ast.Name)]
 methods = []
 fields = []

 for item in node.body:
 if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
 methods.append(f"{item.name}()")
 elif isinstance(item, ast.Assign):
 for target in item.targets:
 if isinstance(target, ast.Name):
 fields.append(target.id)

 classes.append({
 "name": node.name,
 "safe_name": _safe_id(node.name),
 "parents": parents,
 "methods": methods[:6],
 "fields": fields[:5],
 })
 except:
 pass

 if not classes:
 return "classDiagram\n class NoClasses {\n +info String\n }"

 # Colectăm numele sigure cunoscute ca să validăm relațiile
 known = {c["name"]: c["safe_name"] for c in classes}

 mermaid_lines = ["classDiagram"]

 for cls in classes:
 sn = cls["safe_name"]
 if sn != cls["name"]:
 mermaid_lines.append(f' class {sn}["{cls["name"]}"] {{')
 else:
 mermaid_lines.append(f" class {sn} {{")
 for field in cls["fields"]:
 mermaid_lines.append(f" +{field}")
 for method in cls["methods"]:
 mermaid_lines.append(f" +{method}")
 mermaid_lines.append(" }")

 # Relații de moștenire — doar între clase cunoscute în proiect
 for cls in classes:
 for parent in cls["parents"]:
 if parent in known:
 mermaid_lines.append(f" {known[parent]} <|-- {cls['safe_name']}")

 return "\n".join(mermaid_lines)

def generate_dependency_diagram(files_list, root_dir):
 """
 Analizează importurile (`import`, `require`, `using`, `use`, `#include`) din toate fișierele
 și desenează diagrama de dependențe / apeluri de module a proiectului.
 """
 dependencies = []
 file_basenames = {f.stem: str(f.relative_to(root_dir)) for f in files_list}
 
 for file_path in files_list:
 rel_path = str(file_path.relative_to(root_dir))
 suffix = file_path.suffix.lower()
 
 if suffix != '.py':
 # Extracție importuri prin regex pentru limbaje non-Python
 try:
 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 # Căutăm cuvinte cheie de import (import, include, using, use, require)
 imports = re.findall(r'(?:import|using|use|#include|require)\s+["\'<]?([a-zA-Z0-9_\-\./\:]+)["\'>]?;?', content)
 for imp in imports:
 imp_base = imp.split('/')[-1].split('.')[-1].split(':')[-1].strip()
 if imp_base in file_basenames:
 target_file = file_basenames[imp_base]
 dependencies.append((rel_path, target_file))
 except:
 pass
 else:
 # Extracție precisă AST pentru Python
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
 mermaid_lines.append(f" Root --> {name_clean}[\"{f}\"]")
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
 mermaid_lines.append(f" {src_clean}[\"{src}\"] --> {dest_clean}[\"{dest}\"]")

 return "\n".join(mermaid_lines)


def _mid(name: str) -> str:
 """Sanitizează un string pentru a fi ID valid în Mermaid."""
 return re.sub(r'[^a-zA-Z0-9_]', '_', name).lstrip('_') or 'node'


def generate_sequence_diagram(files_list, root_dir):
 """
 Strategie:
 1. Construiește un registru global: nume_funcție -> modul (pentru orice fișier text din codebase)
 2. Caută apeluri cross-modul în corpul funcțiilor
 3. Fallback: dacă nu există cross-modul, arată apeluri intra-modul între funcții top-level
 4. Fallback final: arată participanții cu note despre metodele lor principale
 """
 if not files_list:
 return "sequenceDiagram\n participant Project\n Note over Project: Nu există fișiere în proiect"

 # Registru funcții: {func_name: module_stem}
 func_registry = {}
 module_funcs = {} # {module_stem: [func_name,...]}
 
 for fp in files_list:
 mod = fp.stem
 module_funcs[mod] = []
 suffix = fp.suffix.lower()
 
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 if suffix == '.py':
 tree = ast.parse(content)
 for node in ast.walk(tree):
 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
 func_registry[node.name] = mod
 module_funcs[mod].append(node.name)
 else:
 # Extracție prin regex pentru Java, C++, JS, TS, Go, Rust, C#
 defs = re.findall(r'(?:fn|public|private|protected|void|int|string|async|function)\s+([a-zA-Z0-9_]+)\s*\(', content)
 for d in defs:
 if d not in {'if', 'for', 'while', 'catch', 'switch', 'init', 'void'}:
 func_registry[d] = mod
 module_funcs[mod].append(d)
 except:
 pass

 # Caută apeluri cross-modul
 cross_calls = []
 visited = set()
 for fp in files_list:
 caller_mod = fp.stem
 suffix = fp.suffix.lower()
 
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 if suffix == '.py':
 tree = ast.parse(content)
 for func_node in ast.walk(tree):
 if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
 continue
 caller_func = func_node.name
 for call in ast.walk(func_node):
 if not isinstance(call, ast.Call):
 continue
 callee_name = None
 if isinstance(call.func, ast.Name):
 callee_name = call.func.id
 elif isinstance(call.func, ast.Attribute):
 callee_name = call.func.attr
 if callee_name and callee_name in func_registry:
 callee_mod = func_registry[callee_name]
 if callee_mod != caller_mod:
 edge = (caller_mod, callee_mod, callee_name)
 if edge not in visited:
 visited.add(edge)
 cross_calls.append(edge)
 else:
 # Căutăm apeluri de funcții înregistrate în fișierele non-Python
 for callee_name, callee_mod in func_registry.items():
 if callee_mod != caller_mod and f"{callee_name}(" in content:
 edge = (caller_mod, callee_mod, callee_name)
 if edge not in visited:
 visited.add(edge)
 cross_calls.append(edge)
 except:
 pass

 lines = ["sequenceDiagram"]

 if cross_calls:
 # Cazul ideal: există apeluri cross-modul
 seen_parts = list(dict.fromkeys(
 m for caller, callee, _ in cross_calls[:15] for m in [caller, callee]
 ))[:8]
 for p in seen_parts:
 lines.append(f" participant {p}")
 for caller, callee, method in cross_calls[:15]:
 lines.append(f" {caller}->>{callee}: {method}()")
 return "\n".join(lines)

 # Fallback: apeluri intra-modul (funcții din același fișier care se apelează)
 intra_calls = []
 visited2 = set()
 for fp in files_list:
 mod = fp.stem
 local_funcs = set(module_funcs.get(mod, []))
 suffix = fp.suffix.lower()
 
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 if suffix == '.py':
 tree = ast.parse(content)
 for func_node in ast.walk(tree):
 if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
 continue
 caller_f = func_node.name
 for call in ast.walk(func_node):
 if not isinstance(call, ast.Call):
 continue
 callee_name = None
 if isinstance(call.func, ast.Name):
 callee_name = call.func.id
 if callee_name and callee_name in local_funcs and callee_name != caller_f:
 edge = (f"{mod}.{caller_f}", f"{mod}.{callee_name}", callee_name)
 if edge not in visited2:
 visited2.add(edge)
 intra_calls.append(edge)
 else:
 # Pentru limbaje non-Python: căutăm apeluri de funcții locale în cadrul fișierului
 for caller_f in local_funcs:
 for callee_name in local_funcs:
 if callee_name != caller_f and f"{callee_name}(" in content:
 edge = (f"{mod}.{caller_f}", f"{mod}.{callee_name}", callee_name)
 if edge not in visited2:
 visited2.add(edge)
 intra_calls.append(edge)
 except:
 pass

 if intra_calls:
 seen_parts = list(dict.fromkeys(
 p for c, e, _ in intra_calls[:12] for p in [c, e]
 ))[:8]
 for p in seen_parts:
 safe = p.replace('.', '_')
 lines.append(f' participant {safe} as "{p}"')
 for caller, callee, method in intra_calls[:12]:
 lines.append(f" {caller.replace('.','_')}->>{callee.replace('.','_')}: {method}()")
 return "\n".join(lines)

 # Fallback final: arată modulele cu primele lor funcții
 mods = list(module_funcs.items())[:6]
 for mod, funcs in mods:
 lines.append(f" participant {mod}")
 if len(mods) >= 2:
 for i in range(min(len(mods) - 1, 5)):
 m1, f1 = mods[i]
 m2, _ = mods[i + 1]
 func = f1[0] if f1 else "call"
 lines.append(f" {m1}->>{m2}: {func}()")
 lines.append(f" {m2}-->>{m1}: return")
 else:
 mod, funcs = mods[0] if mods else ("App", [])
 lines.append(f" participant {mod}")
 lines.append(f" Note over {mod}: {', '.join(funcs[:4])}")

 return "\n".join(lines)


def generate_flowchart_diagram(files_list, root_dir):
 """
 Generează un call-graph complet: funcție → funcție.
 Include atât apeluri cross-modul cât și intra-modul.
 Nodurile sunt grupate pe modul prin stilizare.
 """
 if not files_list:
 return "flowchart TD\n A[Nu există fișiere în proiect]"

 # Registru global: func_name -> (module, full_label)
 func_registry = {} 
 for fp in files_list:
 mod = fp.stem
 suffix = fp.suffix.lower()
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 if suffix == '.py':
 tree = ast.parse(content)
 for node in ast.walk(tree):
 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
 label = f"{mod}.{node.name}"
 func_registry[node.name] = (mod, label)
 else:
 # Extracție prin regex pentru limbaje non-Python
 defs = re.findall(r'(?:fn|public|private|protected|void|int|string|async|function)\s+([a-zA-Z0-9_]+)\s*\(', content)
 for d in defs:
 if d not in {'if', 'for', 'while', 'catch', 'switch', 'init', 'void'}:
 label = f"{mod}.{d}"
 func_registry[d] = (mod, label)
 except:
 pass

 if not func_registry:
 # Fallback pentru codebase fără funcții detectate - arată fișierele și conexiunile simple
 lines = ["flowchart LR"]
 for fp in files_list[:12]:
 rel = str(fp.relative_to(root_dir))
 nid = _mid(rel)
 lines.append(f' {nid}["{fp.name}"]')
 return "\n".join(lines)

 # Găsim apelurile
 edges = []
 visited = set()
 for fp in files_list:
 mod = fp.stem
 suffix = fp.suffix.lower()
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 if suffix == '.py':
 tree = ast.parse(content)
 for func_node in ast.walk(tree):
 if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
 continue
 caller_label = f"{mod}.{func_node.name}"
 caller_id = _mid(caller_label)
 for call in ast.walk(func_node):
 if not isinstance(call, ast.Call):
 continue
 callee_name = None
 if isinstance(call.func, ast.Name):
 callee_name = call.func.id
 elif isinstance(call.func, ast.Attribute):
 callee_name = call.func.attr
 if callee_name and callee_name in func_registry:
 callee_mod, callee_label = func_registry[callee_name]
 callee_id = _mid(callee_label)
 if caller_id != callee_id:
 edge = (caller_id, caller_label, callee_id, callee_label)
 if edge not in visited:
 visited.add(edge)
 edges.append(edge)
 else:
 # Pentru limbaje non-Python
 defs = re.findall(r'(?:fn|public|private|protected|void|int|string|async|function)\s+([a-zA-Z0-9_]+)\s*\(', content)
 local_funcs = [d for d in defs if d not in {'if', 'for', 'while', 'catch', 'switch', 'init', 'void'}]
 
 for callee_name, (callee_mod, callee_label) in func_registry.items():
 if f"{callee_name}(" in content:
 callee_id = _mid(callee_label)
 caller_label = f"{mod}.main"
 matches = list(re.finditer(r'(?:fn|public|private|protected|void|int|string|async|function)\s+([a-zA-Z0-9_]+)\s*\(', content))
 if matches:
 call_pos = content.find(f"{callee_name}(")
 caller_name = "main"
 for m in reversed(matches):
 if m.start() < call_pos:
 cand = m.group(1)
 if cand not in {'if', 'for', 'while', 'catch', 'switch', 'init', 'void'} and cand != callee_name:
 caller_name = cand
 break
 caller_label = f"{mod}.{caller_name}"
 
 caller_id = _mid(caller_label)
 if caller_id != callee_id:
 edge = (caller_id, caller_label, callee_id, callee_label)
 if edge not in visited:
 visited.add(edge)
 edges.append(edge)
 except:
 pass

 lines = ["flowchart TD"]
 shown_nodes = {} # id -> label

 for caller_id, caller_label, callee_id, callee_label in edges[:25]:
 shown_nodes[caller_id] = caller_label
 shown_nodes[callee_id] = callee_label

 if not shown_nodes:
 for name, (mod, label) in list(func_registry.items())[:15]:
 nid = _mid(label)
 shown_nodes[nid] = label

 for nid, label in shown_nodes.items():
 lines.append(f' {nid}["{label}"]')

 for caller_id, _, callee_id, _ in edges[:25]:
 if caller_id in shown_nodes and callee_id in shown_nodes:
 lines.append(f" {caller_id} --> {callee_id}")

 return "\n".join(lines)


def generate_package_diagram(files_list, root_dir):
 """
 Grupează fișierele pe directoare (pachete) și arată dependențele prin import.
 Folosește subgraph Mermaid pentru o grupare vizuală clară.
 """
 packages = {}
 for fp in files_list:
 rel = fp.relative_to(root_dir)
 pkg = rel.parts[0] if len(rel.parts) > 1 else "root"
 packages.setdefault(pkg, []).append(fp.name)

 file_basenames = {f.stem: str(f.relative_to(root_dir)) for f in files_list}
 pkg_deps = set()

 for fp in files_list:
 suffix = fp.suffix.lower()
 rel = fp.relative_to(root_dir)
 src_pkg = rel.parts[0] if len(rel.parts) > 1 else "root"
 
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 imported_modules = []
 if suffix == '.py':
 tree = ast.parse(content)
 for node in ast.walk(tree):
 if isinstance(node, ast.Import):
 for alias in node.names:
 imported_modules.append(alias.name.split('.')[0])
 elif isinstance(node, ast.ImportFrom) and node.module:
 imported_modules.append(node.module.split('.')[0])
 else:
 imports = re.findall(r'(?:import|using|use|#include|require)\s+["\'<]?([a-zA-Z0-9_\-\./\:]+)["\'>]?;?', content)
 for imp in imports:
 imp_base = imp.split('/')[-1].split('.')[-1].split(':')[-1].strip()
 imported_modules.append(imp_base)
 
 for imported in imported_modules:
 if imported and imported in file_basenames:
 dest_path = Path(file_basenames[imported])
 dest_pkg = dest_path.parts[0] if len(dest_path.parts) > 1 else "root"
 if src_pkg != dest_pkg:
 pkg_deps.add((src_pkg, dest_pkg))
 except:
 pass

 lines = ["graph LR"]

 # Subgraph pentru fiecare pachet
 for pkg, files in packages.items():
 pkg_id = _mid(pkg)
 lines.append(f' subgraph {pkg_id} ["{pkg}"]')
 for fname in files[:6]:
 file_id = _mid(f"{pkg}_{fname}")
 lines.append(f' {file_id}["{fname}"]')
 if len(files) > 6:
 more_id = _mid(f"{pkg}_more")
 lines.append(f' {more_id}["... +{len(files)-6} fișiere"]')
 lines.append(" end")

 # Săgeți între pachete
 for src, dest in pkg_deps:
 s = _mid(src)
 d = _mid(dest)
 lines.append(f" {s} --> {d}")

 # Dacă totul e în root, arată fișierele ca noduri individuale cu links
 if len(packages) == 1 and "root" in packages:
 lines = ["graph LR"]
 for fp in files_list[:12]:
 nid = _mid(fp.stem)
 lines.append(f' {nid}["{fp.name}"]')
 
 visited_e = set()
 for fp in files_list:
 suffix = fp.suffix.lower()
 try:
 with open(fp, "r", encoding="utf-8", errors="ignore") as f:
 content = f.read()
 
 imported_modules = []
 if suffix == '.py':
 tree = ast.parse(content)
 for node in ast.walk(tree):
 if isinstance(node, ast.Import):
 for alias in node.names:
 imported_modules.append(alias.name.split('.')[0])
 elif isinstance(node, ast.ImportFrom) and node.module:
 imported_modules.append(node.module.split('.')[0])
 else:
 imports = re.findall(r'(?:import|using|use|#include|require)\s+["\'<]?([a-zA-Z0-9_\-\./\:]+)["\'>]?;?', content)
 for imp in imports:
 imp_base = imp.split('/')[-1].split('.')[-1].split(':')[-1].strip()
 imported_modules.append(imp_base)

 for imported in imported_modules:
 if imported and imported in file_basenames:
 e = (_mid(fp.stem), _mid(imported))
 if e not in visited_e and e[0] != e[1]:
 visited_e.add(e)
 lines.append(f" {e[0]} --> {e[1]}")
 except:
 pass

 return "\n".join(lines)
