# AI Codebase Explainer - Local

A local, AI-powered codebase exploration, parsing, and analysis tool built with Streamlit. It uses Abstract Syntax Tree (AST) parsing for structural analysis and diagrams, alongside a CodeBERT-powered semantic vector index (FAISS) for advanced search, code smell detection, and codebase quizzes.

---

## 🚀 Features

### 📁 Codebase Parsing & Structural Analysis
- **ZIP Upload & Scan**: Upload a ZIP archive of your project or scan local directories. 
- **File System Parsing**: Automatically generates interactive directory trees while ignoring build artifacts, virtual environments, and configuration folders (like `.git`, `node_modules`, `venv`).
- **Diagram Generation**: Automatically parses files using Python's `ast` to generate multiple architectural diagrams:
  - UML Class Diagrams
  - Sequence Diagrams
  - Dependency Flow Diagrams
  - Code Flowcharts
  - Package Diagrams

### 🧠 CodeBERT Semantic Vector Index
- **Semantic Vector Storage**: Indexes your codebase into chunks using `microsoft/codebert-base` with a local CPU/GPU FAISS index.
- **Natural Language Code Search**: Find logic inside your codebase using semantic search queries rather than just keyword matches.
- **Attention Matrix Visualization**: Inspect and visualize the transformer-level token attention matrices to see how the model processes code fragments.

### 🛡️ AST & Semantic Security Analysis
- **Security Vulnerability Scanner**: Scans Python files dynamically for potential vulnerabilities such as:
  - Command Injection (e.g. `os.system`)
  - SQL Injection (e.g. string concatenation/f-strings in SQL execution)
  - Unsafe Deserialization (e.g. `pickle` or `yaml.load`)
  - Weak Cryptography (e.g. `md5` or `sha1` hashes)
  - Hardcoded Secrets (e.g. variables named token/password/api_key)
- **Code Smell Detector**: Performs semantic code smell analysis using embeddings, identifying issues like:
  - Functions with too many parameters
  - Missing error handling block structures (`try-except`)
  - Deeply nested conditional structures (too many indentation levels)
  - Missing documentation (lack of docstrings/comments)
  - Multi-responsibility functions (violating SRP)
  - Hardcoded constants (magic numbers/strings)

### 📝 Quiz & Codebase Knowledge Test
- Test your understanding of the codebase structure through interactive quizzes generated dynamically based on the parsed AST structure and semantic similarity checks.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://gitlab.cc.internal/connections-ai-squad/code-explainer-mihaela-dragos.git
   cd code-explainer-mihaela-dragos
   ```

2. **Set up a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Project Structure

- `app.py`: The main Streamlit entrypoint containing UI routing, state setup, layout, and rendering pages.
- `code_parser.py`: Implementation of file parsing, AST traversal, and Mermaid/structural diagram generation.
- `vector_store.py`: Semantic search backend leveraging `CodeBERTIndexer` and FAISS for code embeddings and search matching.
- `security_analyzer.py`: Security scan runner leveraging AST patterns combined with semantic similarity scoring.
- `requirements.txt`: Configuration listing required Python dependencies.
- `style.css`: Custom CSS modifications to deliver a polished dark-mode interface experience.
