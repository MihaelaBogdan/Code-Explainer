import ast
from pathlib import Path
import numpy as np

SECURITY_PATTERNS = {
    "Command Injection": [
        "os.system(user_input)",
        "subprocess.run(cmd, shell=True)"
    ],

    "SQL Injection": [
        "cursor.execute('SELECT * FROM users WHERE id=' + user_input)",
        "query = f'SELECT * FROM users WHERE name={name}'"
    ],

    "Unsafe Deserialization": [
        "pickle.loads(data)",
        "yaml.load(data)"
    ],

    "Weak Cryptography": [
        "hashlib.md5(password.encode())",
        "hashlib.sha1(data)"
    ]
}


class SecurityVisitor(ast.NodeVisitor):

    def __init__(self, source_lines,indexer=None):
        self.findings = []
        self.source_lines = source_lines

        self.indexer = indexer
        self.semantic_vectors = {}

        if self.indexer:

            for vuln_type, examples in SECURITY_PATTERNS.items():

                emb = self.indexer.get_embeddings(examples)

                mean_vec = np.mean(emb, axis=0).astype("float32")

                mean_vec /= np.linalg.norm(mean_vec)

                self.semantic_vectors[vuln_type] = mean_vec


    def add_finding(self, vuln_type, severity, node, description):

        line = getattr(node, "lineno", 0)

        code_line = ""

        if 0 < line <= len(self.source_lines):
            code_line = self.source_lines[line - 1].strip()

        semantic_type, semantic_score = self.semantic_similarity(code_line)

        self.findings.append({
            "type": vuln_type,
            "severity": severity,
            "line": line,
            "code": code_line,
            "description": description,
            "semantic_match": semantic_type,
            "semantic_score": round(semantic_score, 4),
        })

    def semantic_similarity(self, code_line):

        if not self.indexer:
            return None, 0.0

        try:

            emb = self.indexer.get_embeddings([code_line])[0]
            emb = emb / np.linalg.norm(emb)

            best_type = None
            best_score = 0.0

            for vuln_type, vec in self.semantic_vectors.items():

                similarity = float(np.dot(emb, vec))

                if similarity > best_score:
                    best_score = similarity
                    best_type = vuln_type

            return best_type, best_score

        except:
            return None, 0.0


    def visit_Call(self, node):


        if isinstance(node.func, ast.Name):

            if node.func.id in ["eval", "exec"]:

                self.add_finding(
                    "Dynamic Code Execution",
                    "HIGH",
                    node,
                    f"Use of dangerous function '{node.func.id}'."
                )


        elif isinstance(node.func, ast.Attribute):

            func_name = node.func.attr


            if func_name == "system":

                self.add_finding(
                    "Command Injection",
                    "HIGH",
                    node,
                    "os.system detected."
                )


            if func_name in ["run", "Popen", "call"]:

                for kw in node.keywords:

                    if kw.arg == "shell":

                        if (
                            isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):

                            self.add_finding(
                                "Shell Execution",
                                "HIGH",
                                node,
                                "subprocess executed with shell=True."
                            )


            if func_name == "execute":

                if node.args:

                    arg = node.args[0]

                    if isinstance(arg, ast.BinOp):

                        self.add_finding(
                            "Possible SQL Injection",
                            "HIGH",
                            node,
                            "SQL query built using string concatenation."
                        )

                    elif isinstance(arg, ast.JoinedStr):

                        self.add_finding(
                            "Possible SQL Injection",
                            "HIGH",
                            node,
                            "SQL query built using f-string."
                        )


            if func_name in ["load", "loads"]:

                if isinstance(node.func.value, ast.Name):

                    if node.func.value.id == "pickle":

                        self.add_finding(
                            "Unsafe Deserialization",
                            "HIGH",
                            node,
                            "pickle deserialization detected."
                        )


            if func_name in ["md5", "sha1"]:

                self.add_finding(
                    "Weak Cryptography",
                    "MEDIUM",
                    node,
                    f"Weak hash function '{func_name}' detected."
                )

        self.generic_visit(node)


    def visit_Assign(self, node):

        suspicious_names = [
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "jwt"
        ]

        for target in node.targets:

            if isinstance(target, ast.Name):

                name = target.id.lower()

                if any(x in name for x in suspicious_names):

                    if isinstance(node.value, ast.Constant):

                        if isinstance(node.value.value, str):

                            self.add_finding(
                                "Hardcoded Secret",
                                "HIGH",
                                node,
                                f"Possible hardcoded secret in variable '{target.id}'."
                            )

        self.generic_visit(node)


def analyze_python_file(file_path, root_dir, indexer=None):

    path = Path(file_path)

    try:

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    except:
        return []

    try:

        tree = ast.parse(content)

    except SyntaxError:
        return []

    visitor = SecurityVisitor(content.splitlines(), indexer=indexer)

    visitor.visit(tree)

    findings = []

    for finding in visitor.findings:

        findings.append({
            "file": str(path.relative_to(root_dir)),
            **finding
        })

    return findings
