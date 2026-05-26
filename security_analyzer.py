import ast
from pathlib import Path


class SecurityVisitor(ast.NodeVisitor):

    def __init__(self, source_lines):
        self.findings = []
        self.source_lines = source_lines

    def add_finding(self, vuln_type, severity, node, description):

        line = getattr(node, "lineno", 0)

        code_line = ""

        if 0 < line <= len(self.source_lines):
            code_line = self.source_lines[line - 1].strip()

        self.findings.append({
            "type": vuln_type,
            "severity": severity,
            "line": line,
            "code": code_line,
            "description": description
        })

    # =========================
    # FUNCTION CALL ANALYSIS
    # =========================

    def visit_Call(self, node):

        # --------------------------------
        # eval / exec
        # --------------------------------

        if isinstance(node.func, ast.Name):

            if node.func.id in ["eval", "exec"]:

                self.add_finding(
                    "Dynamic Code Execution",
                    "HIGH",
                    node,
                    f"Use of dangerous function '{node.func.id}'."
                )

        # --------------------------------
        # Attribute-based calls
        # ex: os.system()
        # --------------------------------

        elif isinstance(node.func, ast.Attribute):

            func_name = node.func.attr

            # os.system()

            if func_name == "system":

                self.add_finding(
                    "Command Injection",
                    "HIGH",
                    node,
                    "os.system detected."
                )

            # subprocess(... shell=True)

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

            # SQL execute()

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

            # pickle.loads()

            if func_name in ["load", "loads"]:

                if isinstance(node.func.value, ast.Name):

                    if node.func.value.id == "pickle":

                        self.add_finding(
                            "Unsafe Deserialization",
                            "HIGH",
                            node,
                            "pickle deserialization detected."
                        )

            # hashlib.md5()

            if func_name in ["md5", "sha1"]:

                self.add_finding(
                    "Weak Cryptography",
                    "MEDIUM",
                    node,
                    f"Weak hash function '{func_name}' detected."
                )

        self.generic_visit(node)

    # =========================
    # HARD CODED SECRETS
    # =========================

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


def analyze_python_file(file_path, root_dir):

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

    visitor = SecurityVisitor(content.splitlines())

    visitor.visit(tree)

    findings = []

    for finding in visitor.findings:

        findings.append({
            "file": str(path.relative_to(root_dir)),
            **finding
        })

    return findings