import subprocess
from typing import Any, Dict, Tuple

from agent_k.tools.tool_interface import ToolInterface


class PythonExecTool(ToolInterface):
    """
    A Tool that executes Python code securely in a container.
    """

    def get_definition(self) -> Dict[str, Any]:
        """
        Return the JSON/dict definition of the tool's function
        in the format expected by the OpenAI function calling API.
        """
        return {
            "function": {
                "name": "execute_python_code",
                "description": "Executes Python code securely in a container. Python version 3.12 is installed in the container.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Reasoning based on the given context before generating the code.",
                        },
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute. The last line of the code should be a print statement that prints the final calculation result.",
                        },
                    },
                    "required": ["reasoning", "code"],
                },
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the Python code in a Docker container and return the output.
        """
        code = arguments["code"]

        # First, decode escaped newlines if present (e.g., from a JSON string)
        if "\\n" in code and "\n" not in code:
            code = code.encode("utf-8").decode("unicode_escape")
        code_lines = code.splitlines()

        # Corner case: Remove incorrect indentation if whitespace number is not a multiple of 2
        for i, line in enumerate(code_lines):
            left_space = len(line) - len(line.lstrip())
            if left_space % 2 != 0:
                code_lines[i] = line.lstrip()

        # Corner case: If the last line is an assignment, parse the variable name
        if "=" in code_lines[-1]:
            variable_name = code_lines[-1].split("=")[0].strip()
            code_lines += [f"print({variable_name})"]

        # Corner case: If last line is not a print statement, wrap it in a print statement
        if not code_lines[-1].startswith("print("):
            code_lines = code_lines[:-1] + [f"print({code_lines[-1]})"]
            code = "\n".join(code_lines)

        output, errors = self._run_code_in_container(code)
        if errors:
            return f"[Error]\n{errors}"

        return output

    def run_code_block(self, msg_w_code: str) -> str:
        """
        Parse the python code block from the AI generated message and run the code in a container.
        """
        output = -1
        code = msg_w_code.split("```python")[1].split("```")[0].strip()
        code += "\nprint(ans)"

        output, errors = self._run_code_in_container(code)
        if errors:
            return f"[Error]\n{errors}"

        return output

    @staticmethod
    def _run_code_in_container(
        code: str, container_name: str = "sandbox"
    ) -> Tuple[str, str]:
        """
        Helper function that actually runs Python code inside a Docker container named `sandbox` (by default).
        """
        cmd = [
            "docker",
            "exec",
            "-i",
            container_name,
            "python",
            "-c",
            "import sys; exec(sys.stdin.read())",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = process.communicate(code)
        return out, err


if __name__ == "__main__":
    code = "```python\n# Extracted tonnage values in tonnes\ntonnage_indicated = 90900000.0  # Indicated mineral resource tonnage\ntonnage_inferred = 133400000.0  # Inferred mineral resource tonnage\ntonnage_measured = 0.0          # Measured mineral resource tonnage (none assigned)\n\n# Calculate total mineral resource tonnage\nans = tonnage_indicated + tonnage_inferred + tonnage_measured\n```"

    output = PythonExecTool().run_code_block(code)
    print(output)
