import subprocess
from abc import ABC, abstractmethod
from typing import Any


class ToolInterface(ABC):
    """
    An abstract class for any 'tool' that an agent can call.
    Every tool must provide two things:
    1) A definition (in JSON schema format) as expected by OpenAI function calling specifications.
    2) A 'run' method to handle the logic given the arguments.
    """

    @abstractmethod
    def get_definition(self) -> dict[str, Any]:
        """
        Return the JSON/dict definition of the tool's function.
        Example:
        {
            "function": {
                "name": "<tool_function_name>",
                "description": "<what this function does>",
                "parameters": { <JSON schema> }
            }
        }
        """
        pass

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> str:
        """
        Execute the tool using the provided arguments and return a result as a string.
        """
        pass


class PythonExecTool(ToolInterface):
    """
    A Tool that executes Python code securely in a container.
    """

    def get_definition(self) -> dict[str, Any]:
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

    def run(self, arguments: dict[str, Any]) -> str:
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

        # If the last line is not a print statement, wrap it in a print statement
        if not code.splitlines()[-1].startswith("print("):
            code += "\nprint(ans)"

        output, errors = self._run_code_in_container(code)
        output = output.strip()
        if errors:
            return f"[Error]\n{errors}"

        return output

    @staticmethod
    def _run_code_in_container(
        code: str, container_name: str = "python_code_interpreter"
    ) -> tuple[str, str]:
        """
        Helper function that actually runs Python code inside a Docker container named `python_code_interpreter` (by default).
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
