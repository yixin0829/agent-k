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
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute",
                        }
                    },
                    "required": ["code"],
                },
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the Python code in a Docker container and return the output.
        """
        code = arguments["code"]
        code_stripped = code.strip('"""')

        output, errors = self._run_code_in_container(code_stripped)
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
