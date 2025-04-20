from typing import Optional

from agent_k.agents.base_agent import BaseAgent
from agent_k.config.logger import logger
from agent_k.tools.python_code_interpreter import PythonExecTool
from agent_k.tools.tool_manager import ToolManager


class PythonExecAgent(BaseAgent):
    """
    An agent specialized in executing Python code in a Docker container.
    """

    def __init__(
        self,
        model_name: str = "o3-mini",
        developer_prompt: str = """
                    You are a helpful assistant that can write and execute Python code.
                """,
        logger=logger,
        reasoning_effort: Optional[
            str
        ] = None,  # optional; if provided, passed to API calls
    ):
        super().__init__(
            model_name=model_name,
            developer_prompt=developer_prompt,
            logger=logger,
            reasoning_effort=reasoning_effort,
        )
        self.setup_tools()

    def setup_tools(self) -> None:
        """
        Create a ToolManager, instantiate the PythonExecTool and register it with the ToolManager.
        """
        self.tool_manager = ToolManager(logger=self.logger, model_name=self.model_name)

        # Create the Python execution tool
        python_exec_tool = PythonExecTool()

        # Register the Python execution tool
        self.tool_manager.register_tool(python_exec_tool)
