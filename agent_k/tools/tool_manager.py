import json
from typing import Any, Dict, List, Optional

import litellm

from agent_k.tools.chat_messages import ChatMessages
from agent_k.tools.tool_interface import ToolInterface


class ToolManager:
    """
    Manages one or more tools. Allows you to:
      - Register multiple tools
      - Retrieve their definitions
      - Invoke the correct tool by name
      - Handle the entire tool call sequence
    """

    def __init__(self, logger=None, model_name: str = None):
        self.tools = {}
        self.logger = logger
        self.model_name = model_name

    def register_tool(self, tool: ToolInterface) -> None:
        """
        Register a tool by using its function name as the key.
        """
        tool_def = tool.get_definition()
        tool_name = tool_def["function"]["name"]
        self.tools[tool_name] = tool
        # self.logger.debug(f"Registered tool '{tool_name}': {tool_def}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Return the list of tool definitions in the format expected by the OpenAI API.
        """
        definitions = []
        for _, tool in self.tools.items():
            tool_def = tool.get_definition()["function"]
            # self.logger.debug(f"Tool definition retrieved for '{name}': {tool_def}")
            definitions.append({"type": "function", "function": tool_def})
        return definitions

    def handle_tool_call_sequence(
        self,
        response,
        return_tool_response_as_is: bool,
        messages: ChatMessages,
        model_name: str,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        If the model wants to call a tool, parse the function arguments, invoke the tool,
        then optionally return the tool's raw output or feed it back to the model for a final answer.
        """
        # We take the first tool call from the model's response
        first_tool_call = response.choices[0].message.tool_calls[0]
        tool_name = first_tool_call.function.name

        args = json.loads(first_tool_call.function.arguments)

        if tool_name not in self.tools:
            raise ValueError(
                f"Error: The requested tool '{tool_name}' is not registered."
            )

        # 1. Invoke the tool
        tool_response = self.tools[tool_name].run(args)
        if "[Error]" in tool_response:
            self.logger.error(f"Tool '{tool_name}' response: {tool_response}")
            raise ValueError(tool_response)

        self.logger.info(f"Tool '{tool_name}' response: {tool_response}")

        # If returning the tool response "as is," just store and return it
        if return_tool_response_as_is:
            self.logger.debug(
                "Returning tool response as-is without further LLM calls."
            )
            messages.add_assistant_message(tool_response)
            return tool_response

        self.logger.debug(f"Tool call: {first_tool_call}")
        # Otherwise, feed the tool's response back to the LLM for a final answer
        function_call_result_message = {
            "role": "tool",
            "name": tool_name,
            "content": tool_response,
            "tool_call_id": first_tool_call.id,
        }

        complete_payload = messages.get_messages()
        complete_payload.append(response.choices[0].message)
        complete_payload.append(function_call_result_message)

        self.logger.debug(
            "Calling the model again with the tool response to get the final answer."
        )
        # Build parameter dict and only include reasoning_effort if not None
        params = {"model": model_name, "messages": complete_payload}
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

        response_after_tool_call = litellm.completion(**params)

        final_message = response_after_tool_call.choices[0].message.content
        self.logger.debug("Received final answer from model after tool call.")
        messages.add_assistant_message(final_message)
        return final_message
