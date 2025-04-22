from abc import ABC, abstractmethod
from typing import Optional

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent_k.tools.chat_messages import ChatMessages
from agent_k.tools.tool_manager import ToolManager


class BaseAgent(ABC):
    """
    An abstract base agent that defines the high-level approach to handling user tasks
    and orchestrating calls to the OpenAI API.
    """

    def __init__(
        self,
        developer_prompt: str,
        model_name: str,
        logger,
        reasoning_effort: Optional[str] = None,
    ):
        self.developer_prompt = developer_prompt
        self.model_name = model_name
        self.messages = ChatMessages(developer_prompt)
        self.tool_manager: Optional[ToolManager] = None
        self.logger = logger
        self.reasoning_effort = reasoning_effort

    @abstractmethod
    def setup_tools(self) -> None:
        pass

    def add_message(self, content: str) -> None:
        self.messages.add_user_message(content)

    def task(
        self,
        user_task: str,
        tool_call_enabled: bool = True,
        return_tool_response_as_is: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        # Use the reasoning_effort provided in the method call if present, otherwise fall back to the agent's default
        final_reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        )

        # Add user message
        self.add_message(user_task)

        tools = []
        if tool_call_enabled and self.tool_manager:
            tools = self.tool_manager.get_tool_definitions()

        # Build parameter dict and include reasoning_effort only if not None
        params = {
            "model": self.model_name,
            "messages": self.messages.get_messages(),
            "tools": tools,
        }
        if final_reasoning_effort is not None:
            params["reasoning_effort"] = final_reasoning_effort

        self.logger.debug("Sending completion request using Litellm...")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((Exception)),
            before_sleep=lambda retry_state: self.logger.warning(
                f"Retrying API call after error (attempt {retry_state.attempt_number})"
            ),
        )
        def execute_with_retry():
            # First get the completion
            response = litellm.completion(**params)

            # Check if tool call is needed
            tool_calls = response.choices[0].message.tool_calls
            if tool_call_enabled and self.tool_manager and tool_calls:
                self.logger.debug(f"Tool calls requested: {tool_calls}")
                return self.tool_manager.handle_tool_call_sequence(
                    response,
                    return_tool_response_as_is,
                    self.messages,
                    self.model_name,
                    reasoning_effort=final_reasoning_effort,
                )
            else:
                # No tool call needed, return the completion response directly
                return response

        try:
            response = execute_with_retry()

            # If it's a tool call response, it's already processed and can be returned directly
            if tool_call_enabled and self.tool_manager and response:
                return response

            # Otherwise, it's a regular completion response
            tool_calls = response.choices[0].message.tool_calls
            if tool_call_enabled and self.tool_manager and tool_calls:
                self.logger.debug(f"Tool calls requested: {tool_calls}")
                final_response = self.tool_manager.handle_tool_call_sequence(
                    response,
                    return_tool_response_as_is,
                    self.messages,
                    self.model_name,
                    reasoning_effort=final_reasoning_effort,
                )
                return final_response
        except Exception as e:
            self.logger.error(f"Failed after retries: {str(e)}")
            raise

        # No tool call, normal assistant response
        response_message = response.choices[0].message.content
        self.messages.add_assistant_message(response_message)
        return response_message
