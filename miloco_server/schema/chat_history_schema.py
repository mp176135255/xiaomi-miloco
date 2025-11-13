# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""
Chat history data models
Define data structures related to chat history
"""

import json
import logging
from typing import Any, List, Optional, Union

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionMessageToolCallParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import Function
from pydantic import BaseModel, Field

from miloco_server.schema.chat_schema import Event, Instruction, Template

logger = logging.getLogger(__name__)


class ChatHistoryMessages:
    """Chat history messages manager"""

    def __init__(self, messages: Optional[list[ChatCompletionMessageParam]] = None):
        self._messages: list[ChatCompletionMessageParam] = messages if messages is not None else []

    def add_content(self, role: str, content: str):
        """
        Add message
        """
        self._messages.append({"role": role, "content": content})

    def add_content_list(self, role: str, content_list: list[dict]):
        """
        Add message
        """
        add_content = []
        for content_dict in content_list:
            add_content.append(content_dict)
        self._messages.append({"role": role, "content": add_content})

    def add_tool_call_res_content(self, tool_call_id: str, name: str,
                                  content: str):
        """
        Add tool call content
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        })

    def add_message(self, message: dict[str, Any]):
        self._messages.append(message)

    def add_assistant_message(
            self,
            content: str,
            tool_calls: list[ChatCompletionMessageToolCall] = None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message[
                "tool_calls"] = ChatHistoryMessages.message_tool_call_2_param(
                    tool_calls)
        self._messages.append(message)

    def has_initialized(self) -> bool:
        """
        Check if initialized
        """
        return len(
            self._messages) > 0 and self._messages[0].get("role") == "system"

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        """
        Get messages
        """
        return self._messages

    def to_json(self) -> str:
        """
        Convert messages to JSON
        """
        try:
            return json.dumps(self._messages, ensure_ascii=False)
        except (ValueError, TypeError) as e:
            logger.error("Error serializing messages: %s", e, exc_info=True)
            return ""

    @staticmethod
    def from_json(json_str: Optional[str]) -> "ChatHistoryMessages":
        """
        Convert JSON to messages
        """
        if json_str is None or json_str == "":
            return ChatHistoryMessages()
        messages_data = json.loads(json_str)
        chat_history_messages = ChatHistoryMessages(messages_data)
        return chat_history_messages

    @staticmethod
    def message_tool_call_2_param(
        tool_calls: list[ChatCompletionMessageToolCall]
    ) -> list[ChatCompletionMessageToolCallParam]:
        """Convert ChatCompletionMessageToolCall to ChatCompletionMessageToolCallParam"""
        return [
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                type="function",
                function=Function(name=tool_call.function.name,
                                  arguments=tool_call.function.arguments))
            for tool_call in tool_calls
        ]


class ChatHistorySession(BaseModel):
    """Chat history session"""
    data: List[Union[Event, Instruction]] = Field(default_factory=list, description="Session list")

    def add_event(self, event: Event):
        self.data.append(event)

    def add_instruction(self, instruction: Instruction):
        self.data.append(instruction)

    def zip_toast_stream(self) -> None:
        """Merge ToastStream"""
        session = []
        current_toast_stream_header = None
        toast = ""
        for item in self.data:
            if (isinstance(item, Instruction) and item.header.type == "instruction" and
                    item.header.namespace == "Template" and item.header.name == "ToastStream"):
                if not current_toast_stream_header:
                    current_toast_stream_header = item.header.model_copy(deep=True)
                    toast = json.loads(item.payload).get("stream", "")
                else:
                    toast += json.loads(item.payload).get("stream", "")
            else:
                if current_toast_stream_header:
                    toast_stream = Template.ToastStream(stream=toast)
                    instruction = Instruction(
                        header=current_toast_stream_header,
                        payload=toast_stream.model_dump_json())
                    session.append(instruction)
                    current_toast_stream_header = None
                    toast = ""

                session.append(item)

        if current_toast_stream_header:
            toast_stream = Template.ToastStream(stream=toast)
            instruction = Instruction(
                header=current_toast_stream_header,
                payload=toast_stream.model_dump_json())
            session.append(instruction)
            current_toast_stream_header = None
            toast = ""

        self.data = session


class ChatHistorySimpleInfo(BaseModel):
    session_id: str = Field(..., description="Record ID, UUID")
    title: str = Field(..., description="Conversation title")
    timestamp: int = Field(..., description="Timestamp")

class ChatHistoryResponse(ChatHistorySimpleInfo):
    session: ChatHistorySession = Field(
        ...,
        description="Session content, alternating Event/Instruction sequence, can be empty"
    )


class ChatHistoryStorage(ChatHistoryResponse):
    messages: Optional[str] = Field(None, description="Message content serialized JSON schema")
