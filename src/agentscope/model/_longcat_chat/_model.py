# -*- coding: utf-8 -*-
"""The LongCat Chat Completions model (OpenAI-compatible)."""
from collections import OrderedDict
from datetime import datetime
from typing import Literal, Any, AsyncGenerator, TYPE_CHECKING, List

from pydantic import Field

from .._base import ChatModelBase, _TOOL_CHOICE_LITERAL_MODES
from .._model_response import ChatResponse
from .._model_usage import ChatUsage
from ...credential import LongCatCredential
from ...formatter import FormatterBase, OpenAIChatFormatter
from ...message import (
    Msg,
    ThinkingBlock,
    ToolCallBlock,
    TextBlock,
    DataBlock,
    Base64Source,
)
from ...tool import ToolChoice

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai import AsyncStream
else:
    ChatCompletion = Any
    AsyncStream = Any


class LongCatChatModel(ChatModelBase):
    """The LongCat Chat Completions model (OpenAI-compatible endpoint)."""

    class Parameters(ChatModelBase.Parameters):
        """The parameters for the LongCat Chat model."""

        max_tokens: int | None = Field(
            default=None,
            title="Max Tokens",
            description="The maximum number of tokens for the LLM output.",
            gt=0,
        )

        temperature: float | None = Field(
            default=None,
            title="Temperature",
            description="The temperature for the LLM output.",
            ge=0,
            le=2,
        )

        top_p: float | None = Field(
            default=None,
            title="Top P",
            description="The top P value for the LLM output.",
            gt=0,
            le=1,
        )

        parallel_tool_calls: bool = Field(
            default=True,
            title="Parallel Tool Calls",
            description="Whether to enable parallel tool calls.",
        )

    type: Literal["longcat_chat"] = "longcat_chat"
    """The type of the chat model."""

    def __init__(
        self,
        credential: LongCatCredential,
        model: str,
        parameters: "LongCatChatModel.Parameters | None" = None,
        stream: bool = True,
        max_retries: int = 3,
        context_size: int = 131072,
        formatter: FormatterBase | None = None,
    ) -> None:
        super().__init__(
            credential=credential,
            model=model,
            parameters=parameters or self.Parameters(),
            stream=stream,
            max_retries=max_retries,
            context_size=context_size,
        )
        self.formatter = formatter or OpenAIChatFormatter()

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **generate_kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        import openai

        client = openai.AsyncClient(
            api_key=self.credential.api_key.get_secret_value(),
            base_url=self.credential.base_url,
        )

        formatted_messages = await self.formatter.format(messages)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": self.stream,
        }

        if self.parameters.max_tokens is not None:
            kwargs["max_tokens"] = self.parameters.max_tokens
        if self.parameters.temperature is not None:
            kwargs["temperature"] = self.parameters.temperature
        if self.parameters.top_p is not None:
            kwargs["top_p"] = self.parameters.top_p

        kwargs.update(generate_kwargs)

        fmt_tools, fmt_tool_choice = self._format_tools(tools, tool_choice)

        if fmt_tools:
            kwargs["tools"] = fmt_tools
            if not self.parameters.parallel_tool_calls:
                kwargs["parallel_tool_calls"] = False
        if fmt_tool_choice is not None:
            kwargs["tool_choice"] = fmt_tool_choice

        if self.stream:
            kwargs["stream_options"] = {"include_usage": True}

        start_datetime = datetime.now()
        response = await client.chat.completions.create(**kwargs)

        if self.stream:
            return self._parse_stream_response(start_datetime, response)

        return self._parse_completion_response(start_datetime, response)

    async def _parse_stream_response(
        self,
        start_datetime: datetime,
        response: AsyncStream,
        audio_format: str = "wav",
    ) -> AsyncGenerator[ChatResponse, None]:
        usage = None
        response_id: str | None = None
        acc_text = TextBlock(text="")
        acc_tool_calls: OrderedDict = OrderedDict()

        async with response as stream:
            async for chunk in stream:
                if chunk.usage:
                    u = chunk.usage
                    details = getattr(u, "prompt_tokens_details", None)
                    usage = ChatUsage(
                        input_tokens=u.prompt_tokens,
                        output_tokens=u.completion_tokens,
                        time=(datetime.now() - start_datetime).total_seconds(),
                        cache_input_tokens=(
                            getattr(details, "cached_tokens", 0) if details else 0
                        ),
                    )

                response_id = response_id or getattr(chunk, "id", None)

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                delta_text = getattr(delta, "content", None) or ""

                acc_text.text += delta_text

                delta_tool_call_blocks: List[ToolCallBlock] = []
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    idx = tool_call.index
                    args = tool_call.function.arguments or ""
                    if idx in acc_tool_calls:
                        acc_tool_calls[idx]["input"] += args
                    else:
                        acc_tool_calls[idx] = {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": args,
                        }
                    tc = acc_tool_calls[idx]
                    delta_tool_call_blocks.append(
                        ToolCallBlock(id=tc["id"], name=tc["name"], input=args),
                    )

                delta_contents: List[TextBlock | ToolCallBlock] = []
                if delta_text:
                    delta_contents.append(
                        TextBlock(id=acc_text.id, text=delta_text),
                    )
                delta_contents.extend(delta_tool_call_blocks)

                if delta_contents:
                    _kwargs: dict[str, Any] = {
                        "content": delta_contents,
                        "usage": usage,
                        "is_last": False,
                    }
                    if response_id:
                        _kwargs["id"] = response_id
                    yield ChatResponse(**_kwargs)

        final_contents: List[TextBlock | ToolCallBlock | DataBlock] = []
        if acc_text.text:
            final_contents.append(acc_text)
        for tc in acc_tool_calls.values():
            final_contents.append(
                ToolCallBlock(id=tc["id"], name=tc["name"], input=tc["input"]),
            )

        _final_kwargs: dict[str, Any] = {
            "content": final_contents,
            "usage": usage,
            "is_last": True,
        }
        if response_id:
            _final_kwargs["id"] = response_id
        yield ChatResponse(**_final_kwargs)

    def _parse_completion_response(
        self,
        start_datetime: datetime,
        response: ChatCompletion,
        audio_format: str = "wav",
    ) -> ChatResponse:
        content_blocks: List[TextBlock | ToolCallBlock | DataBlock] = []

        if response.choices:
            choice = response.choices[0]
            if choice.message.content:
                content_blocks.append(TextBlock(text=choice.message.content))
            for tool_call in choice.message.tool_calls or []:
                content_blocks.append(
                    ToolCallBlock(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_call.function.arguments,
                    ),
                )

        usage = None
        if response.usage:
            u = response.usage
            details = getattr(u, "prompt_tokens_details", None)
            usage = ChatUsage(
                input_tokens=u.prompt_tokens,
                output_tokens=u.completion_tokens,
                time=(datetime.now() - start_datetime).total_seconds(),
                cache_input_tokens=(
                    getattr(details, "cached_tokens", 0) if details else 0
                ),
            )

        resp_kwargs: dict[str, Any] = {
            "content": content_blocks,
            "is_last": True,
            "usage": usage,
        }
        response_id = getattr(response, "id", None)
        if response_id:
            resp_kwargs["id"] = response_id

        return ChatResponse(**resp_kwargs)

    def _format_tools(
        self,
        tools: list[dict] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[dict] | None, str | dict | None]:
        if tool_choice and tools:
            self._validate_tool_choice(tool_choice, tools)
            if tool_choice.tools:
                allowed = set(tool_choice.tools)
                tools = [t for t in tools if t["function"]["name"] in allowed]
        if not tool_choice:
            return tools, None
        mode = tool_choice.mode
        if mode not in _TOOL_CHOICE_LITERAL_MODES:
            return tools, {"type": "function", "function": {"name": mode}}
        return tools, mode
