# -*- coding: utf-8 -*-
"""The LongCat credential."""
from typing import Literal, Type, TYPE_CHECKING

from pydantic import ConfigDict, Field, SecretStr

from ._base import CredentialBase

if TYPE_CHECKING:
    from ..model import ChatModelBase

_LONGCAT_BASE_URL = "https://api.longcat.chat/openai"


class LongCatCredential(CredentialBase):
    """The credential for LongCat API."""

    model_config = ConfigDict(
        title="LongCat API",
    )

    type: Literal["longcat_credential"] = "longcat_credential"
    """The type of the credential."""

    api_key: SecretStr = Field(
        description="The LongCat API key.",
        title="API Key",
    )

    base_url: str = Field(
        default=_LONGCAT_BASE_URL,
        title="API Base URL",
        description="The base URL for the LongCat OpenAI-compatible API.",
    )

    @classmethod
    def get_chat_model_class(cls) -> Type["ChatModelBase"]:
        """Return the LongCatChatModel class."""
        from ..model import LongCatChatModel

        return LongCatChatModel
