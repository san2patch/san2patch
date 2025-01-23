from typing import TypeVar

from pydantic import BaseModel, Field

OutputT = TypeVar("PromptRet", bound=BaseModel)
