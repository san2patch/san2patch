from pydantic import BaseModel, Field

from san2patch.patching.prompt.base_prompts import BasePrompt


# Prompts
class SlicingModel(BaseModel):
    root_cause_vars: list[str] = Field(
        description="Variables from the Target_Code_Line that are suspected to be related to the vulnerability."
    )
    program_slices: list[str] = Field(
        description="Provide the backward program slices that influence the root_cause_vars."
    )


class BackwardSlicingPrompt(BasePrompt):
    def __init__(self, **kwargs):
        super().__init__(SlicingModel, "wheretofix/backward_slicing.jinja2", **kwargs)
