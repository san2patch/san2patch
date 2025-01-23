from langgraph.graph import StateGraph

from san2patch.patching.graph.ablation.cot_comprehend_graph import (
    generate_comprehend_graph,
)
from san2patch.patching.graph.ablation.cot_howtofix_graph import generate_howtofix_graph
from san2patch.patching.graph.ablation.cot_runpatch_graph import (
    RunPatchState,
    generate_runpatch_graph,
)
from san2patch.patching.graph.ablation.cot_wheretofix_graph import (
    generate_wheretofix_graph,
)
from san2patch.patching.llm.base_llm_patcher import BaseLLMPatcher
from san2patch.patching.llm.openai_llm_patcher import GPT4oPatcher
from san2patch.utils.reducers import *


class CoTPatchState(RunPatchState):
    pass


def generate_cot_patch_graph(
    LLMPatcher: BaseLLMPatcher = GPT4oPatcher,
    consistency_num: int = 3,
    branch_num: int = 5,
    select_num: int = 3,
    cached=False,
):
    graph_name = "cot_patch_graph"
    patch_builder = StateGraph(CoTPatchState)

    patch_builder.add_node(
        "comprehend",
        generate_comprehend_graph(LLMPatcher, cached=False).with_config({"run_name": "Comprehend"}),
    )
    patch_builder.add_node(
        "wheretofix",
        generate_wheretofix_graph(LLMPatcher, cached=False).with_config({"run_name": "WhereToFix"}),
    )
    patch_builder.add_node(
        "howtofix",
        generate_howtofix_graph(LLMPatcher, cached=False).with_config({"run_name": "HowToFix"}),
    )
    patch_builder.add_node(
        "runpatch",
        generate_runpatch_graph(LLMPatcher).with_config({"run_name": "RunPatch"}),
    )
    patch_builder.add_node("patch_end", lambda state: {"last_node": "patch_end"})

    patch_builder.set_entry_point("comprehend")
    patch_builder.set_finish_point("patch_end")

    patch_builder.add_edge("comprehend", "wheretofix")
    patch_builder.add_edge("wheretofix", "howtofix")
    patch_builder.add_edge("howtofix", "runpatch")
    patch_builder.add_edge("runpatch", "patch_end")

    return patch_builder.compile()
