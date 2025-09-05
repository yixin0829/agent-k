"""
Ablation analysis runner for Agent-K.

Non-intrusive: builds ablated graph variants externally and monkey-patches
the imported modules during each ablation run, then restores originals.

Variants implemented:
- exclude_fact_extraction_agent: skip fact extraction agent; pass retrieved docs to program_reasoner
- exclude_code_interpreter: skip Python execution; use LLM to compute final answer
- exclude_self_reflection: skip self-reflection; execute directly
- exclude_self_consistency: if looping, execute anyway instead of self-consistency
- exclude_global_validation_agent: bypass global validation agent
- exclude_react_agent: exclude code interpreter, self-reflection, and self-consistency; retrieve + fact extraction + direct generation (no agent)

Usage:
  # Run all variants with all samples
  uv run python src/experiments/ablation_tests.py --variants all --sample-size none
  # Run all variants with sample size 2 (default)
  uv run python src/experiments/ablation_tests.py --variants all --sample-size 2
  # Run specific variants
  uv run python src/experiments/ablation_tests.py --variants exclude_fact_extraction_agent,exclude_self_consistency
"""

from __future__ import annotations

import argparse
from typing import Callable

import litellm
from langgraph.graph import END, START, StateGraph

import src.config.experiment_config as config_experiment

# Import modules to allow monkey-patching
import src.experiments.agent_k as rag_mod
import src.experiments.multi_method_extraction as pdf_mod
from src.config.logger import logger
from src.utils.code_interpreter import PythonExecTool
from src.utils.general import get_curr_ts

# Configuration Variables
OUTPUT_DIR = "data/experiments/ablation_tests"

AVAILABLE_VARIANTS = [
    "exclude_fact_extraction_agent",
    "exclude_code_interpreter",
    "exclude_self_reflection",
    "exclude_self_consistency",
    "exclude_global_validation_agent",
    "exclude_react_agent",
]


def _join_documents(docs: list[str] | str) -> str:
    if isinstance(docs, str):
        return docs
    try:
        return "\n\n---\n\n".join(docs)
    except Exception:
        return str(docs)


def build_rag_graph_exclude_fact_extraction_agent() -> StateGraph:
    """START -> retrieve -> bypass_fact_extraction -> program_reasoner -> self_reflection
    -> conditional (execute | program_reasoner | self_consistency)
    -> self_consistency -> execute -> format_output -> END
    """

    def bypass_fact_extraction(state: dict):
        logger.info("---BYPASS FACT EXTRACTION AGENT: using retrieved docs as facts---")
        question = state["question"]
        docs_joined = _join_documents(state.get("documents", ""))
        user_msg = f"## Context\n{docs_joined}\n\n## Question\n{question}"
        return {
            "extracted_facts": docs_joined,
            "messages": [
                {"role": "user", "content": user_msg},
            ],
        }

    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("bypass_fact_extraction", bypass_fact_extraction)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("self_reflection", rag_mod.self_reflection)
    graph.add_node("self_consistency", rag_mod.self_consistency)
    graph.add_node("execute", rag_mod.execute)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "bypass_fact_extraction")
    graph.add_edge("bypass_fact_extraction", "program_reasoner")
    graph.add_edge("program_reasoner", "self_reflection")
    graph.add_conditional_edges(
        "self_reflection",
        rag_mod.reflection_router,
        {
            "execute": "execute",
            "program_reasoner": "program_reasoner",
            "self_consistency": "self_consistency",
        },
    )
    graph.add_edge("self_consistency", "execute")
    graph.add_edge("execute", "format_output")
    graph.add_edge("format_output", END)
    return graph


def build_rag_graph_exclude_code_interpreter() -> StateGraph:
    """Replace execution with an LLM-only finalizer that returns XML with
    <reasoning> and <answer>, then END.
    """

    EXECUTE_LLM_ONLY_SYSTEM_PROMPT = (
        "You are a helpful assistant. Compute the numeric answer using the given facts "
        "and the question without running code. Return ONLY the answer XML tag: <answer>...</answer>."
    )

    def execute_llm_only(state: dict):
        logger.info("--EXECUTE (LLM ONLY, NO CODE INTERPRETER)--")
        question = state.get("question", "")
        facts = state.get("extracted_facts", "")
        code_block_msg = state.get("previous_code", [""])[-1]

        messages = [
            {"role": "system", "content": EXECUTE_LLM_ONLY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Facts\n{facts}\n\n"
                    f"## Question\n{question}\n\n"
                    f"## Previously generated code (do NOT execute; use only as guidance)\n{code_block_msg}"
                ),
            },
        ]

        response = litellm.completion(
            model=config_experiment.PYTHON_AGENT_MODEL,
            temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
            messages=messages,
        )
        content = response["choices"][0]["message"]["content"]
        return {"generation": content}

    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("fact_extraction_agent", rag_mod.fact_extraction_agent)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("self_reflection", rag_mod.self_reflection)
    graph.add_node("self_consistency", rag_mod.self_consistency)
    graph.add_node("execute_llm_only", execute_llm_only)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "fact_extraction_agent")
    graph.add_edge("fact_extraction_agent", "program_reasoner")
    graph.add_edge("program_reasoner", "self_reflection")
    graph.add_conditional_edges(
        "self_reflection",
        rag_mod.reflection_router,
        {
            "execute": "execute_llm_only",
            "program_reasoner": "program_reasoner",
            "self_consistency": "self_consistency",
        },
    )
    graph.add_edge("self_consistency", "execute_llm_only")
    graph.add_edge("execute_llm_only", END)
    return graph


def build_rag_graph_exclude_self_reflection() -> StateGraph:
    """START -> retrieve -> fact_extraction_agent -> program_reasoner -> execute -> format_output -> END"""
    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("fact_extraction_agent", rag_mod.fact_extraction_agent)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("execute", rag_mod.execute)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "fact_extraction_agent")
    graph.add_edge("fact_extraction_agent", "program_reasoner")
    graph.add_edge("program_reasoner", "execute")
    graph.add_edge("execute", "format_output")
    graph.add_edge("format_output", END)
    return graph


def build_rag_graph_exclude_self_consistency() -> StateGraph:
    """Map self-consistency route to execute instead of running self-consistency."""

    def execute_with_self_consistency_override(state: rag_mod.GraphState):
        """Execute function that returns -1 when self-consistency would have been triggered."""
        logger.info(
            "--DETECTING SELF-REFLECTION LOOP (SELF-CONSISTENCY OVERRIDE TRIGGERED)--"
        )

        # Check if this execution is happening because self-consistency was triggered
        if len(state["previous_code"]) >= config_experiment.MAX_REFLECTION_ITERATIONS:
            logger.info(
                "--SELF-CONSISTENCY TRIGGERED IN ABLATION TEST, RETURNING -1 AS DEFAULT VALUE--"
            )
            output = "-1"
        else:
            # Normal execution
            code_block_msg = state["previous_code"][-1]
            output = PythonExecTool().run_code_block(code_block_msg)

        logger.info("--EXECUTION OUTPUT--")
        logger.info(output)

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Code interpreter execution result: {output}",
                },
            ],
        }

    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("fact_extraction_agent", rag_mod.fact_extraction_agent)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("self_reflection", rag_mod.self_reflection)
    graph.add_node("execute", execute_with_self_consistency_override)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "fact_extraction_agent")
    graph.add_edge("fact_extraction_agent", "program_reasoner")
    graph.add_edge("program_reasoner", "self_reflection")
    graph.add_conditional_edges(
        "self_reflection",
        rag_mod.reflection_router,
        {
            "execute": "execute",
            "program_reasoner": "program_reasoner",
            # When router chooses self_consistency, execute anyway
            "self_consistency": "execute",
        },
    )
    graph.add_edge("execute", "format_output")
    graph.add_edge("format_output", END)
    return graph


def build_rag_graph_exclude_react_agent() -> StateGraph:
    """START -> retrieve -> fact_extraction_agent -> generate_from_facts -> format_output -> END

    This variant performs retrieval and fact extraction only, then passes the
    extracted facts to a lightweight generation step to produce the final
    answer without code execution. The final XML formatting is handled by
    the existing `format_output` node.
    """

    GENERATE_FROM_FACTS_SYSTEM_PROMPT = (
        "You are a careful assistant. Using ONLY the provided extracted facts and "
        "the question, compute the final numeric answer. Do not write or execute "
        "code. If the required information is unavailable in the facts, return the "
        "default value specified in the question. Keep the reasoning concise."
    )

    def generate_from_facts(state: rag_mod.GraphState):
        logger.info("---GENERATE FROM FACTS (NO CODE)---")
        question = state.get("question", "")
        facts = state.get("extracted_facts", "")

        messages = [
            {"role": "system", "content": GENERATE_FROM_FACTS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (f"## Facts\n{facts}\n\n## Question\n{question}"),
            },
        ]

        response = litellm.completion(
            model=config_experiment.PYTHON_AGENT_MODEL,
            temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
            messages=messages,
        )
        content = response["choices"][0]["message"]["content"]
        return {
            "messages": [
                {"role": "assistant", "content": content},
            ]
        }

    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("fact_extraction_agent", rag_mod.fact_extraction_agent)
    graph.add_node("generate_from_facts", generate_from_facts)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "fact_extraction_agent")
    graph.add_edge("fact_extraction_agent", "generate_from_facts")
    graph.add_edge("generate_from_facts", "format_output")
    graph.add_edge("format_output", END)
    return graph


def build_pdf_graph_without_global_validation_agent():
    """PDF Agent-K graph without the global validation agent.
    Equivalent to: direct_extract_route -> map_extraction_agent -> reduce -> END
    """
    graph_builder = StateGraph(pdf_mod.State)
    graph_builder.add_node("map_extraction_agent", pdf_mod.map_extraction_agent)
    graph_builder.add_node(
        "reduce_extraction_results", pdf_mod.reduce_extraction_results
    )

    graph_builder.add_conditional_edges(
        START,
        pdf_mod.direct_extract_route,
        ["map_extraction_agent"],
    )
    graph_builder.add_edge("map_extraction_agent", "reduce_extraction_results")
    graph_builder.add_edge("reduce_extraction_results", END)
    graph = graph_builder.compile()
    return graph


def run_single_ablation(variant: str, sample_size: int | None) -> None:
    """Monkey-patch appropriate builders, run the extraction, restore originals."""
    logger.info(f"==== Running ablation: {variant} ====")

    # Store originals for restoration
    original_rag_graph_builder = rag_mod.graph_builder
    original_pdf_builder_func: Callable[[], StateGraph] = (
        pdf_mod.build_dpe_w_map_reduce_agent_k_graph
    )
    original_pdf_ns_rag_builder = getattr(pdf_mod, "graph_builder_v6", None)

    try:
        # Adjust sample size per user instruction (test with 2 first)
        config_experiment.PDF_EXTRACTION_SAMPLE_SIZE = sample_size

        # Patch for variants 1-4 + exclude_react_agent (rag graph)
        if variant == "exclude_fact_extraction_agent":
            rag_mod.graph_builder = build_rag_graph_exclude_fact_extraction_agent()
        elif variant == "exclude_code_interpreter":
            rag_mod.graph_builder = build_rag_graph_exclude_code_interpreter()
        elif variant == "exclude_self_reflection":
            rag_mod.graph_builder = build_rag_graph_exclude_self_reflection()
        elif variant == "exclude_self_consistency":
            rag_mod.graph_builder = build_rag_graph_exclude_self_consistency()
        elif variant == "exclude_react_agent":
            rag_mod.graph_builder = build_rag_graph_exclude_react_agent()

        # Keep pdf module's imported alias in sync if it exists
        if (
            variant
            in {
                "exclude_fact_extraction_agent",
                "exclude_code_interpreter",
                "exclude_self_reflection",
                "exclude_self_consistency",
                "exclude_react_agent",
            }
            and original_pdf_ns_rag_builder is not None
        ):
            pdf_mod.graph_builder = rag_mod.graph_builder  # type: ignore[assignment]

        # Patch for variant 5 (outer PDF graph; keep rag graph original)
        if variant == "exclude_global_validation_agent":

            def _patched_pdf_builder() -> StateGraph:
                return build_pdf_graph_without_global_validation_agent()

            pdf_mod.build_dpe_w_map_reduce_agent_k_graph = _patched_pdf_builder  # type: ignore[assignment]

        # Compose output filename
        out_dir = OUTPUT_DIR
        out_file = f"ablation_{variant}_{get_curr_ts()}.csv"

        # Run extraction using the existing function and configs
        df = pdf_mod.extract_from_pdfs(
            sample_size=config_experiment.PDF_EXTRACTION_SAMPLE_SIZE,
            method=config_experiment.PDF_EXTRACTION_METHOD.value,
            eval_type=config_experiment.PDF_EXTRACTION_EVAL_TYPE,
            output_dir=out_dir,
            output_filename=out_file,
        )
        logger.info(
            f"Ablation '{variant}' complete. Rows: {len(df)}. Output saved to {out_dir}/{out_file}"
        )
    finally:
        # Restore originals
        rag_mod.graph_builder = original_rag_graph_builder
        if original_pdf_ns_rag_builder is not None:
            pdf_mod.graph_builder = original_pdf_ns_rag_builder  # type: ignore[assignment]
        pdf_mod.build_dpe_w_map_reduce_agent_k_graph = original_pdf_builder_func  # type: ignore[assignment]


def main():
    parser = argparse.ArgumentParser(description="Run ablation analyses.")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help=(
            "Comma-separated list of variants to run: "
            "exclude_fact_extraction_agent,exclude_code_interpreter,"
            "exclude_self_reflection,exclude_self_consistency,"
            "exclude_global_validation_agent,exclude_react_agent or 'all'"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="none",
        help="Sample size for PDF extraction (integer or 'none' for all; default: all samples)",
    )
    args = parser.parse_args()

    available = AVAILABLE_VARIANTS

    if args.variants.strip().lower() == "all":
        to_run = available
    else:
        requested = [v.strip() for v in args.variants.split(",") if v.strip()]
        invalid = [v for v in requested if v not in available]
        if invalid:
            raise ValueError(f"Unknown variants: {invalid}. Available: {available}")
        to_run = requested

    # Ensure litellm drops unsupported params consistently (mirror usage in modules)
    litellm.drop_params = True

    # Parse sample size
    ss_raw = args.sample_size.strip().lower()
    if ss_raw == "none":
        sample_size = None
    else:
        sample_size = int(ss_raw)

    for variant in to_run:
        run_single_ablation(variant, sample_size=sample_size)


if __name__ == "__main__":
    main()
