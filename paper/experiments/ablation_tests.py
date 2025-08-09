"""
Ablation analysis runner for Agentic RAG v6 and the PDF Fast & Slow pipeline.

Non-intrusive: builds ablated graph variants externally and monkey-patches
the imported modules during each ablation run, then restores originals.

Variants implemented:
- exclude_facts_extractor: skip facts extraction; pass retrieved docs to program_reasoner
- exclude_code_interpreter: skip Python execution; use LLM to compute final answer
- exclude_hallucination_checker: skip the hallucination grader; execute directly
- exclude_self_consistency: if looping, execute anyway instead of self-consistency
- exclude_global_eval_optimizer: bypass global slow validator/optimizer loop

Usage:
  # Run all variants with all samples
  uv run python paper/experiments/ablation_tests.py --variants all --sample-size none
  # Run all variants with sample size 2 (default)
  uv run python paper/experiments/ablation_tests.py --variants all --sample-size 2
  # Run specific variants
  uv run python paper/experiments/ablation_tests.py --variants exclude_facts_extractor,exclude_self_consistency
"""

from __future__ import annotations

import argparse
from typing import Callable, List

import litellm
from langgraph.graph import END, START, StateGraph

import agent_k.config.experiment_config as config_experiment

# Import modules to allow monkey-patching
import paper.experiments.agentic_rag_v6 as rag_mod
import paper.experiments.pdf_agent_fast_n_slow as pdf_mod
from agent_k.config.logger import logger
from agent_k.utils.general import get_curr_ts


def _join_documents(docs: List[str] | str) -> str:
    if isinstance(docs, str):
        return docs
    try:
        return "\n\n---\n\n".join(docs)
    except Exception:
        return str(docs)


def build_rag_graph_exclude_facts_extractor() -> StateGraph:
    """START -> retrieve -> bypass_extract -> program_reasoner -> check_hallucination
    -> conditional (execute | program_reasoner | self_consistency)
    -> self_consistency -> execute -> format_output -> END
    """

    def bypass_extract(state: dict):
        logger.info("---BYPASS EXTRACT: using retrieved docs as facts---")
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
    graph.add_node("bypass_extract", bypass_extract)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("check_hallucination", rag_mod.check_hallucination)
    graph.add_node("self_consistency", rag_mod.self_consistency)
    graph.add_node("execute", rag_mod.execute)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "bypass_extract")
    graph.add_edge("bypass_extract", "program_reasoner")
    graph.add_edge("program_reasoner", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        rag_mod.hallucination_router,
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
        "You are a careful assistant. Compute the numeric answer using the given facts "
        "and the question without running code. Return ONLY two XML tags: "
        "<reasoning>...</reasoning><answer>...</answer>."
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
    graph.add_node("extract", rag_mod.extract)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("check_hallucination", rag_mod.check_hallucination)
    graph.add_node("self_consistency", rag_mod.self_consistency)
    graph.add_node("execute_llm_only", execute_llm_only)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "program_reasoner")
    graph.add_edge("program_reasoner", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        rag_mod.hallucination_router,
        {
            "execute": "execute_llm_only",
            "program_reasoner": "program_reasoner",
            "self_consistency": "self_consistency",
        },
    )
    graph.add_edge("self_consistency", "execute_llm_only")
    graph.add_edge("execute_llm_only", END)
    return graph


def build_rag_graph_exclude_hallucination_checker() -> StateGraph:
    """START -> retrieve -> extract -> program_reasoner -> execute -> format_output -> END"""
    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("extract", rag_mod.extract)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("execute", rag_mod.execute)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "program_reasoner")
    graph.add_edge("program_reasoner", "execute")
    graph.add_edge("execute", "format_output")
    graph.add_edge("format_output", END)
    return graph


def build_rag_graph_exclude_self_consistency() -> StateGraph:
    """Map self-consistency route to execute instead of running self-consistency."""
    graph = StateGraph(rag_mod.GraphState)
    graph.add_node("retrieve", rag_mod.retrieve)
    graph.add_node("extract", rag_mod.extract)
    graph.add_node("program_reasoner", rag_mod.program_reasoner)
    graph.add_node("check_hallucination", rag_mod.check_hallucination)
    graph.add_node("execute", rag_mod.execute)
    graph.add_node("format_output", rag_mod.format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "program_reasoner")
    graph.add_edge("program_reasoner", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        rag_mod.hallucination_router,
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


def build_pdf_graph_without_eval_optimizer():
    """PDF F&S Agentic RAG graph without the global validation/optimizer loop.
    Equivalent to: schema_decompose -> map_slow_extraction_agent -> reduce -> extraction_synthesis -> END
    """
    graph_builder = StateGraph(pdf_mod.State)
    graph_builder.add_node("schema_decompose", pdf_mod.schema_decompose)
    graph_builder.add_node(
        "map_slow_extraction_agent", pdf_mod.map_slow_extraction_agent
    )
    graph_builder.add_node(
        "reduce_slow_extraction_results", pdf_mod.reduce_slow_extraction_results
    )
    graph_builder.add_node("extraction_synthesis", pdf_mod.extraction_synthesis)

    graph_builder.add_edge(START, "schema_decompose")
    graph_builder.add_conditional_edges(
        "schema_decompose",
        pdf_mod.fast_and_slow_route,
        ["map_slow_extraction_agent"],
    )
    graph_builder.add_edge(
        "map_slow_extraction_agent", "reduce_slow_extraction_results"
    )
    graph_builder.add_edge("reduce_slow_extraction_results", "extraction_synthesis")
    graph_builder.add_edge("extraction_synthesis", END)
    graph = graph_builder.compile()
    return graph


def run_single_ablation(variant: str, sample_size: int | None) -> None:
    """Monkey-patch appropriate builders, run the extraction, restore originals."""
    logger.info(f"==== Running ablation: {variant} ====")

    # Store originals for restoration
    original_rag_graph_builder = rag_mod.graph_builder_v6
    original_pdf_builder_func: Callable[[], StateGraph] = (
        pdf_mod.build_dpe_w_map_reduce_agentic_rag_graph
    )
    original_pdf_ns_rag_builder = getattr(pdf_mod, "graph_builder_v6", None)

    try:
        # Adjust sample size per user instruction (test with 2 first)
        config_experiment.PDF_EXTRACTION_SAMPLE_SIZE = sample_size

        # Patch for variants 1-4 (rag graph)
        if variant == "exclude_facts_extractor":
            rag_mod.graph_builder_v6 = build_rag_graph_exclude_facts_extractor()
        elif variant == "exclude_code_interpreter":
            rag_mod.graph_builder_v6 = build_rag_graph_exclude_code_interpreter()
        elif variant == "exclude_hallucination_checker":
            rag_mod.graph_builder_v6 = build_rag_graph_exclude_hallucination_checker()
        elif variant == "exclude_self_consistency":
            rag_mod.graph_builder_v6 = build_rag_graph_exclude_self_consistency()

        # Keep pdf module's imported alias in sync if it exists
        if (
            variant
            in {
                "exclude_facts_extractor",
                "exclude_code_interpreter",
                "exclude_hallucination_checker",
                "exclude_self_consistency",
            }
            and original_pdf_ns_rag_builder is not None
        ):
            pdf_mod.graph_builder_v6 = rag_mod.graph_builder_v6  # type: ignore[assignment]

        # Patch for variant 5 (outer PDF graph; keep rag graph original)
        if variant == "exclude_global_eval_optimizer":

            def _patched_pdf_builder() -> StateGraph:
                return build_pdf_graph_without_eval_optimizer()

            pdf_mod.build_dpe_w_map_reduce_agentic_rag_graph = _patched_pdf_builder  # type: ignore[assignment]

        # Compose output filename
        out_dir = "paper/data/experiments/ablation_tests"
        out_file = f"ablation_{variant}_{get_curr_ts()}.csv"

        # Run extraction using the existing function and configs
        df = pdf_mod.extract_from_inferlink_pdfs(
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
        rag_mod.graph_builder_v6 = original_rag_graph_builder
        if original_pdf_ns_rag_builder is not None:
            pdf_mod.graph_builder_v6 = original_pdf_ns_rag_builder  # type: ignore[assignment]
        pdf_mod.build_dpe_w_map_reduce_agentic_rag_graph = original_pdf_builder_func  # type: ignore[assignment]


def main():
    parser = argparse.ArgumentParser(description="Run ablation analyses.")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help=(
            "Comma-separated list of variants to run: "
            "exclude_facts_extractor,exclude_code_interpreter,"
            "exclude_hallucination_checker,exclude_self_consistency,"
            "exclude_global_eval_optimizer or 'all'"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="2",
        help="Sample size for PDF extraction (integer or 'none' for all; default: 2)",
    )
    args = parser.parse_args()

    available = [
        "exclude_facts_extractor",
        "exclude_code_interpreter",
        "exclude_hallucination_checker",
        "exclude_self_consistency",
        "exclude_global_eval_optimizer",
    ]

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
