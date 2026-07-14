from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, with_retrieval_mode
from .evaluate import evaluate_retrieval_case, load_retrieval_cases, summarize_results
from .index import index_documents, index_summary
from .pipeline import ask_question_with_config
from .providers import create_chat_client, create_embedding_client


app = typer.Typer(help="COMP9444 RAG assistant CLI")
console = Console()


@app.command()
def index(
    config_path: str = typer.Option("configs/rag.yaml", "--config", "-c"),
    force: bool = typer.Option(False, "--force", help="Re-index all manifest documents."),
) -> None:
    """索引新增或修改过的课程 PDF。"""
    config = load_config(config_path)
    embedding_client = create_embedding_client(config.embedding)
    result = index_documents(config, embedding_client, force=force)
    console.print(
        f"Indexed files: {result.indexed_files}; skipped files: {result.skipped_files}; chunks added: {result.chunks_added}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask over the COMP9444 knowledge base."),
    config_path: str = typer.Option("configs/rag.yaml", "--config", "-c"),
    top_k: int | None = typer.Option(None, "--top-k", help="Override top-k retrieval count."),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="Override vector, bm25, hybrid, or rerank."),
    log: bool = typer.Option(True, "--log/--no-log", help="Write query, answer, and retrieved context metadata to JSONL."),
) -> None:
    """基于已索引课件进行一次 RAG 问答。"""
    config = with_retrieval_mode(load_config(config_path), retrieval_mode)
    effective_top_k = top_k or config.top_k
    response = ask_question_with_config(config, question, top_k=effective_top_k, log=log)
    contexts = response.contexts
    if not contexts:
        console.print("[yellow]No retrieved contexts found. Run `python -m src.app index` first.[/yellow]")
        raise typer.Exit(code=1)

    console.print("\n[bold]Answer[/bold]")
    console.print(response.answer.answer or "[yellow]No answer returned.[/yellow]")

    table = Table(title="Retrieved citations")
    table.add_column("#")
    table.add_column("Citation")
    table.add_column("Retrieved by")
    table.add_column("Score")
    for i, item in enumerate(contexts, start=1):
        score = item.rerank_score if item.rerank_score is not None else item.fusion_score
        score = score if score is not None else item.distance
        score_text = f"{score:.4f}" if score is not None else "n/a"
        table.add_row(str(i), item.citation, "+".join(item.retrieval_sources), score_text)
    console.print(table)
    if response.log_path:
        console.print(f"[dim]Query log written to {response.log_path}[/dim]")


@app.command("status")
def status(config_path: str = typer.Option("configs/rag.yaml", "--config", "-c")) -> None:
    """显示当前索引状态。"""
    config = load_config(config_path)
    summary = index_summary(config)
    table = Table(title="Index status")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(str(key), str(value))
    console.print(table)


@app.command("eval-retrieval")
def eval_retrieval(
    cases_path: str = typer.Option("data/eval/retrieval_cases.yaml", "--cases"),
    config_path: str = typer.Option("configs/rag.yaml", "--config", "-c"),
    top_k: int = typer.Option(6, "--top-k"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="Override vector, bm25, hybrid, or rerank."),
    output_json: Path | None = typer.Option(None, "--output-json", help="Write reproducible metrics and per-case results."),
) -> None:
    """运行轻量检索回归测试，不调用最终回答生成。"""
    config = with_retrieval_mode(load_config(config_path), retrieval_mode)
    embedding_client = create_embedding_client(config.embedding)
    chat_client = create_chat_client(config.chat)
    cases = load_retrieval_cases(cases_path)

    results = [
        evaluate_retrieval_case(config, embedding_client, chat_client, case, top_k=top_k)
        for case in cases
    ]

    table = Table(title=f"Retrieval evaluation @ {top_k} ({config.retrieval_mode})")
    table.add_column("Case")
    table.add_column("Source hit")
    table.add_column("Page hit")
    table.add_column("RR")
    table.add_column("nDCG")
    table.add_column("Latency")
    table.add_column("Top citations")
    for result in results:
        top_citations = "\n".join(item.citation for item in result.results[:3])
        table.add_row(
            result.case.id,
            "yes" if result.source_hit else "no",
            "yes" if result.page_hit else "no",
            f"{result.reciprocal_rank:.3f}",
            f"{result.ndcg:.3f}",
            f"{result.latency_ms:.0f} ms",
            top_citations,
        )
    console.print(table)

    summary = summarize_results(results)
    console.print(
        f"Cases: {int(summary['cases'])}; "
        f"source_recall@{top_k}: {summary['source_recall']:.2f}; "
        f"page_recall@{top_k}: {summary['page_recall']:.2f}"
        f"; Hit@{top_k}: {summary['hit_rate']:.2f}"
        f"; MRR@{top_k}: {summary['mrr']:.3f}"
        f"; nDCG@{top_k}: {summary['ndcg']:.3f}"
        f"; avg latency: {summary['avg_latency_ms']:.0f} ms"
        f"; p95 latency: {summary['p95_latency_ms']:.0f} ms"
    )

    if output_json:
        report = {
            "retrieval_mode": config.retrieval_mode,
            "top_k": top_k,
            "summary": summary,
            "cases": [
                {
                    "id": result.case.id,
                    "retrieval_queries": result.retrieval_queries,
                    "source_hit": result.source_hit,
                    "page_hit": result.page_hit,
                    "reciprocal_rank": result.reciprocal_rank,
                    "ndcg": result.ndcg,
                    "latency_ms": result.latency_ms,
                    "citations": [item.citation for item in result.results],
                }
                for result in results
            ],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[dim]Evaluation report written to {output_json}[/dim]")


if __name__ == "__main__":
    app()
