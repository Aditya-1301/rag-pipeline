import typer, sys, os, json, glob
from rich import print
from rich.table import Table
from raglib.config import settings
from raglib.chunker import chunk_text
from raglib.store import docstore, reset_storage
from raglib.retriever import Retriever
from raglib.answer import answer_question

app = typer.Typer(help="RAG Terminal MVP")

@app.command()
def init():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    print(f"[green]OK[/green] data dir: {settings.DATA_DIR}")
    print(f"Docstore: {settings.DOCSTORE_PATH}\nIndex: {settings.INDEX_PATH}")

@app.command()
def reset(confirm: bool = typer.Option(False, help="Confirm deletion")):
    if not confirm:
        print("[yellow]Add --confirm to proceed[/yellow]")
        raise typer.Exit(code=1)
    reset_storage()
    print("[green]Storage reset.[/green]")

@app.command()
def ingest(
    text: str = typer.Option(None, help="Raw text to ingest"),
    file: str = typer.Option(None, help="Path to a UTF-8 .txt/.md file"),
    glob_pat: str = typer.Option(None, "--glob", help="Glob of files to ingest (e.g., 'docs/**/*.txt')"),
    source: str = typer.Option("cli", help="Source label"),
):
    chunks = []
    if text:
        for c in chunk_text(text, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS):
            c["meta"] = {"source": source}
            chunks.append(c)
    if file:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        for c in chunk_text(content, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS):
            c["meta"] = {"source": file}
            chunks.append(c)
    if glob_pat:
        for path in glob.glob(glob_pat, recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                for c in chunk_text(content, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS):
                    c["meta"] = {"source": path}
                    chunks.append(c)
            except Exception as e:
                print(f"[red]Skip[/red] {path}: {e}")

    if not chunks:
        print("[red]Nothing to ingest.[/red]")
        raise typer.Exit(code=1)

    docstore.add(chunks)
    retr = Retriever()
    n = retr.add(chunks)
    print(f"[green]Ingested[/green] {n} chunks.")

@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(None, help="Override TOP_K"),
    llm: str = typer.Option(None, help="Use 'openai' to draft with OpenAI if key is set"),
    max_sources: int = typer.Option(5, help="Max sources to cite/show in the answer"),
):
    res = answer_question(question, top_k=top_k, llm=llm, max_sources=max_sources)
    print("\n[bold]Answer[/bold]:")
    print(res["answer"])

    tbl = Table(title="Sources")
    tbl.add_column("#", style="cyan", no_wrap=True)
    tbl.add_column("Source", style="magenta")
    tbl.add_column("Score", justify="right")
    for i, s in enumerate(res["sources"]):
        src = s.get("meta", {}).get("source", f"doc_{i}")
        score = f"{s.get('score', 0):.3f}"
        tbl.add_row(str(i+1), src, score)
    print()
    print(tbl)

if __name__ == "__main__":
    app()
