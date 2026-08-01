"""DocQWise CLI."""
import click
from docqwise._version import __version__

@click.group()
@click.version_option(__version__, prog_name="docqwise")
def app():
    """DocQWise: Read. Extract. Retrieve."""

@app.command()
def info():
    """Show system info."""
    click.echo(f"DocQWise v{__version__}")
    click.echo("Read. Extract. Retrieve.")
    click.echo("https://github.com/VK-Ant/docqwise")

@app.command()
@click.argument("source")
@click.option("--workers", default=1, help="Number of CPU workers")
@click.option("--progress/--no-progress", default=True)
def ingest(source, workers, progress):
    """Ingest documents from a source."""
    from docqwise import Docqwise
    dq = Docqwise(workers=workers)
    click.echo(f"Ingesting from: {source}")
    dq.ingest(source)

@app.command()
@click.argument("source")
@click.option("--template", default=None, help="Extraction template name")
@click.option("--tables", is_flag=True, help="Extract tables only")
def extract(source, template, tables):
    """Extract fields or tables from a document."""
    from docqwise import Docqwise
    dq = Docqwise()
    if tables:
        result = dq.extract_tables(source)
    else:
        result = dq.extract_fields(source, template=template)
    click.echo(result)

@app.command()
@click.argument("query")
@click.option("--top-k", default=5)
def query(query, top_k):
    """Semantic search across ingested documents."""
    from docqwise import Docqwise
    dq = Docqwise()
    results = dq.retrieve(query, top_k=top_k)
    click.echo(results)

@app.command()
@click.argument("question")
@click.option("--source", default=None)
def ask(question, source):
    """Ask a question about your documents."""
    from docqwise import Docqwise
    dq = Docqwise()
    answer = dq.ask(question, source=source)
    click.echo(answer)

@app.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.option("--workers", default=4)
def serve(host, port, workers):
    """Start REST API server."""
    click.echo(f"Starting DocQWise server on {host}:{port}")
    click.echo("REST API server coming in v0.1.0")

@app.command()
@click.option("--port", default=8080)
def mcp(port):
    """Start MCP server."""
    click.echo(f"Starting DocQWise MCP server on port {port}")
    click.echo("MCP server coming in v0.1.0")

if __name__ == "__main__":
    app()
