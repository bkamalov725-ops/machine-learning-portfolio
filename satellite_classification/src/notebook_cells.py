"""Notebook cell serialization helpers, independent of earlier experiments."""
import ast
import textwrap


def markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": textwrap.dedent(source).strip() + "\n",
    }


def code(cell_id: str, source: str) -> dict:
    source = textwrap.dedent(source).strip() + "\n"
    ast.parse(source)
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "source": source, "execution_count": None, "outputs": [],
    }
