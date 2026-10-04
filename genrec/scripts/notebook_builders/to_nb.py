"""Turn a '# %%' percent-format script into a notebook; with --run, execute it (outputs saved in the notebook).

    python scripts/notebook_builders/to_nb.py scripts/notebook_builders/sasrec_cells.py notebooks/movielens/sasrec.ipynb --run
    python ... tfrs_cells.py notebooks/movielens/tfrs.ipynb --run --kernel genrec-tf   # TensorFlow environment
    python ... colab_train_cells.py notebooks/movielens/train_qwen_colab.ipynb --colab   # not run locally

The notebook runs with its own folder as the working directory.
"""
import re
import sys
import uuid
from pathlib import Path

import nbformat
from nbclient import NotebookClient

KERNELS = {"python3": "Python 3 (genrec)", "genrec-tf": "Python (.venv-tf)"}


def build(src_path: str) -> nbformat.NotebookNode:
    cells = []
    for chunk in re.split(r"^# %%", Path(src_path).read_text(), flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
    nb = nbformat.v4.new_notebook(cells=cells)
    for c in nb.cells:
        c.id = uuid.uuid4().hex[:8]
    return nb


def main():
    src, dst = sys.argv[1], Path(sys.argv[2])
    kernel = sys.argv[sys.argv.index("--kernel") + 1] if "--kernel" in sys.argv else "python3"
    nb = build(src)
    nb.metadata = {"kernelspec": {"display_name": KERNELS.get(kernel, kernel), "language": "python", "name": kernel},
                   "language_info": {"name": "python"}}
    if "--colab" in sys.argv:   # Colab opens it with a GPU runtime selected
        nb.metadata.update({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}})
    if "--run" in sys.argv:
        NotebookClient(nb, timeout=None, kernel_name=kernel, resources={"metadata": {"path": str(dst.parent)}}).execute()
    nbformat.write(nb, dst)
    print("wrote", dst, len(nb.cells), "cells", "(executed)" if "--run" in sys.argv else "")


if __name__ == "__main__":
    main()
