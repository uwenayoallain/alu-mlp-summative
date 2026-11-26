"""Execute the notebook cell by cell, saving after each cell so outputs survive an interruption."""

import sys
import time

import nbformat
from nbclient import NotebookClient

# Usage: run_notebook.py [notebook] [first_cell]
# With first_cell, only the setup cell and cells from first_cell on run; earlier outputs are kept.
path = sys.argv[1] if len(sys.argv) > 1 else "notebook/coffee_leaf_disease.ipynb"
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
nb = nbformat.read(path, as_version=4)
setup = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code")
todo = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and (i >= first or i == setup)]
for i in todo:
    nb.cells[i].outputs, nb.cells[i].execution_count = [], None
client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "notebook"}})
start = time.time()
with client.setup_kernel():
    for i in todo:
        cell = nb.cells[i]
        if True:
            client.execute_cell(cell, i)
            nbformat.write(nb, path)
            print(f"cell {i} done at {time.time() - start:.0f}s", flush=True)
print("notebook finished", flush=True)
