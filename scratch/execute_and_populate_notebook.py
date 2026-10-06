# -*- coding: utf-8 -*-
"""
Executes cells in FA_EFER_ResNet18_VGGFace2.ipynb and embeds
all text outputs and base64 matplotlib plots directly into the notebook.
"""

import os
import sys
import io
import json
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_and_populate():
    nb_path = r"d:\Research4\FA_EFER_ResNet18_VGGFace2.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_env = {
        "__name__": "__main__",
        "__file__": "FA_EFER_ResNet18_VGGFace2.ipynb"
    }

    exec_counter = 1

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code = "".join(cell["source"])
        print(f"\n[EXECUTING CELL {exec_counter}] ...")

        # Hook into matplotlib to capture figures
        captured_figs = []
        original_show = plt.show

        def custom_show(*args, **kwargs):
            fig_nums = plt.get_fignums()
            for num in fig_nums:
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                b64_str = base64.b64encode(buf.read()).decode("utf-8")
                captured_figs.append(b64_str)
            plt.close("all")

        plt.show = custom_show

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, global_env)
        except Exception as e:
            print(f"Error in cell {i}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.show = original_show

        # Collect outputs
        cell_outputs = []

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            lines = [l + "\n" for l in stdout_text.splitlines()]
            cell_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": lines
            })
            print(f"  Captured stdout: {len(lines)} lines")

        stderr_text = stderr_buf.getvalue()
        if stderr_text and "UserWarning" not in stderr_text:
            lines = [l + "\n" for l in stderr_text.splitlines()]
            cell_outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": lines
            })

        for b64 in captured_figs:
            cell_outputs.append({
                "data": {
                    "image/png": b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            print(f"  Captured plot figure (base64 size: {len(b64)} bytes)")

        cell["outputs"] = cell_outputs
        cell["execution_count"] = exec_counter
        exec_counter += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[SUCCESS] Notebook execution complete! Saved all outputs into {nb_path}")

if __name__ == "__main__":
    run_and_populate()
