import nbformat
from pathlib import Path


def update_hpo_output_dir_in_notebook(notebook_path: Path) -> None:
    nb = nbformat.read(str(notebook_path), as_version=4)
    changed = False

    old_block = (
        'HPO_OUTPUT_DIR = ROOT_DIR / "outputs" / "hpo"\n'
        'HPO_OUTPUT_DIR = validate_path_before_mkdir(HPO_OUTPUT_DIR, context="directory")\n'
        "HPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    )

    new_block = (
        "from orchestration.paths import resolve_output_path\n"
        "\n"
        "# Use centralized HPO root from paths.yaml (respects env_overrides / storage_env)\n"
        'HPO_ROOT = resolve_output_path(ROOT_DIR, CONFIG_DIR, "hpo")\n'
        "\n"
        "# Keep fold_splits as a study-level meta artifact, not mixed with trials\n"
        'HPO_META_DIR = validate_path_before_mkdir(HPO_ROOT / "_meta", context="directory")\n'
        "HPO_META_DIR.mkdir(parents=True, exist_ok=True)\n"
    )

    for cell in nb.cells:
        if cell.get("cell_type") == "code" and "HPO_OUTPUT_DIR = ROOT_DIR / \"outputs\" / \"hpo\"" in "".join(
            cell.get("source", "")
        ):
            source = "".join(cell.get("source", ""))
            if old_block in source:
                cell["source"] = source.replace(old_block, new_block)
                changed = True

    if changed:
        nbformat.write(nb, str(notebook_path))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    notebook = root / "notebooks" / "01_orchestrate_training_colab.ipynb"
    if notebook.exists():
        update_hpo_output_dir_in_notebook(notebook)


if __name__ == "__main__":
    main()



