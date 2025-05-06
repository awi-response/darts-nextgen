"""Script to create API docs.

For the public API documentation, we use this script to create for every public facing funtion a markdown file,
containing their documentation via mkdocsstrings.

We could have used [mkdocs-api-autonav](https://github.com/tlambert03/mkdocs-api-autonav) or
[mkdocs-autoapi](https://github.com/jcayers20/mkdocs-autoapi) for this usecase.
However, we have quite a complicate code structure because of our workspace setup and we also don't follow the
best-practices concerning public facing APIs.
Therefore, we manually create these files before buildtime of the docs without any further automation or linking done.
This script will generate the markdown files and a cfg-string which can be copy-pasted to the mkdocs.yml: nav section.
"""

# %%

import importlib
from collections import defaultdict
from inspect import getmembers, isclass, isfunction
from pathlib import Path
from typing import Literal

import pyperclip
from rich import traceback

traceback.install(
    show_locals=True, locals_hide_dunder=True, locals_hide_sunder=True
)  # Change to False if you encounter too large tracebacks


# %%


top_level_modules = [
    "darts",  # -> DARTS
    "darts_acquisition",
    "darts_ensemble",
    "darts_export",
    "darts_postprocessing",
    "darts_preprocessing",
    "darts_segmentation",
    "darts_utils",
    # "darts_superresolution",
]
second_level_modules = [
    "darts.pipelines",  # -> DARTS Pipelines
    "darts.legacy_training",  # -> DARTS Legacy Training
    "darts_segmentation.training",
    "darts_segmentation.metrics",
]
util_modules = {
    "darts_utils.cuda": ["free_cupy", "free_torch"],
    "darts_utils.rich": ["RichManagerSingleton"],
    "darts_utils.namegen": ["generate_id", "generate_name", "generate_counted_name"],
}


# %%


docfilecontent = Literal["class", "function"]
DocFileInfo = tuple[Path, docfilecontent]
DocFileInfos = dict[str, DocFileInfo]


def _generate_module_docfiles(modulename: str, parent_docpath: Path, filter: list[str] | None = None) -> DocFileInfos:
    module = importlib.import_module(modulename)
    module_docpath = parent_docpath / modulename.split(".")[-1]
    module_docpath.mkdir(exist_ok=True, parents=True)
    # module_header_name = modulename.replace(".", ": ").replace("_", " ").title().replace("Darts", "DARTS")

    toc = []
    docfileinfos: DocFileInfos = {}
    for funcname, _ in getmembers(module, isfunction):
        if filter and funcname not in filter:
            continue
        function_docfile = module_docpath / f"{funcname}.md"
        with function_docfile.open("w") as f:
            f.write("---\n")
            f.write("hide:\n")
            f.write("  - toc\n")
            f.write("---\n")
            f.write(f"# <code class='doc-symbol doc-symbol-nav doc-symbol-function'></code>{modulename}.{funcname}\n\n")
            f.write(f"::: {modulename}.{funcname}\n")
            f.write("    options:\n")
            f.write("      show_root_heading: false\n")
        toc_entry = f"- [{funcname}]({funcname}.md)\n"
        toc.append(toc_entry)
        docfileinfos[funcname] = (function_docfile, "function")
    for classname, _ in getmembers(module, isclass):
        if filter and classname not in filter:
            continue
        class_docfile = module_docpath / f"{classname}.md"
        with class_docfile.open("w") as f:
            f.write("---\n")
            f.write("hide:\n")
            f.write("  - toc\n")
            f.write("---\n")
            f.write(f"# <code class='doc-symbol doc-symbol-nav doc-symbol-class'></code>{modulename}.{classname}\n\n")
            f.write(f"::: {modulename}.{classname}\n")
            f.write("    options:\n")
            f.write("      show_root_heading: false\n")
        toc_entry = f"- [{classname}]({classname}.md)\n"
        toc.append(toc_entry)
        docfileinfos[classname] = (class_docfile, "class")

    module_docfile = module_docpath / "index.md"
    with open(module_docfile, "w") as f:
        f.write(f"# <code class='doc-symbol doc-symbol-nav doc-symbol-module'></code>{modulename}\n\n")
        # f.write(f"# {module_header_name} Reference\n\n")
        f.write(f"\n::: {modulename}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: false\n")
        f.write("      summary:\n")
        f.write("        attributes: true\n")
        f.write("        functions: true\n")
        f.write("        classes: true\n")
        f.write("        modules: false\n")
        f.write("      members: true\n")
        f.writelines(toc)

    return docfileinfos


reference_docpath = Path("../docs/reference")
reference_docpath.mkdir(exist_ok=True)
docfileinfos: dict[str, dict[str, DocFileInfo | DocFileInfos]] = defaultdict(dict)

for slm in second_level_modules:
    parent, child = slm.split(".")
    docfileinfos[parent][child] = _generate_module_docfiles(slm, reference_docpath / parent)

for tlm in top_level_modules:
    docfileinfos[tlm].update(_generate_module_docfiles(tlm, reference_docpath))

for um, umfilter in util_modules.items():
    docfileinfos["darts_utils"][um.split(".")[1]] = _generate_module_docfiles(
        um, reference_docpath / "darts_utils", umfilter
    )

len(docfileinfos)


# %%


# Create an API Reference overview file
with open(reference_docpath / "index.md", "w") as f:
    f.write("# DARTS API Reference\n\n")

    for tlm in sorted(docfileinfos.keys()):
        tlm_docfileinfos = docfileinfos[tlm]
        tlm_header_name = tlm.replace(".", ": ").replace("_", " ").title().replace("Darts", "DARTS")
        f.write(f"- [{tlm_header_name}](./{tlm}/index.md)\n")
        for slm, slm_docfileinfos in tlm_docfileinfos.items():
            if isinstance(slm_docfileinfos, dict):
                slm_header_name = slm.replace(".", ": ").replace("_", " ").title().replace("Darts", "DARTS")
                f.write(f"  - [{slm_header_name}](./{tlm}/{slm}/index.md)\n")


# %%


navstring = "\t- API Reference:\n"
navstring += "\t\t- reference/index.md\n"
for tlm in sorted(docfileinfos.keys()):
    tlm_docfileinfos = docfileinfos[tlm]
    tlm_header = f"<code class='doc-symbol doc-symbol-nav doc-symbol-module'></code>{tlm}"
    navstring += f"\t\t- {tlm_header}:\n"
    navstring += f"\t\t\t- reference/{tlm}/index.md\n"
    for slm in sorted(tlm_docfileinfos.keys()):
        slm_docfileinfos = tlm_docfileinfos[slm]
        if isinstance(slm_docfileinfos, dict):
            slm_header = f"<code class='doc-symbol doc-symbol-nav doc-symbol-module'></code>{slm}"
            navstring += f"\t\t\t- {slm_header}:\n"
            navstring += f"\t\t\t\t- reference/{tlm}/{slm}/index.md\n"
            for content_name, (content_docfile, content_type) in slm_docfileinfos.items():
                html_code_class = "doc-symbol-class" if content_type == "class" else "doc-symbol-function"
                content_header = f"<code class='doc-symbol doc-symbol-nav {html_code_class}'></code>{content_name}"
                navstring += f"\t\t\t\t- {content_header}: reference/{tlm}/{slm}/{content_docfile.name}\n"

        elif isinstance(slm_docfileinfos, tuple):
            content_name, (content_docfile, content_type) = slm, slm_docfileinfos
            html_code_class = "doc-symbol-class" if content_type == "class" else "doc-symbol-function"
            content_header = f"<code class='doc-symbol doc-symbol-nav {html_code_class}'></code>{content_name}"
            navstring += f"\t\t\t- {content_header}: reference/{tlm}/{content_docfile.name}\n"
navstring = navstring.replace("\t", "    ")
pyperclip.copy(navstring)
# Now the necessary string was copied to the clipboard.
# Replace the "API Reference" section with this string.
