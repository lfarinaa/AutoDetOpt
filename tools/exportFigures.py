"""Export the key figures of the executed notebook to docs/figures, for the README.

Run it from anywhere after executing v0/trackerOptimisation.ipynb (the figures are the notebook's own outputs):

    python tools/exportFigures.py

The committed notebook has no outputs (nbstripout), so this reads the working copy that you executed. Re-export and commit
the figures only when the results change in a way worth showing.
"""
import base64
import json
import sys
from pathlib import Path

repositoryRoot = Path(__file__).resolve().parent.parent
notebookPath = repositoryRoot / "v0" / "trackerOptimisation.ipynb"
outputDirectory = repositoryRoot / "docs" / "figures"

# File name, a string that identifies the code cell that draws the figure, and which of the cell's images to take.
figureSelection = (
    ("designViews.png", "designFigure = plt.figure", 0),
    ("restartOverview.png", "numberOfRestarts", 1),
    ("restartSummary.png", "numberOfRestarts", 0),
    ("performanceEvolution.png", "performanceFigure, axesGrid", 0),
)


def findImages(notebook, cellKey):
    """The PNG images (as bytes) in the outputs of the one code cell whose source contains cellKey."""
    matches = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cellKey in "".join(cell["source"])]
    if len(matches) != 1:
        raise SystemExit(f"expected one code cell with '{cellKey}', found {len(matches)}")
    images = []
    for output in matches[0].get("outputs", []):
        if "image/png" in output.get("data", {}):
            pngData = output["data"]["image/png"]
            images.append(base64.b64decode("".join(pngData) if isinstance(pngData, list) else pngData))
    return images


def main():
    notebook = json.loads(notebookPath.read_text(encoding="utf-8"))
    outputDirectory.mkdir(parents=True, exist_ok=True)
    for fileName, cellKey, imageNumber in figureSelection:
        images = findImages(notebook, cellKey)
        if len(images) <= imageNumber:
            raise SystemExit(f"{fileName}: the cell has {len(images)} images and no number {imageNumber}. Execute the notebook first.")
        (outputDirectory / fileName).write_bytes(images[imageNumber])
        print(f"wrote {outputDirectory.relative_to(repositoryRoot) / fileName}  ({len(images[imageNumber]) / 1e3:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
