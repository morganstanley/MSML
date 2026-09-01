import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image  # only for type-hinting; not strictly required
import pickle

# Each item: (PIL.Image.Image, ground_truth, prediction)
BatchItem = Tuple[Image.Image, str, str]

def show_batch(cnt, batch: List[BatchItem], rows: int = 3, cols: int = 3) -> None:
    """
    Display a 4×4 grid (16 images) from a list of (image, actual, predicted) tuples.

    Parameters
    ----------
    batch : list[BatchItem]
        Must contain exactly rows × cols items.
    rows, cols : int
        Grid dimensions; defaults to 4 × 4.
    """
    # assert len(batch) == rows * cols, f"Expected {rows*cols} items, got {len(batch)}"

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, (img, ques, actual, pred) in zip(axes, batch):
        ax.imshow(img)
        ax.set_title(f"Q: {ques}\nGT: {actual}\nPred: {pred}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./img-output/mistakes-%s.png"%str(cnt))


with open('./img-output/llava-1.6-mistakes.pkl', "rb") as f:
    wrongs = pickle.load(f)

for cnt in range(len(wrongs)//9):
    show_batch(cnt, wrongs[cnt:9*(cnt+1)])  # Display the first 16 mistakes
