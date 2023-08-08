import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

import os
import argparse
from tqdm import tqdm
import numpy as np

from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

NUM=10000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-speech", "-s", default="dev_clean_0_1.npy", type=str)
    parser.add_argument("--input-unit", "-u", default="../dev_clean.km/dev_clean.km_0_1.npy", type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    args = parser.parse_args()
    tsne_2D = TSNE(
        n_components=2,
        perplexity=50,
        init='pca',
        random_state=0,
        n_jobs=-1,
    )
    os.system(f"mkdir -p figure")
    speech_data = np.load(args.input_speech, mmap_mode="r")
    unit_data = np.load(args.input_unit, mmap_mode="r")
    labels=['Speech', 'Unit']

    speech_labels = np.array([0]).repeat(NUM)
    unit_labels = np.array([1]).repeat(NUM)

    print(f"| load speech frames {len(speech_data)}, unit frames {len(unit_data)}")

    for layer in range(7):
        data = np.concatenate([
            speech_data[:NUM, layer, :],
            unit_data[NUM:2*NUM, layer, :],
        ], axis=0)

        if os.path.exists(f"{args.output}_layer{layer}.npy"):
            X = np.load(f"{args.output}_layer{layer}.npy", mmap_mode="r")
            print(f"| load {args.output}_layer{layer}.npy, shape: {X.shape}")
        else:
            print(f"| fitting {args.output}_layer{layer}.npy ...")
            X = tsne_2D.fit_transform(data)
            np.save(f"{args.output}_layer{layer}.npy", X)
        
        XS = X[:NUM]
        XU = X[NUM:]
        Y = np.concatenate([speech_labels[:NUM], unit_labels[:NUM]], axis=0)

        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))
        # title="20k samples"
        # ax.set_title(title)

        ax.scatter(XS[:, 0], XS[:, 1], color='#2196F3', rasterized=True, alpha=0.75, s=1, label="Speech")
        ax.scatter(XU[:, 0], XU[:, 1], color='#FF00FF', rasterized=True, alpha=0.75, s=1, label="Unit")
        ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")
        # ax.legend(loc="upper right", frameon=False)



        plt.show()
        plt.savefig(f"{args.output}_layer{layer}.svg", dpi=500, format="svg", bbox_inches='tight', pad_inches=0) 
        # plt.savefig(f"{args.output}_layer{layer}.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0) 

if __name__ == "__main__":
    main()

