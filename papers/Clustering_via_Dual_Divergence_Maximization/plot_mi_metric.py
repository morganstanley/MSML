import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_mi_wrt_clusters(mi, c, font_size=11):

    unique_c, sizes_c = np.unique(c, return_counts=True)
    sort_idx = sizes_c.argsort()
    unique_c = unique_c[sort_idx]
    sizes_c = sizes_c[sort_idx]
    sort_idx = None

    mi_c = np.zeros((unique_c.size, unique_c.size))
    for i, c_i in enumerate(unique_c):
        idx_i = (c == c_i)
        for j, c_j in enumerate(unique_c):
            idx_j = (c == c_j)
            mi_c[i,j] = mi[idx_i, :][:, idx_j].mean()

    diag_idx = np.eye(unique_c.size, dtype=bool)

    sizes_c_matrix = np.outer(sizes_c, sizes_c)

    intra_cluster_score = np.nansum(mi_c[diag_idx]*sizes_c_matrix[diag_idx])/sizes_c_matrix[diag_idx].sum()
    inter_cluster_score = np.nansum(mi_c[~diag_idx]*sizes_c_matrix[~diag_idx])/sizes_c_matrix[~diag_idx].sum()

    if sizes_c.size < 8:
        plt.close()
        ax = sns.heatmap(
            mi_c,
            # vmax=min(1.0, mi_c.max()),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
        )
        ax.set_xticklabels(sizes_c, fontsize=font_size)
        ax.set_yticklabels(sizes_c, fontsize=font_size)
        ax.set_xlabel('Clusters in the order of sizes', fontsize=font_size)
        ax.set_ylabel('Clusters in the order of sizes', fontsize=font_size)
        plt.title(f'MI: IntraCluster={round(intra_cluster_score, 4)}, InterCluster{round(inter_cluster_score, 4)}', fontsize=font_size)
        plt.show()
    else:
        plt.close()
        # plt.imshow(mi_c, vmax=min(1.0, mi_c.max()))
        plt.imshow(mi_c)
        plt.rcParams.update({'font.size': font_size})
        plt.xticks(np.arange(sizes_c.size, dtype=np.int), sizes_c, fontsize=font_size)
        plt.yticks(np.arange(sizes_c.size, dtype=np.int), sizes_c, fontsize=font_size)
        plt.colorbar()
        plt.title(f'MI: IntraCluster={round(intra_cluster_score, 4)}, InterCluster={round(inter_cluster_score, 4)}')
        plt.show()
