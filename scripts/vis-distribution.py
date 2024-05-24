import argparse
import collections
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

import numpy as np

import gff.constants


def parse_args(argv):
    parser = argparse.ArgumentParser("")

    parser.add_argument("data_path", type=Path)

    return parser.parse_args(argv)


def mk_bar_graph(index, clim_distrs, labels, xlabel, fnames, title=None, ylabel=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=300)

    pos = np.linspace(-0.15, 0.15, len(clim_distrs))
    width = 0.6 / len(clim_distrs)

    for i, distrs in enumerate(clim_distrs):
        bottoms = np.zeros_like(distrs[0])
        for j, stack in enumerate(distrs):
            ax.bar(
                index + pos[i],
                stack,
                label=labels[i][j],
                width=width,
                bottom=bottoms,
                align="center",
            )
            bottoms += stack
    ax.set_xticks(index)
    ax.set_xlim([0.5, len(index) + 0.5])
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    for fname in fnames:
        fig.savefig(fname)
    plt.close(fig)


def main(args):
    with open(args.data_path / "measured_distribution.json") as f:
        distr = json.load(f)

    overall_cont = {}
    overall_clim = {}
    for k in distr:
        overall_clim[k] = collections.defaultdict(lambda: 0)
        overall_cont[k] = collections.defaultdict(lambda: 0)
        for continent, clim_counts in distr[k].items():
            if k == "floods":
                clim_counts = clim_counts["zones"]
            for clim_zone, count in clim_counts.items():
                overall_cont[k][continent] += count
                overall_clim[k][clim_zone] += count

    cont_as_np = {}
    clim_as_np = {}
    for k in overall_clim:
        cont_as_np[k] = np.zeros(len(gff.constants.HYDROATLAS_CONTINENT_NAMES))
        for i, z in enumerate(gff.constants.HYDROATLAS_CONTINENT_NAMES):
            cont_as_np[k][i] = overall_cont[k][str(z)]
        clim_as_np[k] = np.zeros(len(gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES))
        for i, z in enumerate(gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES):
            clim_as_np[k][i] = overall_clim[k][str(z)]

    mk_bar_graph(
        np.array(list(gff.constants.HYDROATLAS_CONTINENT_NAMES)),
        [[cont_as_np["expected"]], [cont_as_np["created"], cont_as_np["negatives"] / 2]],
        [
            ["Expected distribution"],
            ["Generated distribution (flood)", "Generated distribution (no-flood)"],
        ],
        xlabel="Continent",
        fnames=["vis/cont-distr.png", "vis/cont-distr.eps"],
        title="Continent distribution",
    )
    mk_bar_graph(
        np.array(list(gff.constants.HYDROATLAS_CLIMATE_ZONE_NAMES)),
        [[clim_as_np["expected"]], [clim_as_np["created"], clim_as_np["negatives"] / 2]],
        [
            ["Expected distribution"],
            ["Generated distribution (flood)", "Generated distribution (no-flood)"],
        ],
        xlabel="Climate zone",
        fnames=["vis/clim-zone-distr.png", "vis/clim-zone-distr.eps"],
        title="Climate zone distribution",
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
