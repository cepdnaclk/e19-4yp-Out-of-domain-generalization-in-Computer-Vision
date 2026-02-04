"""
Visualization tools for evolutionary analysis.
Generates plots and charts from analyzed data.
"""

from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.figure import Figure

from biomedxpro.analysis.loader import EvolutionHistory
from biomedxpro.analysis.metrics import (
    ChampionAnalyzer,
    ConvergenceDetector,
    FitnessProgressionAnalyzer,
)


def plot_fitness_curves(
    history: EvolutionHistory,
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (14, 8),
    show_confidence: bool = True,
) -> Figure:
    """
    Plot fitness progression for all islands.

    Args:
        history: Evolution history
        output_path: If provided, saves the plot to this path
        figsize: Figure size (width, height)
        show_confidence: If True, shows mean ± std as shaded region

    Returns:
        matplotlib Figure object
    """
    analyzer = FitnessProgressionAnalyzer(history)
    progression_data = analyzer.compute_global_progression()

    fig, ax = plt.subplots(figsize=figsize)

    # Color map for islands
    colors = list(mcolors.TABLEAU_COLORS.values())

    for idx, (island, stats_list) in enumerate(progression_data.items()):
        if not stats_list:
            continue

        generations = [s.generation for s in stats_list]
        max_fitness = [s.max_fitness for s in stats_list]
        mean_fitness = [s.mean_fitness for s in stats_list]
        min_fitness = [s.min_fitness for s in stats_list]

        color = colors[idx % len(colors)]

        # Plot max fitness line
        ax.plot(
            generations,
            max_fitness,
            label=f"{island} (Max)",
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=0.9,
        )

        # Plot mean fitness with confidence band
        if show_confidence:
            ax.fill_between(
                generations,
                min_fitness,
                max_fitness,
                color=color,
                alpha=0.15,
            )
            ax.plot(
                generations,
                mean_fitness,
                color=color,
                linewidth=1,
                linestyle="--",
                alpha=0.6,
            )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (Inverted BCE)", fontsize=12)
    ax.set_title("Fitness Progression Across Islands", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved fitness curves to {output_path}")

    return fig


def plot_champion_timeline(
    history: EvolutionHistory,
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot timeline showing when each champion was born and their survival.

    Args:
        history: Evolution history
        output_path: If provided, saves the plot to this path
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """

    analyzer = ChampionAnalyzer(history)
    champions_info = analyzer.get_all_champions_info()

    if not champions_info:
        logger.warning("No champions found to plot")
        return plt.figure(figsize=figsize)

    fig, ax = plt.subplots(figsize=figsize)

    # Sort champions by birth generation
    sorted_champions = sorted(
        champions_info.items(), key=lambda x: x[1].generation_born
    )

    y_positions = list(range(len(sorted_champions)))
    colors = list(mcolors.TABLEAU_COLORS.values())

    for y_pos, (island, info) in zip(y_positions, sorted_champions):
        color = colors[y_pos % len(colors)]

        # Draw horizontal bar from birth to final generation
        ax.barh(
            y_pos,
            width=info.final_generation - info.generation_born,
            left=info.generation_born,
            height=0.6,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Mark birth generation with a star
        ax.scatter(
            info.generation_born,
            y_pos,
            marker="*",
            s=200,
            color=color,
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )

        # Add fitness label
        ax.text(
            info.final_generation + 0.5,
            y_pos,
            f" Fitness: {info.final_fitness:.3f}",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([island for island, _ in sorted_champions], fontsize=10)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_title("Champion Birth and Survival Timeline", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.invert_yaxis()

    # Add legend
    star_patch = mpatches.Patch(color="gray", label="★ Birth Generation")
    ax.legend(handles=[star_patch], loc="lower right", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved champion timeline to {output_path}")

    return fig


def plot_convergence_heatmap(
    history: EvolutionHistory,
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 8),
    plateau_threshold: int = 5,
) -> Figure:
    """
    Plot heatmap showing fitness values across islands and generations.
    Highlights convergence points.

    Args:
        history: Evolution history
        output_path: If provided, saves the plot to this path
        figsize: Figure size (width, height)
        plateau_threshold: Number of generations without improvement for convergence

    Returns:
        matplotlib Figure object
    """
    detector = ConvergenceDetector(history, plateau_threshold=plateau_threshold)
    convergence_info = detector.detect_all_convergence()

    fig_analyzer = FitnessProgressionAnalyzer(history)
    progression_data = fig_analyzer.compute_global_progression()

    fig, ax = plt.subplots(figsize=figsize)

    # Build matrix: islands x generations
    islands = sorted(history.islands)
    num_generations = history.num_generations

    fitness_matrix: list[list[float]] = []
    for island in islands:
        stats_list = progression_data.get(island, [])
        row: list[float] = [0.0] * num_generations
        for stats in stats_list:
            row[stats.generation] = stats.max_fitness
        fitness_matrix.append(row)

    # Plot heatmap
    im = ax.imshow(fitness_matrix, aspect="auto", cmap="viridis", interpolation="none")

    # Mark convergence points
    for idx, island in enumerate(islands):
        conv_info = convergence_info.get(island)
        if conv_info and conv_info.converged and conv_info.convergence_generation:
            ax.scatter(
                conv_info.convergence_generation,
                idx,
                marker="X",
                s=200,
                color="red",
                edgecolors="white",
                linewidths=2,
                zorder=3,
                label="Convergence Point" if idx == 0 else "",
            )

    ax.set_xticks(range(0, num_generations, max(1, num_generations // 10)))
    ax.set_yticks(range(len(islands)))
    ax.set_yticklabels(islands, fontsize=10)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Island", fontsize=12)
    ax.set_title(
        f"Fitness Heatmap (X = Converged after {plateau_threshold} gen plateau)",
        fontsize=14,
        fontweight="bold",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Max Fitness", fontsize=11)

    # Legend
    if any(info.converged for info in convergence_info.values()):
        ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved convergence heatmap to {output_path}")

    return fig


def plot_improvement_rates(
    history: EvolutionHistory,
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot bar chart comparing improvement rates across islands.

    Args:
        history: Evolution history
        output_path: If provided, saves the plot to this path
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    detector = ConvergenceDetector(history)
    convergence_info = detector.detect_all_convergence()

    islands = sorted(convergence_info.keys())
    improvement_rates = [
        convergence_info[island].improvement_rate for island in islands
    ]
    converged_status = [convergence_info[island].converged for island in islands]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["red" if converged else "green" for converged in converged_status]

    ax.bar(
        range(len(islands)),
        improvement_rates,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xticks(range(len(islands)))
    ax.set_xticklabels(islands, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Improvement Rate (Fitness/Generation)", fontsize=12)
    ax.set_title(
        "Average Fitness Improvement Rate by Island", fontsize=14, fontweight="bold"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Legend
    green_patch = mpatches.Patch(color="green", alpha=0.7, label="Still Improving")
    red_patch = mpatches.Patch(color="red", alpha=0.7, label="Converged")
    ax.legend(handles=[green_patch, red_patch], loc="best", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved improvement rates to {output_path}")

    return fig


def generate_all_plots(
    history: EvolutionHistory,
    output_dir: str | Path,
    plateau_threshold: int = 5,
) -> dict[str, Path]:
    """
    Generate all available plots and save to output directory.

    Args:
        history: Evolution history
        output_dir: Directory to save plots
        plateau_threshold: Convergence detection threshold

    Returns:
        Dictionary mapping plot_name -> file_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating all plots in {output_dir}")

    plots = {}

    # Fitness curves
    fitness_path = output_dir / "fitness_progression.png"
    plot_fitness_curves(history, output_path=fitness_path)
    plots["fitness_progression"] = fitness_path

    # Champion timeline
    timeline_path = output_dir / "champion_timeline.png"
    plot_champion_timeline(history, output_path=timeline_path)
    plots["champion_timeline"] = timeline_path

    # Convergence heatmap
    heatmap_path = output_dir / "convergence_heatmap.png"
    plot_convergence_heatmap(
        history, output_path=heatmap_path, plateau_threshold=plateau_threshold
    )
    plots["convergence_heatmap"] = heatmap_path

    # Improvement rates
    improvement_path = output_dir / "improvement_rates.png"
    plot_improvement_rates(history, output_path=improvement_path)
    plots["improvement_rates"] = improvement_path

    logger.success(f"Generated {len(plots)} plots in {output_dir}")

    return plots
