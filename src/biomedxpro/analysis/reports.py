"""
Report generation for evolutionary analysis.
Export results in multiple formats: JSON, Markdown, HTML.
"""

import base64
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from biomedxpro.analysis.loader import EvolutionHistory
from biomedxpro.analysis.metrics import (
    ChampionAnalyzer,
    ConvergenceDetector,
    FitnessProgressionAnalyzer,
)
from biomedxpro.analysis.visualizations import (
    plot_champion_timeline,
    plot_convergence_heatmap,
    plot_fitness_curves,
    plot_improvement_rates,
)


@dataclass
class AnalysisReport:
    """Complete analysis report for an evolutionary run."""

    experiment_name: str
    num_generations: int
    num_islands: int
    total_individuals: int
    fitness_progression: dict[str, list[dict[str, Any]]]
    champions: dict[str, dict[str, Any]]
    convergence: dict[str, dict[str, Any]]
    earliest_champion: Optional[dict[str, Any]]
    deepest_lineage: Optional[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def to_json(self, output_path: Optional[str | Path] = None) -> str:
        """
        Export report as JSON.

        Args:
            output_path: If provided, saves to file

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
            logger.info(f"Saved JSON report to {output_path}")

        return json_str

    def to_markdown(self, output_path: Optional[str | Path] = None) -> str:
        """
        Export report as Markdown.

        Args:
            output_path: If provided, saves to file

        Returns:
            Markdown string
        """
        lines = [
            f"# Evolution Analysis Report: {self.experiment_name}",
            "",
            "## Overview",
            "",
            f"- **Generations**: {self.num_generations}",
            f"- **Islands**: {self.num_islands}",
            f"- **Total Individuals Evaluated**: {self.total_individuals}",
            "",
            "## Champions",
            "",
        ]

        # Champion summary table
        lines.append("| Island | Birth Gen | Final Fitness | Lineage Depth | Survival Time |")
        lines.append("|--------|-----------|---------------|---------------|---------------|")

        for island, champ_data in self.champions.items():
            lines.append(
                f"| {island} | {champ_data['generation_born']} | "
                f"{champ_data['final_fitness']:.4f} | "
                f"{champ_data['lineage_depth']} | "
                f"{champ_data['survival_time']} |"
            )

        lines.extend(["", "## Convergence Analysis", ""])

        # Convergence table
        lines.append("| Island | Converged | Conv. Generation | Plateau Length | Improvement Rate |")
        lines.append("|--------|-----------|------------------|----------------|------------------|")

        for island, conv_data in self.convergence.items():
            converged = "✓" if conv_data["converged"] else "✗"
            conv_gen = conv_data["convergence_generation"] or "N/A"
            lines.append(
                f"| {island} | {converged} | {conv_gen} | "
                f"{conv_data['plateau_length']} | "
                f"{conv_data['improvement_rate']:.6f} |"
            )

        lines.extend(["", "## Key Findings", ""])

        if self.earliest_champion:
            lines.append(
                f"- **Earliest Champion**: {self.earliest_champion['concept']} "
                f"(born in generation {self.earliest_champion['generation_born']})"
            )

        if self.deepest_lineage:
            lines.append(
                f"- **Deepest Lineage**: {self.deepest_lineage['concept']} "
                f"({self.deepest_lineage['lineage_depth']} ancestors)"
            )

        # Fitness progression summary
        lines.extend(["", "## Fitness Progression", ""])
        for island, stats_list in self.fitness_progression.items():
            if not stats_list:
                continue
            final_stats = stats_list[-1]
            lines.append(
                f"- **{island}**: Final fitness = {final_stats['max_fitness']:.4f} "
                f"(mean: {final_stats['mean_fitness']:.4f})"
            )

        markdown = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")
            logger.info(f"Saved Markdown report to {output_path}")

        return markdown

    def to_html(
        self,
        history: EvolutionHistory,
        output_path: Optional[str | Path] = None,
        plateau_threshold: int = 5,
    ) -> str:
        """
        Export report as HTML with embedded plots.

        Args:
            history: Evolution history (needed to regenerate plots)
            output_path: If provided, saves to file
            plateau_threshold: Convergence detection threshold

        Returns:
            HTML string
        """
        # Generate plots as base64-encoded images
        plots = {}

        for plot_name, plot_func, kwargs in [
            ("fitness", plot_fitness_curves, {}),
            ("timeline", plot_champion_timeline, {}),
            ("heatmap", plot_convergence_heatmap, {"plateau_threshold": plateau_threshold}),
            ("improvement", plot_improvement_rates, {}),
        ]:
            fig = plot_func(history, **kwargs)  # type: ignore[operator]
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plots[plot_name] = img_base64
            buf.close()

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evolution Analysis: {self.experiment_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ font-weight: bold; color: #2980b9; }}
        .plot {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .overview-card {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        .overview-card h3 {{
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .overview-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Evolution Analysis Report</h1>
        <h2>Experiment: {self.experiment_name}</h2>

        <div class="overview-grid">
            <div class="overview-card">
                <h3>Generations</h3>
                <div class="value">{self.num_generations}</div>
            </div>
            <div class="overview-card">
                <h3>Islands</h3>
                <div class="value">{self.num_islands}</div>
            </div>
            <div class="overview-card">
                <h3>Individuals</h3>
                <div class="value">{self.total_individuals}</div>
            </div>
        </div>

        <h2>📊 Fitness Progression</h2>
        <div class="plot">
            <img src="data:image/png;base64,{plots['fitness']}" alt="Fitness Curves">
        </div>

        <h2>🏆 Champions</h2>
        <table>
            <tr>
                <th>Island</th>
                <th>Birth Gen</th>
                <th>Final Fitness</th>
                <th>Lineage Depth</th>
                <th>Survival Time</th>
            </tr>
"""

        for island, champ_data in self.champions.items():
            html += f"""
            <tr>
                <td>{island}</td>
                <td>{champ_data['generation_born']}</td>
                <td>{champ_data['final_fitness']:.4f}</td>
                <td>{champ_data['lineage_depth']}</td>
                <td>{champ_data['survival_time']}</td>
            </tr>
"""

        html += """
        </table>

        <div class="plot">
            <img src="data:image/png;base64,""" + plots['timeline'] + """" alt="Champion Timeline">
        </div>

        <h2>🎯 Convergence Analysis</h2>
        <table>
            <tr>
                <th>Island</th>
                <th>Converged</th>
                <th>Conv. Generation</th>
                <th>Plateau Length</th>
                <th>Improvement Rate</th>
            </tr>
"""

        for island, conv_data in self.convergence.items():
            converged = "✓" if conv_data["converged"] else "✗"
            conv_gen = conv_data["convergence_generation"] or "N/A"
            html += f"""
            <tr>
                <td>{island}</td>
                <td>{converged}</td>
                <td>{conv_gen}</td>
                <td>{conv_data['plateau_length']}</td>
                <td>{conv_data['improvement_rate']:.6f}</td>
            </tr>
"""

        html += """
        </table>

        <div class="plot">
            <img src="data:image/png;base64,""" + plots['heatmap'] + """" alt="Convergence Heatmap">
        </div>

        <div class="plot">
            <img src="data:image/png;base64,""" + plots['improvement'] + """" alt="Improvement Rates">
        </div>

        <h2>🔍 Key Findings</h2>
        <ul>
"""

        if self.earliest_champion:
            html += f"""
            <li><strong>Earliest Champion:</strong> {self.earliest_champion['concept']} 
                (born in generation {self.earliest_champion['generation_born']})</li>
"""

        if self.deepest_lineage:
            html += f"""
            <li><strong>Deepest Lineage:</strong> {self.deepest_lineage['concept']} 
                ({self.deepest_lineage['lineage_depth']} ancestors)</li>
"""

        html += """
        </ul>
    </div>
</body>
</html>
"""

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info(f"Saved HTML report to {output_path}")

        return html


def generate_report(
    history: EvolutionHistory,
    plateau_threshold: int = 5,
) -> AnalysisReport:
    """
    Generate complete analysis report from history.

    Args:
        history: Evolution history
        plateau_threshold: Convergence detection threshold

    Returns:
        AnalysisReport object
    """
    logger.info("Generating analysis report...")

    # Fitness progression
    fitness_analyzer = FitnessProgressionAnalyzer(history)
    progression_data = fitness_analyzer.compute_global_progression()

    # Convert FitnessStats to dicts
    fitness_progression = {
        island: [asdict(stats) for stats in stats_list]
        for island, stats_list in progression_data.items()
    }

    # Champions
    champion_analyzer = ChampionAnalyzer(history)
    champions_info = champion_analyzer.get_all_champions_info()
    champions = {island: asdict(info) for island, info in champions_info.items()}

    # Convergence
    convergence_detector = ConvergenceDetector(history, plateau_threshold=plateau_threshold)
    convergence_info = convergence_detector.detect_all_convergence()
    convergence = {island: asdict(info) for island, info in convergence_info.items()}

    # Special champions
    earliest = champion_analyzer.find_earliest_champion()
    deepest = champion_analyzer.find_deepest_lineage()

    report = AnalysisReport(
        experiment_name=history.log_path.stem,
        num_generations=history.num_generations,
        num_islands=history.num_islands,
        total_individuals=len(history.records),
        fitness_progression=fitness_progression,
        champions=champions,
        convergence=convergence,
        earliest_champion=asdict(earliest) if earliest else None,
        deepest_lineage=asdict(deepest) if deepest else None,
    )

    logger.success("Analysis report generated")
    return report
