"""Local grader for EnergyTool AI — mirror of MTS grader methodology.

Metrics match the online grader:
    MatchRate  = mean(score) across questions
    WinRate    = wins / (wins + losses)     # ties excluded
    Score      = 0.7 * MatchRate + 0.3 * WinRate

Judge uses 2-round A/B (baseline↔ours swap) to eliminate positional bias,
exactly like the MTS grader.
"""