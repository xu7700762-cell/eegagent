"""Unsigned, traceable physiological descriptions of actually requested tools."""
from __future__ import annotations

import math


def physiological_evidence(biomarker=None, covariance=None, reference_quality="unavailable", p_cal=None):
    """Describe change without assigning a universal symptom direction.

    Tool probabilities are auxiliary model outputs. They are not validated
    directional evidence unless an independent validation artifact is supplied.
    No band-power or unsigned covariance heuristic produces a class vote here.
    """
    if reference_quality not in ("valid", "invalid", "unavailable"):
        raise ValueError("Unknown reference quality")
    result = dict(biomarker=None, covariance=None,
                  verification=dict(status="inconclusive", conflict_with_deep_model=None,
                                    direction_validated=False,
                                    explanation="Descriptors alone do not establish high/low direction"))
    if biomarker is not None:
        powers = biomarker.get("relative_band_power")
        changes = biomarker.get("reference_log_change") if reference_quality == "valid" else None
        if changes is not None and (len(changes) != 4 or not all(math.isfinite(v) for v in changes)):
            raise ValueError("Expected finite delta/theta/alpha/beta log changes")
        result["biomarker"] = dict(relative_band_power=powers,
            reference_log_change=changes, delta_change=None if changes is None else float(changes[0]),
            theta_change=None if changes is None else float(changes[1]),
            alpha_change=None if changes is None else float(changes[2]),
            beta_change=None if changes is None else float(changes[3]),
            change_unit="natural_log_power_ratio_to_initial_pre_task_rest",
            reference_quality=reference_quality,
            interpretation="spectral_change_observed" if changes is not None else "absolute_spectrum_only",
            direction_validated=False,
            provenance=biomarker.get("provenance", "mean_of_QC_accepted_5s_windows; Welch 256Hz"))
    if covariance is not None:
        distance = covariance.get("distance") if reference_quality == "valid" else None
        if distance is not None and (not math.isfinite(distance) or distance < 0):
            raise ValueError("Covariance distance must be finite and nonnegative")
        result["covariance"] = dict(distance=distance, reference_quality=reference_quality,
            interpretation="deviation_from_reference" if distance is not None else "reference_unavailable",
            distance_definition="mean Frobenius norm of log reference-whitened window covariance",
            direction_validated=False, provenance=covariance.get("provenance", "initial_pre_task_reference"))
    return result
