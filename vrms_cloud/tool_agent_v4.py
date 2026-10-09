"""Forced-classification policy: prefer learned combination to heuristic votes."""
from pathlib import Path
import argparse
from .tool_agent_v3 import run,SYSTEM as PREVIOUS_SYSTEM

DEFAULT_OUT=Path("outputs/vrms_agent_loso/v7_gpt_r1_20261008")
SYSTEM=PREVIOUS_SYSTEM+"""
Policy clarification for this FORCED-BINARY experiment:
Request corrective_evidence as your primary decision tool. Its raw_combined_class
is the trained classification estimate from the actual EEG tools. Use that
learned combination as the primary estimate; it learned score scaling and tool
dependence from internal data. A subjective count of correlated model classes
or an isolated auxiliary calibrated score must not replace that learned model.
The selected-margin suggestion is a selective-correction guard that can fall
back to the old MIL class. Do not treat that fallback as stronger class evidence
when the combined model is merely near .5. This task requires High/Low for all
paths; choose the best learned combined class and lower confidence when its
margin or nested audit is weak. Weak audit evidence remains a limitation and
must be stated, rather than converted into a claim of reliable publication.
Retain raw_combined_class unless a requested tool is missing/inconsistent or
independent direction-specific validation clearly supports a stronger competing
class. In the latter case name the supporting independent-validation measure;
do not override on physiological intuition, a single neighbour, or the old
uncalibrated MIL score. Never rescale the combined score or invent a probability.
Explain how the actual EEG tools support or conflict with your final class.
"""


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--probe",action="store_true")
    args=parser.parse_args();run(out=DEFAULT_OUT,probe=args.probe,system=SYSTEM)
