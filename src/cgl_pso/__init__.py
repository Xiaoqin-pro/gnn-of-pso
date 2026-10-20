"""CGL-PSO: Contribution-Guided Graph Learning Particle Swarm Optimization.

An independent research route (branch ``feat/cgl-pso``): the social learning
relation between particles is an online, directed, sparse graph whose edge
weights are updated from *realised* search feedback. No GNN, no PPO, no offline
training -- three small, ablatable mechanisms on top of the standard PSO update
(reused unchanged from the verified engine in ``cgl_pso._infra.optim``).
"""

from .contribution import ContributionGraph, ContributionUpdate
from .candidates import CandidateSet, build_candidates, CANDIDATE_SOURCES
from .policy import CGL_POLICIES, CGLPolicy, make_cgl_policy

__all__ = [
    "ContributionGraph",
    "ContributionUpdate",
    "CandidateSet",
    "build_candidates",
    "CANDIDATE_SOURCES",
    "CGL_POLICIES",
    "CGLPolicy",
    "make_cgl_policy",
]

__version__ = "0.1.0"
