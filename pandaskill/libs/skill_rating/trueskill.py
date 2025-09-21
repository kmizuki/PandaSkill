"""Interface for TrueSkill so that it can be integrate as OpenSkill"""

import numpy as np
from trueskill import rate


class TrueSkillRating:
    def __init__(self, mu, sigma) -> None:
        self.mu = mu
        self.sigma = sigma


class TrueSkill:
    def rate(self, teams, scores):
        ranks = np.argsort(-np.array(scores)).tolist()
        return rate(teams, ranks=ranks)

    def rating(self, mu, sigma):
        return TrueSkillRating(mu, sigma)
