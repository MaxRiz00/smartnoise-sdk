from scipy.optimize import root_scalar
from scipy.stats import norm
import numpy as np

def delta_eps_mu(*, eps: float, mu: float) -> float:
    """
    Compute dual between mu-GDP and (epsilon, delta)-DP.
    Args:
        eps: eps
        mu: mu
    """
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)

def find_mu_for_eps_delta(eps: float, delta: float) -> float:
    """
    Find mu value satisfying (ε,δ)-DP constraints using binary search.
    """
    def objective(mu):
        return delta_eps_mu(eps=eps, mu=mu) - delta
    
    # Search in reasonable range for mu
    result = root_scalar(objective, 
                        bracket=[0.1, 100.0],
                        method='brentq')
    
    return result.root


