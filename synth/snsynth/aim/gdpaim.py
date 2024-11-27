import numpy as np
import pandas as pd

try:
    from mbi import FactoredInference, Dataset, Domain, GraphicalModel
except ImportError:
    print("Please install mbi with:\n   pip install git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca")

import itertools
from snsynth.base import Synthesizer
from mbi import Dataset, FactoredInference, Domain
from snsynth.utils import cdp_rho, exponential_mechanism, gaussian_noise, powerset
from scipy import sparse
from scipy.stats import norm


prng = np.random

from scipy.optimize import root_scalar
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



class Identity(sparse.linalg.LinearOperator):
    def __init__(self, n):
        self.shape = (n,n)
        self.dtype = np.float64
    def _matmat(self, X):
        return X
    def __matmul__(self, X):
        return X
    def _transpose(self):
        return self
    def _adjoint(self):
        return self

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2 ** 20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class gdpAIMSynthesizer(Synthesizer):
    """gdpAIM: An Adaptive and Iterative Mechanism

    :param epsilon: privacy budget for the synthesizer
    :type epsilon: float
    :param delta: privacy parameter.  Should be small, in the range of 1/(n * sqrt(n))
    :type delta: float
    :param verbose: print diagnostic information during processing
    :type verbose: bool

    Based on the code available in:
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py
    """

    def __init__(self, epsilon=None, delta=None, mu=None, max_model_size=80, degree=2, num_marginals=None, 
                 max_cells: int = 10000, rounds=None, verbose=False):

        if (mu is not None) and (epsilon is not None or delta is not None):
            raise ValueError("Provide either mu or (epsilon and delta), but not both.")
        if mu is None and (epsilon is None or delta is None):
            raise ValueError("Either provide mu or both epsilon and delta.")
        
        if mu is not None:
            self.mu = mu
        else:
            if isinstance(epsilon, int):
                epsilon = float(epsilon)
            # Compute mu from epsilon and delta
            self.mu = find_mu_for_eps_delta(epsilon, delta)

        self.rounds = rounds
        self.max_model_size = max_model_size
        self.max_cells = max_cells
        self.degree = degree
        self.num_marginals = num_marginals
        self.verbose = verbose
        self.epsilon = epsilon
        self.delta = delta
        self.synthesizer = None
        self.num_rows = None
        self.original_column_names = None

    def fit(
            self,
            data,
            *ignore,
            transformer=None,
            categorical_columns=[],
            ordinal_columns=[],
            continuous_columns=[],
            preprocessor_eps=0.0,
            nullable=False,
    ):
        if type(data) is pd.DataFrame:
            self.original_column_names = data.columns

        train_data = self._get_train_data(
            data,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
            nullable=nullable,
            preprocessor_eps=preprocessor_eps
        )

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        cards = self._transformer.cardinality
        if any(c is None for c in cards):
            raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

        dimensionality = np.prod(cards)
        if self.verbose:
            print(f"Fitting with {dimensionality} dimensions")
        
        print(self._transformer.output_width)
        colnames = ["col" + str(i) for i in range(self._transformer.output_width)]

        if len(cards) != len(colnames):
            raise ValueError("Cardinality and column names must be the same length.")

        domain = Domain(colnames, cards)
        self.num_rows = len(data)

        # self.rho = 0.0 if self.delta == 0.0 else cdp_rho(self.epsilon, self.delta)

        data = pd.DataFrame(train_data, columns=colnames)
        data = Dataset(df=data, domain=domain)
        workload = self.get_workload(
            data, degree=self.degree, max_cells=self.max_cells, num_marginals=self.num_marginals
        )

        self.gdpAIM(data, workload)

    def sample(self, samples=None):
        if samples is None:
            samples = self.num_rows
        data = self.synthesizer.synthetic_data(rows=samples)
        data_iter = [tuple([c for c in t[1:]]) for t in data.df.itertuples()]
        return self._transformer.inverse_transform(data_iter)

    @staticmethod
    def get_workload(data: Dataset, degree: int, max_cells: int, num_marginals: int = None):
        workload = list(itertools.combinations(data.domain, degree))
        workload = [cl for cl in workload if data.domain.size(cl) <= max_cells]
        if num_marginals is not None:
            workload = [workload[i] for i in prng.choice(len(workload), num_marginals, replace=False)]

        # workload = [(cl, 1.0) for cl in workload]
        return workload

    def _worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(sensitivity.values())  # if all weights are 0, could be a problem
        return exponential_mechanism(errors, eps, max_sensitivity)

    def gdpAIM(self, data, workload):
        rounds = self.rounds or 16 * len(data.domain)
        # workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        oneway = [cl for cl in candidates if len(cl) == 1]

        alpha = 0.9


        ## zCDP sigma,epsilon
        #sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        #epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        ## GDP sigma,epsilon
        sigma = np.sqrt(rounds/(alpha*self.mu**2))
        phi_term = norm.cdf(-np.sqrt((1-alpha)/rounds)*0.5*self.mu)
        epsilon = np.log(1/(phi_term) - 1)

        measurements = []
        print('Initial Sigma', sigma)
        #rho_used = len(oneway) * 0.5 / sigma ** 2
        mu_sq_used = len(oneway)/sigma ** 2
        
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))

        engine = FactoredInference(data.domain, iters=1000, warm_start=True)
        model = engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            # Budget annealing
            # old condition: self.rho - rho_used < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2)
            if self.mu **2 - mu_sq_used < 2* ( 1 / sigma**2 + (-2 * norm.ppf(1/(np.exp(epsilon)+1)))**2 ) :
                # Just use up whatever remaining budget there is for one last round
                remaining = self.mu **2 - mu_sq_used 
                #sigma = np.sqrt(1 / (2 * (alpha) * remaining))
                #epsilon = np.sqrt(8 * (1-alpha) * remaining)
                sigma = np.sqrt(1/(alpha*remaining))
                phi_term = norm.cdf(-np.sqrt((1-alpha))*0.5*np.sqrt(remaining))
                epsilon = np.log(1/(phi_term) - 1)
                terminate = True

            #rho_used += 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            mu_sq_used += 1 / sigma**2 + (-2 * norm.ppf(1/(np.exp(epsilon)+1)))**2

            #size_limit = self.max_model_size * rho_used / self.rho
            size_limit = self.max_model_size * mu_sq_used / self.mu**2

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self._worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = engine.estimate(measurements)
            w = model.project(cl).datavector()
            if self.verbose:
                print('Selected', cl, 'Size', n, 'Budget Used', mu_sq_used / self.mu**2)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                if self.verbose:
                    print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma / 2)
                sigma /= 2
                epsilon *= 2

        engine.iters = 2500
        model = engine.estimate(measurements)

        if self.verbose:
            print("Estimating marginals")

        self.synthesizer = model

    def get_errors(self, data: Dataset, workload):
        errors = []
        for proj, wgt in workload:
            X = data.project(proj).datavector()
            Y = self.synthesizer.project(proj).datavector()
            e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
            errors.append(e)
        print('Average Error: ', np.mean(errors))
        return errors
