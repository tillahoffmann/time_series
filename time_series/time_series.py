import collections
import contextlib
import itertools as it
from time import time

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import sklearn.cluster
import sklearn.mixture
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import cm

import bayespy as bp
from bayespy.inference.vmp import transformations


def get_best_permutation(A, z, method='average', plot_dendrogram=False):
    """
    Get the best permutation of indices minimising the squared distance between clusters in the
    observation model (only useful for visualisation).
    """
    num_nodes, num_groups = z.shape
    z = np.argmax(z, axis=1)

    distance = spd.cdist(A, A) ** 2

    aggregated = np.zeros((num_groups, num_groups))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if z[i] != z[j]:
                aggregated[z[i], z[j]] += distance[i, j]

    aggregated += aggregated.T

    linkage = sch.linkage(spd.squareform(np.sqrt(aggregated)), method=method)
    dendrogram = sch.dendrogram(linkage, no_plot=not plot_dendrogram)
    leaves = np.asarray(dendrogram['leaves'])

    return np.concatenate([np.where(l == z)[0] for l in leaves])


def label_axes(axes, x=0.05, y=0.95, va='top', offset=0, **kwargs):
    """
    Attach alphabetical labels to a sequence of axes.
    """
    for i, ax in enumerate(np.ravel(axes)):
        char = bytes([int.from_bytes(b'a', 'little') + i + offset]).decode()
        ax.text(x, y, '(%s)' % char, va=va, transform=ax.transAxes, **kwargs)


def ellipse_from_precision(xy, precision, scale=2, **kwargs):
    """
    Create an ellipse from a covariance matrix.

    Parameters
    ----------
    xy : np.ndarray
        position of the ellipse
    cov : np.ndarray
        covariance matrix
    scale : float
        scale of the ellipse (default is three standard deviations)
    kwargs : dict
        keyword arguments passed on to `matplotlib.patches.Ellipse`

    Returns
    -------
    ellipse
    """
    cov = np.linalg.inv(precision)
    evals, evecs = np.linalg.eigh(cov)
    angle = np.arctan2(*reversed(evecs[:, 0]))
    # angle = np.arccos(evecs[0, 0])
    width, height = scale * np.sqrt(evals)
    return mpatches.Ellipse(xy, width, height, np.rad2deg(angle), **kwargs)


def latent_factor_model(y, hyperparameters, A=None):
    num_obs, num_nodes = y.shape
    num_factors = hyperparameters['num_factors']
    # Define the latent factors
    _x = bp.nodes.GaussianARD(0, 1, plates=(num_obs, 1), shape=(num_factors,), name='x')
    ard_prior = hyperparameters['ard_prior']
    if ard_prior == 'independent':
        # Automatic relevance determination prior for the observation model
        _lambda = bp.nodes.Gamma(
            hyperparameters['lambda/shape'],
            hyperparameters['lambda/scale'],
            plates=(num_factors,),
            name='lambda'
        )
        # Compute the predictor and add noise to get the observation
        _A = bp.nodes.GaussianARD(
            0,
            _lambda,
            plates=(1, num_nodes),
            shape=(num_factors,),
            name='A'
        )
    elif ard_prior == 'shared':
        # Automatic relevance determination prior for the observation model
        # (but shared across the dimensions so we need to use model selection)
        _lambda = bp.nodes.Gamma(
            hyperparameters['lambda/shape'],
            hyperparameters['lambda/scale'],
            name='lambda'
        )
        # Compute the predictor and add noise to get the observation
        _A = bp.nodes.GaussianARD(
            0,
            _lambda,
            plates=(1, num_nodes),
            shape=(num_factors,),
            name='A'
        )
    elif ard_prior is None:
        _A = bp.nodes.GaussianARD(
            0,
            hyperparameters['A/precision'],
            shape=(num_factors,),
            plates=(1, num_nodes),
            name='A'
        )
    else:
        raise KeyError(ard_prior)

    _predictor = bp.nodes.SumMultiply('d,d->', _x, _A, name='predictor')

    _tau = bp.nodes.Gamma(
        hyperparameters['tau/shape'],
        hyperparameters['tau/scale'],
        name='tau',
        plates=(num_nodes,)
    )

    _y = bp.nodes.GaussianARD(_predictor, _tau, name='y')

    # Observe the model and initialise the observation model randomly to
    # ensure that we don't end up with a trivial "all-zero" model
    _y.observe(y, mask=np.isfinite(y))
    if A is None:
        _A.initialize_from_random()
    else:
        _A.initialize_from_value(A)

    # Add rotations to speed up the algorithm
    rotations = [
        transformations.RotateGaussianARD(_x),
        transformations.RotateGaussianARD(_A),
    ]
    optimizer = transformations.RotationOptimizer(*rotations, num_factors)

    # Construct an inference model
    variables = [_y, _x, _A, _tau]
    if ard_prior:
        variables.append(_lambda)
    Q = bp.inference.VB(*variables)
    Q.set_callback(optimizer.rotate)
    return Q


def latent_factor_community_model(y, hyperparameters, A=None, **kwargs):
    hyperparameters.update(kwargs)
    num_factors = hyperparameters['num_factors']
    num_groups = hyperparameters['num_groups']
    num_obs, num_nodes = y.shape
    # Define the latent factors
    _x = bp.nodes.GaussianARD(0, 1, plates=(num_obs, 1), shape=(num_factors,), name='x')

    # Mixture structure
    _rho = bp.nodes.Dirichlet(hyperparameters['rho/concentration'] * np.ones(num_groups), name='rho')
    _z = bp.nodes.Categorical(_rho, plates=(num_nodes,), name='z')

    ard_prior = hyperparameters['ard_prior']
    if ard_prior == 'independent':
        _lambda = bp.nodes.Gamma(
            hyperparameters['lambda/shape'],
            hyperparameters['lambda/scale'],
            plates=(num_groups, num_factors,),
            name='lambda',
        )
        _mu = bp.nodes.GaussianARD(
            np.zeros(num_factors),
            _lambda,
            shape=(num_factors,),
            name='mu',
        )
    elif ard_prior == 'shared':
        raise RuntimeError("the shared prior gives incorrect results")
        _lambda = bp.nodes.Gamma(
            hyperparameters['lambda/shape'],
            hyperparameters['lambda/scale'],
            plates=(num_groups, 1),
            name='lambda',
        )
        _mu = bp.nodes.GaussianARD(
            0,
            _lambda,
            plates=(num_groups,),
            shape=(num_factors,),
            name='mu',
        )
    elif ard_prior is None:
        _mu = bp.nodes.Gaussian(
            np.zeros(num_factors),
            hyperparameters['mu/precision'] * np.identity(num_factors),
            plates=(num_groups,),
            name='mu'
        )
    else:
        raise KeyError(ard_prior)

    _Lambda = bp.nodes.Wishart(
        num_factors + hyperparameters['Lambda/shape'],
        num_factors * hyperparameters['Lambda/scale'] * np.identity(num_factors),
        plates=(num_groups,),
        name='Lambda'
    )

    _A = bp.nodes.Mixture(_z, bp.nodes.Gaussian, _mu, _Lambda, name='A')

    # Compute the predictor and add noise to get the observation
    _predictor = bp.nodes.SumMultiply('d,d->', _x, _A, name='predictor')
    _tau = bp.nodes.Gamma(
        hyperparameters['tau/shape'],
        hyperparameters['tau/scale'],
        name='tau',
        plates=(num_nodes,)
    )
    # Observer the model where data are available
    _y = bp.nodes.GaussianARD(_predictor, _tau, name='y')
    _y.observe(y, mask=np.isfinite(y))

    if A is None:
        _A.initialize_from_random()
    else:
        _A.initialize_from_value(A)

    # Intialise a very diffuse prior to ensure we can infer the parameters without
    # having to worry about the community structure in the first pass
    _Lambda.initialize_from_parameters(
        num_factors,
        1e4 * num_factors * np.identity(num_factors)
    )

    # Construct an inference model
    variables = [_y, _x, _A, _tau, _z, _Lambda, _mu, _rho]
    if ard_prior:
        variables.append(_lambda)
    Q = bp.inference.VB(*variables, tol=hyperparameters['tolerance'])
    return Q


def initialize_from_kmeans(Q):
    """
    Initialize the clustering algorithm using K-means.

    Parameters
    ----------
    Q :
        variational bayes model

    Returns
    -------
    kmeans :
        scikit-learn K-means estimator
    """
    num_groups, = Q['mu'].plates
    kmeans = sklearn.cluster.KMeans(num_groups)
    A = Q['A'].get_moments()[0]
    kmeans.fit(A)

    Q['mu'].initialize_from_value(kmeans.cluster_centers_)
    Q['z'].initialize_from_value(kmeans.labels_)

    return kmeans


def initialize_from_bgm(Q, hyperparameters):
    kwargs = {
        'n_components': hyperparameters['num_groups'],
        'covariance_type': 'full',
        'init_params': 'kmeans',
        'weight_concentration_prior_type': 'dirichlet_distribution',
        'weight_concentration_prior': hyperparameters['rho/concentration'],
        'mean_precision_prior': 1,
        'degrees_of_freedom_prior': hyperparameters['Lambda/shape'] + hyperparameters['num_factors'],
        'covariance_prior': np.eye(hyperparameters['num_factors']) * hyperparameters['Lambda/scale'],
        'n_init': 100
    }
    bgm = sklearn.mixture.BayesianGaussianMixture(**kwargs)
    A = Q['A'].get_moments()[0]
    bgm.fit(A)

    Q['z'].initialize_from_value(bgm.predict(A))
    Q['mu'].initialize_from_value(bgm.means_)
    return bgm


def reconstruct_cov(Q, include_x=True):
    """
    Reconstruct the covariance matrix of the latent factor model.

    Parameters
    ----------
    Q :
        variational bayes model
    include_x : bool
        whether to explicitly account for the variance of the latent variables or assume
        unit variance
    """
    A = np.squeeze(Q['A'].get_moments()[0])
    tau = Q['tau'].get_moments()[0]
    
    if include_x:
        x = np.squeeze(Q['x'].get_moments()[0])
        xx = np.dot(x.T, x) / x.shape[0]
    else:
        xx = np.eye(A.shape[1])

    return A.dot(xx).dot(A.T) + np.diag(1 / tau)


def relabel(idx, return_lookup=False):
    """
    Relabel indices such that they are consecutive integers.
    """
    lookup = {}
    for i in idx:
        if i not in lookup:
            lookup[i] = len(lookup)
    idx = np.asarray([lookup[i] for i in idx])
    return (idx, lookup) if return_lookup else idx


def pcolorcoords(values, scale='lin'):
    """
    Generate coordinates for `pcolor` or `pcolormesh`.
    """
    if scale == 'log':
        values = np.log(values)
    else:
        assert scale == 'lin'

    # Compute the difference and find the midpoints
    delta = np.diff(values) * 0.5
    values = np.concatenate([
        [values[0] - delta[0]],
        values[:-1] + delta,
        [values[-1] + delta[-1]],
    ])

    if scale == 'log':
        values = np.exp(values)
    return values


def build_parameters(hyperparameters, *variables, **kwargs):
    """
    Build a sequence of parameter sets by taking the cartesian product of `variables`. If not given,
    `variables` defaults to all variables in `hyperparameters` that are sequences but not `str`.
    """
    hyperparameters = hyperparameters.copy()
    hyperparameters.update(**kwargs)
    
    # Get the keys to compute the product over
    if not variables:
        variables = [key for key, value in sorted(hyperparameters.items())
                     if isinstance(value, collections.Sequence) and not isinstance(value, str)]
    variables = collections.OrderedDict([(var, hyperparameters[var]) for var in variables])

    list_params = []
    for values in it.product(*[hyperparameters[var] for var in variables]):
        params = hyperparameters.copy()
        params.update(dict(zip(variables, values)))
        list_params.append(params)

    return variables, list_params


@contextlib.contextmanager
def timeit(message, verbose):
    start = time()
    yield None
    end = time()
    if verbose:
        print("%s in %.3f" % (message, end - start))


def fit_model(y, hyperparameters, *, steps=None, verbose=False, A=None, **kwargs):
    """
    Fit a latent factor community model.
    """
    if steps is None:
        steps = 5
    if isinstance(steps, int):
        steps = list(range(steps))

    hyperparameters.update(kwargs)
    np.random.seed(hyperparameters['seed'])

    if 0 in steps:
        with timeit("created model", verbose):
            Q = latent_factor_model(y, hyperparameters, A=A)

    if 1 in steps:
        with timeit("fitted embedding", verbose):
            Q.update(repeat=None, verbose=False, tol=hyperparameters['tolerance'])
        if verbose:
            print('lowerbound: %f' % Q.compute_lowerbound())

    if 2 in steps:
        with timeit("created community model", verbose):
            Q1 = Q
            Q = latent_factor_community_model(y, hyperparameters)
            # Initialise the model from the previous model
            for k in ['A', 'x', 'tau']:
                v = Q1[k].get_moments()[0]
                if k == 'A':
                    v = v[0]
                Q[k].initialize_from_value(v)

        init = hyperparameters['initialization']
        with timeit("initialized %s" % init, verbose):
            if hyperparameters['initialization'] == 'kmeans':
                initialize_from_kmeans(Q)
            elif hyperparameters['initialization'] == 'bgm':
                initialize_from_bgm(Q, hyperparameters)
            elif hyperparameters['initialization'] == 'random':
                Q['z'].initialize_from_random()
            else:
                raise KeyError(hyperparameters['initialization'])

            # Update variables once to turn them into proper distributions
            Q.update('Lambda', 'rho', 'mu', 'z', 'A', 'x', 'tau', verbose=False)

            if verbose:
                print('lowerbound: %f' % Q.compute_lowerbound())
                
    # Just fit the GMM
    if 3 in steps:
        with timeit("fitted GMM", verbose):
            Q.update('Lambda', 'rho', 'mu', 'z', 'lambda', verbose=False, repeat=None, tol=hyperparameters['tolerance'])
        if verbose:
                print("lowerbound: %f" % Q.compute_lowerbound())

    # update_order = ['y', 'x', 'A', 'tau']
    if 4 in steps:
        # update_order.extend(['Lambda', 'z', 'mu', 'rho'])
        # if hyperparameters['ard_prior']:
        #    update_order.append('lambda')

        with timeit("fitted full model", verbose):
            Q.update(repeat=None, verbose=False, tol=hyperparameters['tolerance'])
        if verbose:
            print("lowerbound: %f" % Q.compute_lowerbound())

    return Q


def plot_mixture(Q, ax=None, cmap='tab10', permutation=None, scale=2, plot_empty=False):
    ax = ax or plt.gca()
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    _A = Q['A'].get_moments()[0]
    _z = Q['z'].get_moments()[0]
    _labels = np.argmax(_z, axis=1)
    _labels, lookup = relabel(_labels, True)
    if permutation is not None:
        _labels = permutation[_labels]
    _centroids = Q['mu'].get_moments()[0]
    _precisions = Q['Lambda'].get_moments()[0]
    num_groups = len(lookup)

    # Plot the points
    vmax = max(num_groups - 1, 1)
    ax.scatter(*_A.T, c=_labels, cmap=cmap, vmin=0, vmax=vmax)

    # Plot the covariance matrices
    for i, (centroid, precision) in enumerate(zip(_centroids, _precisions)):
        i = lookup.get(i)
        if i is not None:
            if permutation is not None:
                i = permutation[i]
            color = cmap(i / vmax)
        elif plot_empty:
            color = 'gray'
        else:
            color = None
        if color is not None:
            ax.add_artist(ellipse_from_precision(centroid, precision, alpha=.2, edgecolor='k',
                                                 facecolor=color, scale=scale))
            ax.scatter(*centroid, color='k', marker='x')
       
    
def evaluate_labels(Q):
    return np.argmax(Q['z'].get_moments()[0], axis=1)

