import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from ..utils import reorder_constraints
from .base import BaseOptimizer


def constraint_lhs_for_reduce(
    constraint_lhs, constraint_order, n_features, normalize_columns, theta
):
    """Constraint matrix as the optimizer's inner loop sees it.

    Returns the constraints in "feature" order (column = feature * n_targets
    + target). With ``normalize_columns`` the inner loop works on
    w~ = w * ||theta_j||, so ``C w = d`` is rewritten as
    ``C diag(1 / ||theta_j||) w~ = d``; the constraints then still hold for
    the coefficients of the original library.
    """
    if constraint_order == "target":
        constraint_lhs = reorder_constraints(
            constraint_lhs, n_features, output_order="feature"
        )
    constraint_lhs = np.asarray(constraint_lhs, dtype=float)
    if normalize_columns:
        feat_norms = np.linalg.norm(np.asarray(theta), 2, axis=0)
        n_targets = constraint_lhs.shape[1] // n_features
        constraint_lhs = constraint_lhs / np.repeat(feat_norms, n_targets)
    return constraint_lhs


def constrained_lstsq(
    x, y, ind, constraint_lhs, constraint_rhs, alpha=0.0, ineq_lhs=None, ineq_rhs=None
):
    """Solve the equality-constrained ridge regression

    .. math::
        \\min_w \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_2^2
        \\quad \\text{s.t.} \\quad Cw = d, \\; Aw \\le b

    jointly across all targets, restricted to the active support
    ``ind`` (shape (n_targets, n_features), boolean). Coefficients
    outside the support are held fixed at zero (and their columns are
    dropped from the constraint, since they cannot contribute to it).
    ``constraint_lhs`` and ``ineq_lhs`` must be in "feature" order. Without
    inequality rows the KKT system is solved directly; with them the problem
    is a small QP solved with cvxpy.
    """
    n_samples, n_features = x.shape
    n_targets = y.shape[1]

    active_cols = [np.flatnonzero(ind[i]) for i in range(n_targets)]
    sizes = [len(c) for c in active_cols]
    offsets = np.cumsum([0] + sizes)
    total_active = offsets[-1]

    coef_full = np.zeros((n_targets, n_features))
    if total_active == 0:
        return coef_full

    # Each target's regression is stacked into one block-diagonal
    # system, because a constraint may couple coefficients that belong
    # to different targets.
    x_block = np.zeros((n_samples * n_targets, total_active))
    y_stack = np.zeros(n_samples * n_targets)
    for i in range(n_targets):
        rows = slice(i * n_samples, (i + 1) * n_samples)
        cols = slice(offsets[i], offsets[i + 1])
        x_block[rows, cols] = x[:, active_cols[i]]
        y_stack[rows] = y[:, i]

    h = x_block.T.dot(x_block) + alpha * np.eye(total_active)
    g = x_block.T.dot(y_stack)

    # constraint_lhs columns are grouped feature-major:
    # column index = feature * n_targets + target
    flat_idx = np.concatenate(
        [active_cols[i] * n_targets + i for i in range(n_targets)]
    )
    c_active = constraint_lhs[:, flat_idx]
    d = constraint_rhs

    if ineq_lhs is not None and len(ineq_lhs):
        w_active = _constrained_qp(
            x_block, y_stack, alpha, c_active, d, ineq_lhs[:, flat_idx], ineq_rhs
        )
        for i in range(n_targets):
            coef_full[i, active_cols[i]] = w_active[offsets[i] : offsets[i + 1]]
        return coef_full

    n_c = c_active.shape[0]
    kkt = np.block([[h, c_active.T], [c_active, np.zeros((n_c, n_c))]])
    rhs = np.concatenate([g, d])
    sol, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
    w_active = sol[:total_active]

    if not np.allclose(c_active.dot(w_active), d, atol=1e-8, rtol=1e-5):
        warnings.warn(
            "The active support after thresholding cannot satisfy the "
            "given linear constraints exactly; returning a "
            "least-squares approximation instead.",
            ConvergenceWarning,
        )

    for i in range(n_targets):
        coef_full[i, active_cols[i]] = w_active[offsets[i] : offsets[i + 1]]
    return coef_full


def _constrained_qp(x_block, y_stack, alpha, c_eq, d_eq, a_ineq, b_ineq):
    """min ||x_block w - y||^2 + alpha ||w||^2  s.t.  c_eq w = d_eq, a_ineq w <= b_ineq"""
    try:
        import cvxpy as cp
    except ImportError as e:  # pragma: no cover
        raise ImportError("inequality constraints require cvxpy") from e
    w = cp.Variable(x_block.shape[1])
    cost = cp.sum_squares(x_block @ w - y_stack)
    if alpha:
        cost = cost + alpha * cp.sum_squares(w)
    cons = [a_ineq @ w <= b_ineq]
    if c_eq.shape[0]:
        cons.append(c_eq @ w == d_eq)
    prob = cp.Problem(cp.Minimize(cost), cons)
    solver = cp.CLARABEL if cp.CLARABEL in cp.installed_solvers() else None
    prob.solve(solver=solver)
    if w.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        warnings.warn(
            f"Constrained least-squares QP failed ({prob.status}); returning zeros.",
            ConvergenceWarning,
        )
        return np.zeros(x_block.shape[1])
    return np.asarray(w.value)


class ConstrainedSTLSQ(BaseOptimizer):
    """Sequentially thresholded least squares with linear equality
    (and optionally inequality) constraints.

    Attempts to minimize the objective function

    .. math::

        \\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2

    .. math::

        \\text{subject to } Cw = d, \\; Aw \\le b

    by iterating between an equality-constrained ridge regression (solved
    via the Lagrange-multiplier / KKT linear system) and masking out
    elements of the weight array ``w`` that fall below a given threshold,
    in the same manner as :class:`STLSQ`. Because a linear constraint can
    couple coefficients belonging to different target variables, every
    target is solved for jointly at each iteration, unlike plain
    :class:`STLSQ`, which regresses each target independently.

    See the following references for more details:

        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        IEEE Access 8 (2020): 169259-169271.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    constraint_lhs : numpy ndarray, shape (n_constraints, n_features * n_targets)
        The left hand side matrix C of the constraint Cw = d. There should
        be one row per constraint. Required.

    constraint_rhs : numpy ndarray, shape (n_constraints,)
        The right hand side vector d of the constraint Cw = d. Required.

    inequality_lhs : numpy ndarray, shape (n_ineq, n_features * n_targets), optional
        The matrix A of optional inequality constraints Aw <= b, in the same
        ``constraint_order`` as ``constraint_lhs``. With inequalities each
        regression is a QP solved with cvxpy (cvxpy must be installed).

    inequality_rhs : numpy ndarray, shape (n_ineq,), optional
        The right hand side b of Aw <= b.

    constraint_order : string, optional (default "target")
        The format in which ``constraint_lhs`` was passed. Must be one of
        "target" or "feature". "target" indicates that the constraints are
        grouped by target: the first ``n_features`` columns correspond to
        constraint coefficients on the library features for the first
        target, the next ``n_features`` columns to the second target, and
        so on. "feature" indicates the constraints are grouped by library
        feature instead. See :func:`pysindy.utils.reorder_constraints`.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before
        regression by dividing by the L2-norm. The threshold and ``alpha``
        then act on the normalized coefficients, while the constraints are
        still imposed on the coefficients of the original library.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features,) or (n_targets, n_features),
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every iteration.

    unbias : bool, optional (default True)
        Whether to refit the coefficients on the final support with no
        ridge penalty (``alpha=0``), still subject to ``Cw = d``. This removes
        the bias introduced by ``alpha``. Unlike the base-class unbias step,
        which is an unconstrained per-target least-squares fit, the refit
        keeps the constraints satisfied.

    Attributes
    ----------
    coef_ : array, shape (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_targets, n_features)
        Array of bools indicating which coefficients of the weight vector
        have not been masked out, i.e. the support of ``self.coef_``.

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import PolynomialLibrary, SINDy
    >>> from pysindy.optimizers import ConstrainedSTLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> # constrain the coefficient of x0 in the x0' equation and the
    >>> # coefficient of x1 in the x1' equation to be opposite (target order,
    >>> # library = [1, x0, x1, x2]): row picks out those two entries.
    >>> lhs = np.zeros((1, 3 * 4))
    >>> lhs[0, 1] = 1  # x0 coefficient of x0' equation
    >>> lhs[0, 4 + 2] = 1  # x1 coefficient of x1' equation
    >>> rhs = np.array([0.0])
    >>> opt = ConstrainedSTLSQ(
    ...     threshold=.1, alpha=.5, constraint_lhs=lhs, constraint_rhs=rhs
    ... )
    >>> model = SINDy(optimizer=opt, feature_library=PolynomialLibrary(degree=1))
    >>> model.fit(x, t=t[1]-t[0])
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.05,
        max_iter=20,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        normalize_columns=False,
        copy_X=True,
        initial_guess=None,
        verbose=False,
        unbias=True,
        inequality_lhs=None,
        inequality_rhs=None,
    ):
        super().__init__(
            max_iter=max_iter,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            initial_guess=initial_guess,
            unbias=unbias,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        if constraint_lhs is None or constraint_rhs is None:
            raise ValueError(
                "constraint_lhs and constraint_rhs are both required; use "
                "STLSQ instead if no linear constraints are needed"
            )
        if constraint_order not in ("feature", "target"):
            raise ValueError("constraint_order must be 'feature' or 'target'")
        if (inequality_lhs is None) != (inequality_rhs is None):
            raise ValueError("inequality_lhs and inequality_rhs must be given together")

        self.threshold = threshold
        self.alpha = alpha
        self.constraint_lhs = np.asarray(constraint_lhs, dtype=float)
        self.constraint_rhs = np.asarray(constraint_rhs, dtype=float)
        self.constraint_order = constraint_order
        self.verbose = verbose
        self.inequality_lhs = (
            None if inequality_lhs is None else np.asarray(inequality_lhs, dtype=float)
        )
        self.inequality_rhs = (
            None if inequality_rhs is None else np.asarray(inequality_rhs, dtype=float)
        )

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _constraint_lhs_feature_order(self, n_features):
        return constraint_lhs_for_reduce(
            self.constraint_lhs,
            self.constraint_order,
            n_features,
            self.normalize_columns,
            getattr(self, "Theta_", None),
        )

    def _regress(
        self, x: np.ndarray, y: np.ndarray, ind: np.ndarray, constraint_lhs, alpha=None
    ):
        """Constrained ridge regression on the support ``ind``;
        see :func:`constrained_lstsq`."""
        alpha = self.alpha if alpha is None else alpha
        ineq_lhs = None
        if self.inequality_lhs is not None:
            ineq_lhs = constraint_lhs_for_reduce(
                self.inequality_lhs,
                self.constraint_order,
                x.shape[1],
                self.normalize_columns,
                getattr(self, "Theta_", None),
            )
        return constrained_lstsq(
            x,
            y,
            ind,
            constraint_lhs,
            self.constraint_rhs,
            alpha,
            ineq_lhs=ineq_lhs,
            ineq_rhs=self.inequality_rhs,
        )

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of constrained
        sequentially-thresholded least squares.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        n_targets, n_features = self.coef_.shape
        constraint_lhs = self._constraint_lhs_feature_order(n_features)

        ind = self.ind_
        n_features_selected = np.sum(ind)
        optvar = np.zeros((n_targets, n_features))

        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "Total error: |y - Xw|^2 + a * |w|_2",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        for k in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                optvar = np.zeros((n_targets, n_features))
                break

            optvar = self._regress(x, y, ind, constraint_lhs)
            self.iters += 1
            ind = np.abs(optvar) >= self.threshold
            optvar[~ind] = 0

            self.history_.append(optvar)
            if self.verbose:
                R2 = np.sum((y - np.dot(x, optvar.T)) ** 2)
                L2 = self.alpha * np.sum(optvar**2)
                L0 = np.count_nonzero(optvar)
                row = [k, R2, L2, L0, R2 + L2]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10d}"
                    " ... {4:10.4e}".format(*row)
                )
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                f"{type(self).__name__} did not converge after "
                f"{self.max_iter} iterations.",
                ConvergenceWarning,
            )

        self.coef_ = optvar
        self.ind_ = ind

    def _unbias(self, x: np.ndarray, y: np.ndarray):
        """Refit on the final support with alpha=0, keeping Cw = d.

        Overrides the base-class unbias step, whose unconstrained per-target
        least squares would break the constraints.
        """
        if not np.any(self.ind_):
            return
        constraint_lhs = self._constraint_lhs_feature_order(self.coef_.shape[1])
        self.coef_ = self._regress(x, y, self.ind_, constraint_lhs, alpha=0.0)

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )
