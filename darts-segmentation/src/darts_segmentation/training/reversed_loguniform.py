import numpy as np
import scipy.stats
from scipy.stats._distn_infrastructure import rv_continuous


class reversed_loguniform_gen(rv_continuous):
    r"""A reversed log-uniform continuous random variable.

    This distribution has equal probability density on a logarithmic scale
    approaching an upper bound (default 1), but never reaching it.

    For example, [0.9, 0.99), [0.99, 0.999), and [0.999, 0.9999) all have
    equal probability.

    Notes
    -----
    The probability density function for this class is:

    .. math::

        f(x, a, b, n) = \frac{1}{(n-x) \log((n-a)/(n-b))}

    for :math:`a \le x \le b < n`, where :math:`n` is the upper bound
    (default 1). This class takes :math:`a`, :math:`b`, and optionally
    :math:`n` as shape parameters.

    The distribution is created by transforming a loguniform distribution:
    if Y ~ loguniform(n-b, n-a), then X = n - Y ~ reversed_loguniform(a, b, n).

    Examples
    --------
    >>> from scipy.stats import reversed_loguniform
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots(1, 1)

    Generate random variates:

    >>> r = reversed_loguniform(0.5, 0.9999).rvs(size=1000)

    Display histogram on transformed scale to show equal probability:

    >>> ax.hist(-np.log10(1 - r))
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_xlabel("Transformed value (-log10(1-x))")
    >>> plt.show()

    """

    def _argcheck(self, a, b, n=1):
        return (a >= 0) & (b > a) & (b < n)

    def _get_support(self, a, b, n=1):
        return a, b

    def _pdf(self, x, a, b, n=1):
        # reversed_loguniform.pdf(x, a, b, n) = 1 / ((n-x) * log((n-a)/(n-b)))
        return np.exp(self._logpdf(x, a, b, n))

    def _logpdf(self, x, a, b, n=1):
        # Transform: if Y ~ loguniform(n-b, n-a), then X = n - Y
        # pdf_X(x) = pdf_Y(n-x) = 1 / ((n-x) * log((n-a)/(n-b)))
        return -np.log(n - x) - np.log(np.log(n - a) - np.log(n - b))

    def _cdf(self, x, a, b, n=1):
        # CDF: P(X <= x) = P(n - Y <= x) = P(Y >= n - x) = 1 - CDF_Y(n-x)
        # CDF_Y(y) = (log(y) - log(n-b)) / (log(n-a) - log(n-b))
        # So: 1 - CDF_Y(n-x) = 1 - (log(n-x) - log(n-b)) / (log(n-a) - log(n-b))
        return (np.log(n - a) - np.log(n - x)) / (np.log(n - a) - np.log(n - b))

    def _ppf(self, q, a, b, n=1):
        # Inverse of CDF: solve for x
        # q = (log(n-a) - log(n-x)) / (log(n-a) - log(n-b))
        # q * (log(n-a) - log(n-b)) = log(n-a) - log(n-x)
        # log(n-x) = log(n-a) - q * (log(n-a) - log(n-b))
        # n-x = exp(log(n-a) - q * (log(n-a) - log(n-b)))
        # x = n - exp(log(n-a) - q * (log(n-a) - log(n-b)))
        return n - np.exp(np.log(n - a) - q * (np.log(n - a) - np.log(n - b)))

    def _munp(self, k, a, b, n=1):
        # Moments: E[X^k] where X = n - Y and Y ~ loguniform(n-b, n-a)
        # This requires integration, use numerical integration for now
        if k == 0:
            return 1.0
        # Use numerical integration as fallback
        return super()._munp(k, a, b, n)

    def _entropy(self, a, b, n=1):
        # Entropy is same as loguniform since it's just a transformation
        return 0.5 * (np.log(n - a) + np.log(n - b)) + np.log(np.log(n - a) - np.log(n - b))

    def _rvs(self, a, b, n=1, size=None, random_state=None):
        # Generate from loguniform and transform
        # Y ~ loguniform(n-b, n-a), then X = n - Y
        y = scipy.stats.loguniform.rvs(n - b, n - a, size=size, random_state=random_state)
        return n - y


reversed_loguniform = reversed_loguniform_gen(name="reversed_loguniform", shapes="a, b, n")
