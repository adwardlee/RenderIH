import math
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    # introduction for filter: https://jaantollander.com/post/noise-filtering-using-one-euro-filter/#fn:1

    def __init__(self, min_cutoff=0.10, beta=0.07,
                 d_cutoff=1.0, dx0=0):
        # beta = 5
        # min_cutoff = 0.05
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.dx_prev = dx0
        #self.t_prev = float(t0)


    def __call__(self, x, x_prev):
        if x_prev is None:
            return x
        t_e = 1
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)

        # Memorize the previous values.
        self.dx_prev = dx_hat
        return x_hat