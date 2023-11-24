from numpy import mean, min, sum, maximum

def EProgSimplex_new(b, l=1):
    #
    # Problem
    #
    #  min  1/2 || x - b||^2
    #  s.t. x>=0, 1'x=l
    #
    n = b.shape[0]
    x = b - mean(b) + l/n
    if min(x) < 0:
        # diff = |1'x - l|
        diff = 1
        lambda_m = 0
        it = 0
        x0 = x
        while abs(diff) > 1e-10:
            x = x0 - lambda_m
            npos = sum(x > 0)
            diff = sum(x[x > 0]) - l
            lambda_m = lambda_m + diff/npos
            it = it+1
            if it > 100:
                break
        x = maximum(x, 0)
    return x