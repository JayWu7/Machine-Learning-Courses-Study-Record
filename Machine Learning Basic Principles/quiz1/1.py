'''
computing the combinations value
'''

from scipy.special import comb


def helper():
    '''
    helping in calculating the probability
    '''
    res = 0
    for i in range(26, 41):
        res += comb(50, i) * (0.4 ** i) * (0.6 ** (50 - i))
    return res

print(helper())
