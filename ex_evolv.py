#!/usr/bin/env python
import random
from operator import itemgetter

from pybrain.structure.evolvables.evolvable import Evolvable


class Poly(object):

    def __init__(self, *coeff):
        self.coeff = list(coeff)

    def __str__(self):
        n = len(self.coeff) - 1

        p = []

        for index, coeff in enumerate(self.coeff):
            if coeff != 0:
                exp = n - index
                if exp == 0:
                    item = '%d' % (coeff)
                elif exp == 1:
                    item = '%dx' % (coeff)
                else:
                    item = '%dx^%d' % (coeff, exp)
                p.append(item)

        s = ' + '.join(p)
        return s.replace('+ -', '- ')

    def eval(self, x):
        n = len(self.coeff) - 1
        s = 0
        for index, c in enumerate(self.coeff):
            exp = n - index
            s += c * (x ** exp)
        return s

    def r2(self, data):
        r2 = 0
        for x, y in data:
            v = self.eval(x)
            r2 += (y - v) ** 2
        return r2



class PolyEvolve(Evolvable, Poly):

    def mutate(self):
        i = random.randint(0, len(self.coeff)-1)
        self.coeff[i] += random.randint(-1, 1)

    def copy(self):
        return PolyEvolve(*self.coeff)

    def randomize(self, n=None):
        if n is None:
            n = len(self.coeff)
        self.coeff = [random.randint(-10, 10) for x in xrange(n)]

    def __repr__(self):
        return u'PolyEvolve(%r)' % (self.coeff)

    def fitness(self, data):
        return -self.r2(data)


if __name__ == '__main__':

    nCoeffs = random.randint(4,7)

    target = PolyEvolve()
    target.randomize(n=nCoeffs)

    func = lambda x: 5 * (x ** 3) + -3 * (x ** 2) + 1 * (x ** 1) + 1 * (x ** 0)
    point = lambda x: (x, func(x))
    data = sorted([point(random.random() * 200 - 100) for x in xrange(25)], key=itemgetter(0))


    from pybrain.optimization import HillClimber

    seed = PolyEvolve()
    seed.randomize(n=nCoeffs)

    maxIters = 10000
    L = HillClimber(lambda x: x.fitness(data), seed, maxEvaluations=maxIters)

    result, fitness = L.learn()

    fmt = '{:>12}{:>24}{:>24}'
    s = fmt.format('X', 'Y', 'V')
    fmt = '{:>12.2f}{:>24.2f}{:>24.2f}'
    print '-' * len(s)
    print s
    print '-' * len(s)

    for x, y in data:
        v = result.eval(x)
        print fmt.format(x, y, v)

    print '-' * len(s)
    print 'Max Iterations:', maxIters
    print 'Target:', target
    print 'Result:', result
    print 'Fitness:', fitness

    # from pylab import plot, show

    # xAxis = range(-100, 100)
    # plot(
    #     [x for x, y in data], [y for x, y in data], 'ro',
    #     xAxis, [target.eval(x) for x in xAxis], 'r-',
    #     xAxis, [result.eval(x) for x in xAxis], 'b-'
    # )
    # show()



