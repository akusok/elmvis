import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def GA(X, A, pop=100, gen=200, verbose=False):
    N, d = X.shape
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("indices", np.random.permutation, N)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        x1 = X[individual, :]
        return np.trace(x1.T.dot(A).dot(x1))/d,

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxOrdered)
#    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    popl = toolbox.population(n=pop)
    if not verbose:
        res, log = algorithms.eaSimple(popl, toolbox, cxpb=0.2, mutpb=0.3, ngen=gen, verbose=False)
    else:
        from matplotlib import pyplot as plt
        import operator
        fit_stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
        fit_stats.register('mean', np.mean)
        fit_stats.register('max', np.max)
        res, log = algorithms.eaSimple(popl, toolbox, cxpb=0.2, mutpb=0.3, ngen=gen,
                                       verbose=True, stats=fit_stats)
        plt.clf()
        plots = plt.plot(log.select('max'), 'c-',
                         log.select('mean'), 'b-', antialiased=True)
        plt.legend(plots, ('Maximum fitness', 'Mean fitness'), 4)
        plt.ylabel('Fitness')
        plt.xlabel('Iterations')
        plt.show()

    pbest = tools.selBest(res, k=1)[0]
    simbest = evalOneMax(pbest)
    return pbest, simbest
