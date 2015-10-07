from mpi4py import MPI
from allensdk.model.biophys_sim.config import Config
from utils import Utils
import neuron_parallel

import argparse

import numpy as np
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

BOUND_LOWER, BOUND_UPPER = 0.0, 1.0

UTILS = None
T_VEC = None
V_VEC = None
I_VEC = None

def eval_param_set(params):
    UTILS.set_normalized_parameters(params)
    UTILS.h.finitialize()
    UTILS.h.run()
    feature_errors = UTILS.calculate_feature_errors(T_VEC.as_numpy(), V_VEC.as_numpy(), I_VEC.as_numpy())
    return [np.sum(feature_errors)]

def uniform(lower, upper, size=None):
    if size is None:
        return [random.uniform(a, b) for a, b in zip(lower, upper)]
    else:
        return [random.uniform(a, b) for a, b in zip([lower] * size, [upper] * size)]

def best_sum(d):
    return np.sum(d, axis=1).min()

def initPopulation(pcls, ind_init, popfile):
    popdata = np.loadtxt(popfile)
    return pcls(ind_init(UTILS.normalize_actual_parameters(line)) for line in popdata.tolist())


def main():
    global UTILS, V_VEC, I_VEC, T_VEC

    parser = argparse.ArgumentParser(description='Start a DEAP testing run.')
    parser.add_argument('seed', type=int)
    parser.add_argument('config_path')
    args = parser.parse_args()
    seed = args.seed

    # Set up NEURON
    config = Config().load(args.config_path)

    UTILS = Utils(config)

    manifest = config.manifest
    morphology_path = manifest.get_path('MORPHOLOGY')
    UTILS.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    UTILS.load_cell_parameters()
    UTILS.insert_iclamp()
    stim_params = config.data["stimulus"][0]
    UTILS.set_iclamp_params(stim_params["amplitude"], stim_params["delay"],
        stim_params["duration"])

    UTILS.h.tstop = stim_params["delay"] * 2.0 + stim_params["duration"]
    UTILS.h.cvode_active(1)
    UTILS.h.cvode.atolscale("cai", 1e-4)
    UTILS.h.cvode.maxstep(10)

    V_VEC, I_VEC, T_VEC = UTILS.record_values()

    neuron_parallel.runworker()

    # Set up genetic algorithm

    print "Setting up GA"
    random.seed(seed)

    ngen = 500 
    mu = 1200
    cxpb = 0.1
    mtpb = 0.35
    eta = 10.0

    ndim = len(config.data["channels"]) + len(config.data["addl_params"])

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", uniform, BOUND_LOWER, BOUND_UPPER, ndim)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_param_set)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOWER, up=BOUND_UPPER,
        eta=eta)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOWER, up=BOUND_UPPER,
        eta=eta, indpb=mtpb)
    toolbox.register("variate", algorithms.varAnd)
    toolbox.register("select", tools.selBest)
    toolbox.register("map", neuron_parallel.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("best", best_sum)

    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "min", "max", "best"

    if "STARTPOP" in manifest.path_info:
        print "Using a pre-defined starting population"
        start_pop_path = config.manifest.get_path("STARTPOP")
        toolbox.register("population_start", initPopulation, list, creator.Individual)
        pop = toolbox.population_start(start_pop_path)
    else:
        pop = toolbox.population(n=mu)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(mu)
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print logbook.stream

    for gen in range(1, ngen + 1):
        offspring = toolbox.variate(pop, toolbox, cxpb, 1.0)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)

        pop[:] = toolbox.select(pop + offspring, mu)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print logbook.stream

    fit_dir = config.manifest.get_path("FITDIR")
    seed_dir = fit_dir + "/s{:d}/".format(seed)
    np.savetxt(seed_dir + "final_pop.txt", np.array(map(UTILS.actual_parameters_from_normalized, pop)))
    np.savetxt(seed_dir + "final_pop_fit.txt", np.array([ind.fitness.values for ind in pop]))
    np.savetxt(seed_dir + "final_hof.txt", np.array(map(UTILS.actual_parameters_from_normalized, hof)))
    np.savetxt(seed_dir + "final_hof_fit.txt", np.array([ind.fitness.values for ind in hof]))
    neuron_parallel.done()
    UTILS.h.quit()

if __name__ == "__main__": main()
