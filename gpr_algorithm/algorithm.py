import operator
from collections import defaultdict, Counter
from enum import Enum
from random import randint
from typing import List, Iterable, Dict

import geppy as gep
import numpy as np
from deap import base, tools


class GPRAttributeSuffix(str, Enum):
    """
    Linguistic terms of the antecedents of the generated metarules.
    """
    IS_HIGH = '_is_high'
    IS_LOW = '_is_low'
    IS_VERY_HIGH = '_is_very_high'
    IS_VERY_LOW = '_is_very_low'
    IS_MEDIUM = '_is_medium'


GPR_ATTRIBUTE_SUFFIX_TRANSLATES = {
    GPRAttributeSuffix.IS_HIGH: ' is High',
    GPRAttributeSuffix.IS_VERY_HIGH: ' is very High',
    GPRAttributeSuffix.IS_LOW: ' is Low',
    GPRAttributeSuffix.IS_VERY_LOW: ' is very Low',
    GPRAttributeSuffix.IS_MEDIUM: ' is Medium'
}

GPR_ATTRIBUTE_PREFIX = 'a'


def default_eval_function(y_true, y_pred):
    comparison = y_true == y_pred
    score = 0
    unique, counts = np.unique(comparison, return_counts=True)
    for predicted_correctly, count in zip(unique, counts):
        if predicted_correctly:
            score += count
        else:
            score -= 2 * count
    return score


def wrap_crossover(fun):
    def _fun(ind1, ind2):
        if len(ind1) == len(ind2):
            return fun(ind1, ind2)
        return ind1, ind2

    return _fun


class GPRClass(int, Enum):
    THEN = 1
    ELSE = 0


class GPRFitness(base.Fitness):
    weights = (1,)


class GPRChromosome(gep.Chromosome):
    def __init__(self, gene_gen, n_genes, linker=None):
        super().__init__(gene_gen, n_genes, linker)
        self.fitness = GPRFitness()


class GPR:

    def __init__(self,
                 feature_names: List[str],
                 target_names=None,
                 n_populations=100,
                 n_generations=100,
                 eval_fun=default_eval_function,
                 threshold=0.5,
                 verbose=True,
                 max_n_of_rules=6,  # genes_length
                 max_n_of_ands=6,  # head_length
                 base_pb=0.1,
                 ):

        if target_names is None:
            target_names = ['0', '1']

        self.complemented_samples: np.ndarray = np.array([[]])
        self.sample_labels: np.ndarray = np.array([])

        self.feature_names = feature_names
        self.feature_names_translates = {f'{GPR_ATTRIBUTE_PREFIX}{i:03d}': fn for i, fn in enumerate(feature_names)}
        self.target_names = target_names
        self.n_populations = n_populations
        self.n_generations = n_generations
        self.eval_fun = eval_fun
        self.threshold = threshold
        self.verbose = verbose

        self.primitive_set = self._init_primitive_set(self.feature_names_translates.keys())
        self.toolbox = self._init_toolbox(self.primitive_set, base_pb)
        self.generate_population = self._init_generate_population_function(
            self.primitive_set, max_n_of_rules, max_n_of_ands
        )
        self._init_evaluation_function(self.toolbox)
        self.hall_of_fame = tools.HallOfFame(1)
        self.stats = self._init_stats()

    @staticmethod
    def _init_primitive_set(attribute_names: Iterable[str]):
        lows = []
        highs = []
        for a in attribute_names:
            lows.append(f'{a}{GPRAttributeSuffix.IS_LOW}')
            highs.append(f'{a}{GPRAttributeSuffix.IS_HIGH}')

        primitive_set = gep.PrimitiveSet('Main', input_names=lows + highs)
        primitive_set.add_function(operator.mul, 2)
        return primitive_set

    @staticmethod
    def _init_generate_population_function(primitive_set: gep.PrimitiveSet, max_n_of_rules: int, max_n_of_ands: int):
        def generate_gene():
            return gep.Gene(
                pset=primitive_set,
                head_length=max_n_of_ands
            )

        def sum(*x):
            return np.sum(x)

        def generate_chromosome():
            return GPRChromosome(
                gene_gen=generate_gene,
                n_genes=randint(1, max_n_of_rules),
                linker=sum
            )

        def generate_population(n: int):
            return tools.initRepeat(list, generate_chromosome, n)

        return generate_population

    @staticmethod
    def _init_toolbox(primitive_set, base_pb: float):
        toolbox = gep.Toolbox()
        toolbox.register('select', tools.selRoulette)
        toolbox.register('compile', gep.compile_, pset=primitive_set)

        toolbox.register('mut_uniform', gep.mutate_uniform, pset=primitive_set, pb=base_pb)
        toolbox.register('mut_invert', gep.invert, pb=base_pb)
        toolbox.register('mut_is_ts', gep.is_transpose, pb=base_pb)
        toolbox.register('mut_ris_ts', gep.ris_transpose, pb=base_pb)
        toolbox.register('mut_gene_ts', gep.gene_transpose, pb=base_pb)

        toolbox.register('cx_1p', wrap_crossover(gep.crossover_one_point), pb=base_pb)
        toolbox.register('cx_2p', wrap_crossover(gep.crossover_two_point), pb=base_pb)
        toolbox.register('cx_gene', wrap_crossover(gep.crossover_gene), pb=base_pb)

        return toolbox

    @staticmethod
    def _init_stats():
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        return stats

    def _init_evaluation_function(self, toolbox):
        def evaluate(individual):
            func = self._compile_chromosome(individual)
            predictions = (np.apply_along_axis(func, 1, self.complemented_samples) > self.threshold).astype(int)
            return self.eval_fun(self.sample_labels, predictions),

        toolbox.register('evaluate', evaluate)

    def _compile_chromosome(self, chromosome):
        fun = self.toolbox.compile(chromosome)

        def _fun(arr):
            return fun(*arr)

        return _fun

    @staticmethod
    def _compliment_samples(x: np.ndarray):
        samples = np.atleast_2d(x)
        n_params = samples.shape[1]
        complemented_samples = np.zeros((samples.shape[0], n_params * 2))
        complemented_samples[:, :n_params] = 1 - samples  # is low
        complemented_samples[:, n_params:] = samples  # is high
        return complemented_samples

    @property
    def _best_fit(self):
        return self.hall_of_fame[0]

    @property
    def _best_fit_function(self):
        return self._compile_chromosome(self._best_fit)

    @staticmethod
    def _shorten_terminals(terminals: List[str]):
        attr_suffixes = defaultdict(list)
        for t in terminals:
            attr, *other = t.split('_')
            suffix = '_' + '_'.join(other)
            attr_suffixes[attr].append(suffix)

        shortened_terminals = []
        for attr, suffixes in attr_suffixes.items():
            first_suffix = suffixes[0]
            if GPRAttributeSuffix.IS_LOW in suffixes and GPRAttributeSuffix.IS_HIGH in suffixes:
                shortened_terminals.append(f'{attr}{GPRAttributeSuffix.IS_MEDIUM}')
            elif len(suffixes) > 1:
                if first_suffix == GPRAttributeSuffix.IS_LOW:
                    shortened_terminals.append(f'{attr}{GPRAttributeSuffix.IS_VERY_LOW}')
                else:
                    shortened_terminals.append(f'{attr}{GPRAttributeSuffix.IS_VERY_HIGH}')
            else:
                shortened_terminals.append(f'{attr}{first_suffix}')

        return shortened_terminals

    @staticmethod
    def _translate_terminal(terminals: List[str], feature_names_translates: Dict[str, str]):
        translated_terminals = []
        for t in terminals:
            for s_from, s_to in feature_names_translates.items():
                t = t.replace(s_from, s_to)

            for s_from, s_to in GPR_ATTRIBUTE_SUFFIX_TRANSLATES.items():
                t = t.replace(s_from, s_to)

            translated_terminals.append(t)
        return translated_terminals

    @property
    def rules(self) -> List[str]:
        """
        Generates linguistic “if-then” metarules automatically.

        :return: list of metarules
        """
        rules = []
        supports = []

        then_class_name = self.target_names[GPRClass.THEN] if self.target_names is not None else GPRClass.THEN
        else_class_name = self.target_names[GPRClass.ELSE] if self.target_names is not None else GPRClass.ELSE

        for g in self._best_fit:
            func = self._compile_chromosome(gep.Chromosome.from_genes([g]))

            prediction_values = np.apply_along_axis(func, 1, self.complemented_samples)
            support = prediction_values[self.sample_labels == GPRClass.THEN].mean()

            terminals = [s.name for s in g.kexpression if s.arity == 0]
            shortened_terminals = self._shorten_terminals(terminals)
            translated_terminals = self._translate_terminal(shortened_terminals, self.feature_names_translates)
            joined = ' AND '.join(translated_terminals)
            rule = f'IF {joined} THEN {then_class_name} | Support: {support:.4f}'
            rules.append(rule)
            supports.append(support)

        rules = [r for _, r in sorted(zip(supports, rules), reverse=True)]
        rules.append(f'ELSE {else_class_name}')
        return rules

    @property
    def ranking(self):
        """
        Counts the occurrences of each of the attributes and generates a ranking of these attributes.

        :return: a dictionary of the most important attributes sorted descending
        """
        attr_occurrences = []
        for g in self._best_fit:
            terminals = [s.name for s in g.kexpression if s.arity == 0]
            for t in terminals:
                attr = t.split('_')[0]
                attr_occurrences.append(attr)
        counts = Counter(attr_occurrences)
        size = len(attr_occurrences)
        return [f'{self.feature_names_translates[a]}: {c / size:.4f}' for c, a in
                sorted(zip(counts.values(), counts.keys()), reverse=True)]

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the GPR model according to the given training data.

        :param x: training vectors
        :param y: target values
        :return: fitted model
        """
        self.complemented_samples = self._compliment_samples(x)
        self.sample_labels = y

        gep.gep_simple(
            self.generate_population(n=self.n_populations),
            self.toolbox,
            n_generations=self.n_generations,
            n_elites=2,
            stats=self.stats,
            hall_of_fame=self.hall_of_fame,
            verbose=self.verbose
        )

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Method to predict the labels.

        :param x: unlabeled vectors to classify
        :return: class labels for samples in x
        """
        complemented_samples = self._compliment_samples(x)
        return (np.apply_along_axis(self._best_fit_function, 1, complemented_samples) > self.threshold).astype(int)
