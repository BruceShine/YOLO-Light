import os
import yaml
from math import floor
from copy import deepcopy
from itertools import count

from genotype import Genotype


class Population:
    def __init__(self, is_from_scratch, generation, evolution_config,  population_state=None):
        self.genotype_dir = evolution_config['genotype_dir']
        self.genotypes = {}
        self.best_genotype = None
        self.generation = generation

        if is_from_scratch:
            for i in range(1, evolution_config['population_threshold'] + 1):
                new_genotype = Genotype(i, generation, evolution_config['init_genotype'], evolution_config)
                self.genotypes[i] = new_genotype

            self.population_state = []

        else:
            genotype_files = os.listdir(self.genotype_dir)
            for genotype_file in genotype_files:
                if genotype_file[10] == '-':
                    genotype_generation = genotype_file[9]
                else:
                    genotype_generation = genotype_file[9:11]

                genotype_dir_file = self.genotype_dir + genotype_file
                new_genotype = Genotype(0, int(genotype_generation), genotype_dir_file,
                                        evolution_config, use_old_key=True)
                new_genotype.genotype_yaml = genotype_dir_file
                self.genotypes[new_genotype.genotype_key] = new_genotype

            self.population_state = population_state

        self.genotype_indexer = count(max([genotype.genotype_key for genotype in self.genotypes.values()]) + 1)

    def duplicate_population(self, generation, dynamic_state):
        offsprings = {}
        for i in range(floor(dynamic_state['reproduce_freq'][-1])):
            for genotype in self.genotypes.values():
                new_genotype = deepcopy(genotype)
                new_genotype.generation = generation
                new_genotype.genotype_key = next(self.genotype_indexer)
                new_genotype.genotype_dict['age'] = 1
                new_genotype.genotype_dict['fitness_history'] = []
                offsprings[new_genotype.genotype_key] = new_genotype
        return offsprings

    def update_population_state(self, generation, state_file):
        genotypes_sorted = dict(
            sorted(self.genotypes.items(), key=lambda g: g[1].genotype_dict['fitness'], reverse=True))
        best_genotype = next(iter(genotypes_sorted.items()))[1]
        self.best_genotype = best_genotype

        if len(self.population_state) < generation:
            self.population_state.append(
                [self.best_genotype.genotype_key, self.best_genotype.genotype_dict['fitness']])
        else:
            self.population_state[-1] = \
                [self.best_genotype.genotype_key, self.best_genotype.genotype_dict['fitness']]

        with open(state_file, 'r') as file:
            evolution_state = yaml.safe_load(file)

        evolution_state['population_state'] = self.population_state

        with open(state_file, 'w') as file:
            yaml.safe_dump(evolution_state, file, sort_keys=False, default_flow_style=False)

    def save_genotypes(self):
        for genotype in self.genotypes.values():
            genotype.save_genotype()

    def output_population_info(self, output_file):
        output_file.write('best genotype: \n')
        self.best_genotype.print_architecture(output_file, True)

        best_key_list = []
        best_fitness_list = []
        for best_key, best_fitness in self.population_state:
            best_key_list.append(best_key)
            best_fitness_list.append(best_fitness)

        output_file.write(f'best genotype keys so far: {best_key_list}\n')
        output_file.write(f'best genotype fitness so far: {best_fitness_list}\n')

