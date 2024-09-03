import os
import yaml
import time

from population import Population
from species import SpeciesSet
from evolution_operation import add_cell, modify_cell, crossover_organ, select_individuals
from estimation import estimate_population, complete_estimation


class NeuroEvolution:
    def __init__(self, is_from_scratch, evolution_state_file):
        local_dir = os.path.dirname(__file__)
        self.state_file = os.path.join(local_dir, evolution_state_file)
        with open(self.state_file, 'r') as file:
            evolution_state = yaml.safe_load(file)

        self.evolution_config = evolution_state['evolution_config']
        self.train_config = evolution_state['train_config']
        self.is_from_scratch = is_from_scratch

        if self.is_from_scratch:
            self.generation = 1
            self.population = Population(is_from_scratch, self.generation, self.evolution_config)
            self.specieset = SpeciesSet(is_from_scratch)
            self.dynamic_state = evolution_state['dynamic_config']
        else:
            self.dynamic_state = evolution_state['dynamic_state']
            self.generation = self.dynamic_state['last_generation'] + 1
            self.population = Population(is_from_scratch, self.generation, self.evolution_config,
                                         population_state=evolution_state['population_state'])
            self.specieset = SpeciesSet(is_from_scratch, evolution_state['species_state'], self.population)

    def neuroevolution(self):
        while self.generation <= self.evolution_config['generation_limit']:
            start_time = time.time()

            print(f'|========== Generation {self.generation} ==========|')
            if self.is_from_scratch:
                print('\n|---------- evolve ----------|')

                # [Duplication]
                offsprings = self.population.duplicate_population(self.generation, self.dynamic_state)

                # [Mutation]
                add_cell(offsprings, self.evolution_config, self.dynamic_state)
                modify_cell(offsprings, self.evolution_config, self.dynamic_state)

                # [Crossover]
                crossover_organ(offsprings, self.dynamic_state)

                # save genotype files in population after evolution
                self.population.genotypes.update(offsprings)
                self.population.save_genotypes()

                # [Speciation]
                self.specieset.speciate(self.population, self.evolution_config, self.generation)
                self.specieset.update_species_state(self.generation, self.state_file)

            # [Estimation]
            print('|---------- estimate ----------|')
            estimate_population(self.specieset.species_dict, self.train_config, self.dynamic_state)

            # [Selection]
            select_individuals(self.generation, self.specieset, self.population, self.evolution_config, self.state_file)

            satisfied_genotypes = complete_estimation(self.population, self.train_config)

            end_time = time.time()
            current_time = round((end_time - start_time) / 3600.0, 2)

            self.self_adapt(current_time)

            self.output_evolution_info(self.evolution_config['evolution_info'])

            if len(satisfied_genotypes) > 0:
                for genotype in satisfied_genotypes:
                    f = open(self.evolution_config['evolution_info'], 'a')
                    print(f'\nThe SATISFIED individual:', file=f)
                    genotype.print_architecture(f, True)
                    f.close()
                return

            self.generation += 1
            self.is_from_scratch = True

        return self.population.best_genotype

    def self_adapt(self, current_time):
        self.dynamic_state['reproduce_freq'].append(self.dynamic_state['reproduce_freq'][-1])
        self.dynamic_state['add_cell_ratio'].append(self.dynamic_state['add_cell_ratio'][-1])
        self.dynamic_state['modify_cell_freq'].append(self.dynamic_state['modify_cell_freq'][-1])
        self.dynamic_state['crossover_ratio'].append(self.dynamic_state['crossover_ratio'][-1])
        self.dynamic_state['train_epoch'].append(self.dynamic_state['train_epoch'][-1])

        if self.generation > self.evolution_config['start_dynamics']:
            f_opt = self.train_config['fitness_threshold']
            f_cur = self.population.best_genotype.genotype_dict['fitness']

            if f_opt >= f_cur:
                self.dynamic_state['convergence_rate'].append(
                    round(1 - ((f_opt-f_cur)/f_opt) ** (1/(self.generation-self.evolution_config['start_dynamics'])), 2))

            if self.dynamic_state['convergence_rate'][-1] < self.train_config['convergence_rate_threshold']:
                if self.dynamic_state['reproduce_freq'][-1] < self.evolution_config['reproduce_freq_ceiling']:
                    self.dynamic_state['reproduce_freq'][-1] += self.evolution_config['reproduce_freq_step']

                if self.dynamic_state['add_cell_ratio'][-1] > self.evolution_config['add_cell_ratio_floor']:
                    self.dynamic_state['add_cell_ratio'][-1] -= self.evolution_config['add_cell_ratio_step']

                if self.dynamic_state['modify_cell_freq'][-1] < self.evolution_config['modify_cell_freq_ceiling']:
                    self.dynamic_state['modify_cell_freq'][-1] += self.evolution_config['modify_cell_freq_step']

                if self.dynamic_state['crossover_ratio'][-1] > self.evolution_config['crossover_ratio_floor']:
                    self.dynamic_state['crossover_ratio'][-1] -= self.evolution_config['crossover_ratio_step']

                if self.dynamic_state['train_epoch'][-1] < self.train_config['train_epoch_threshold']:
                    self.dynamic_state['train_epoch'][-1] *= 2
                    if self.dynamic_state['train_epoch'][-1] > self.train_config['train_epoch_threshold']:
                        self.dynamic_state['train_epoch'][-1] = self.train_config['train_epoch_threshold']

        self.dynamic_state['generation_time'].append(current_time)
        self.dynamic_state['last_generation'] = self.generation

        with open(self.state_file, 'r') as file:
            evolution_state = yaml.safe_load(file)

        evolution_state['dynamic_state'] = self.dynamic_state

        with open(self.state_file, 'w') as file:
            yaml.safe_dump(evolution_state, file, sort_keys=False, default_flow_style=False)

    def output_evolution_info(self, out_file):
        with open(out_file, 'a') as f:
            f.write(f'|========== generation {self.generation} ==========|\n')

            f.write('|---------- dynamic state ----------|\n')
            for key, value in self.dynamic_state.items():
                if key not in ['generation_time', 'convergence_rate', 'last_generation']:
                    f.write(f'{key}: {value[:-1]}\n')
            f.write('\n')

            f.write('|---------- species info ----------|\n')
            self.specieset.output_specieset_info(self.generation, f)

            f.write('|---------- population info ----------|\n')
            self.population.output_population_info(f)

            f.write(f"convergence rate: {self.dynamic_state['convergence_rate']}\n")
            f.write(f"generation time (h): {self.dynamic_state['generation_time']}\n")
            f.write(f"total time (h): {sum(self.dynamic_state['generation_time'])}\n")
            f.write('\n\n\n\n\n')


if __name__ == '__main__':
    evolution_state = NeuroEvolution(is_from_scratch=False, evolution_state_file='sodabottles_1c_evostate.yaml')
    evolution_state.neuroevolution()



