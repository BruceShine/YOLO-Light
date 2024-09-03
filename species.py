import functools

import yaml
from itertools import count


class Species:
    def __init__(self, species_key, generation):
        self.species_key = species_key
        self.generation = generation
        self.members = {}
        self.representation = None
        self.fitness = 0.0
        self.fitness_history = []

    def update_species(self, current_generation):
        sorted_members = dict(sorted(self.members.items(), key=lambda m: m[1].genotype_dict['fitness'], reverse=True))
        self.members = sorted_members
        self.representation = next(iter(self.members.items()))[1]
        self.fitness = self.representation.genotype_dict['fitness']
        if len(self.fitness_history) == current_generation - self.generation:
            self.fitness_history.append(self.fitness)
        else:
            self.fitness_history[-1] = self.fitness

    def output_species_info(self, output_file):
        output_file.write(f'species key - {self.species_key}: \n')
        output_file.write(f'species members ({len(self.members)}): {list(self.members.keys())}\n')
        output_file.write(f'species fitness - {self.fitness}\n')
        output_file.write('species representative member info: \n')
        self.representation.print_architecture(output_file=output_file)


class SpeciesSet(object):
    def __init__(self, is_from_scratch, species_state=None, population=None):
        self.species_dict = {}

        if is_from_scratch:
            self.species_state = []

            self.species_counter = count(1)

        else:
            self.species_state = species_state
            generation = len(self.species_state)
            for species_key, member_list, fitness_history in self.species_state[-1]:
                self.species_dict[species_key] = Species(species_key, generation)
                for member_key in member_list:
                    self.species_dict[species_key].members[member_key] = population.genotypes[member_key]
                self.species_dict[species_key].fitness_history = fitness_history

            self.species_counter = count(max([species_key for species_key in self.species_dict.keys()]) + 1)

    def speciate(self, population, evolution_config, generation):
        # speciate remain genotypes
        for genotype in population.genotypes.values():
            min_distance = evolution_config['homologue_distance']
            target_species_key = None

            # compute distance between genotype with each representation, find candidates
            for species_key, species in self.species_dict.items():
                distance = compute_distance(genotype, species.representation, evolution_config['distance_coefficient'])

                if distance < min_distance:
                    min_distance = distance
                    target_species_key = species_key

            if target_species_key is not None:
                self.species_dict[target_species_key].members[genotype.genotype_key] = genotype
            else:
                # one genotype have no match candidates, new species and be its representation
                new_species_key = next(self.species_counter)
                self.species_dict[new_species_key] = Species(new_species_key, generation)
                self.species_dict[new_species_key].representation = genotype
                self.species_dict[new_species_key].members[genotype.genotype_key] = genotype

    def update_species_state(self, generation, state_file):
        # sort members in each species by fitness
        for species in self.species_dict.values():
            species.update_species(generation)

        sorted_spcies_dict = dict(sorted(self.species_dict.items(), key=lambda s: s[1].fitness, reverse=True))
        self.species_dict = sorted_spcies_dict

        if len(self.species_state) < generation:
            self.species_state.append(
                [[species_key, list(species.members.keys()), species.fitness_history]
                 for species_key, species in self.species_dict.items()])

        else:
            self.species_state[-1] = \
                [[species_key, list(species.members.keys()), species.fitness_history]
                 for species_key, species in self.species_dict.items()]

        with open(state_file, 'r') as file:
            evolution_state = yaml.safe_load(file)

        evolution_state['species_state'] = self.species_state

        with open(state_file, 'w') as file:
            yaml.safe_dump(evolution_state, file, sort_keys=False, default_flow_style=False)

    def output_specieset_info(self, generation, output_file):
        for species in self.species_dict.values():
            species.output_species_info(output_file)

        for species_key, menber_list, fitness_history in self.species_state[generation-1]:
            output_file.write(f'species {species_key}, fitness history: {fitness_history}\n')
        output_file.write('\n')


def compute_distance(this_genotype, another_genotype, distance_coefficient):
    distance = 0

    for part in ['backbone', 'head']:
        this_cell_keys = []
        another_cell_keys = []
        for this_cell_list in this_genotype.genotype_dict['chains'][part]:
            this_cell_keys += this_cell_list
        for another_cell_list in another_genotype.genotype_dict['chains'][part]:
            another_cell_keys += another_cell_list

        same_cell_keys = [key for key in this_cell_keys if key in another_cell_keys]
        different_cell_keys_this = [key for key in this_cell_keys if key not in another_cell_keys]
        different_cell_keys_another = [key for key in another_cell_keys if key not in this_cell_keys]
        different_cell_keys = different_cell_keys_this + different_cell_keys_another

        for cell_key in same_cell_keys:
            if this_genotype.genotype_dict[part][cell_key][2] != \
                    another_genotype.genotype_dict[part][cell_key][2]:
                distance += 1

        distance += len(different_cell_keys)

    return distance * distance_coefficient
