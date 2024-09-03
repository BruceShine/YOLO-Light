from copy import deepcopy
from math import ceil, floor
from random import choice, choices, sample
from itertools import islice
from numpy.random import choice as npchoice


def add_cell(offsprings, evolution_config, dynamic_state):
    valid_list = [genotype for genotype in offsprings.values()
                  if len(genotype.genotype_dict['backbone']) + len(genotype.genotype_dict['head']) <
                  evolution_config['length_threshold']]

    chosen_genotypes = sample(valid_list, k=ceil(len(valid_list) * (dynamic_state['add_cell_ratio'][-1] / 100.0)))

    for genotype in chosen_genotypes:
        print(f'genotype {genotype.genotype_key}')

        chain_num = genotype.genotype_dict['chain_num']
        chosen_organ = choices(['backbone', 'head'], weights=evolution_config['organ_prob'], k=1)[0]
        chosen_chain = choice([i for i in range(chain_num)])
        cell_list = [i for i in genotype.genotype_dict['chains'][chosen_organ][chosen_chain]]
        cell_list = cell_list[:-1] \
            if chosen_organ == 'backbone' and 'v10' in evolution_config['init_genotype'] else cell_list

        if len(cell_list) == 0:
            chosen_location = 0
            for i in range(chosen_chain):
                chosen_location += len(genotype.genotype_dict['chains'][chosen_organ][i])
            insert_loc = 0
        else:
            chosen_location = choice(cell_list)
            insert_loc = genotype.genotype_dict['chains'][chosen_organ][chosen_chain].index(chosen_location)

        if chosen_organ == 'backbone':
            chosen_cell_type = choice(evolution_config['backbone_cell_types'])
        else:
            chosen_cell_type = choice(evolution_config['head_cell_types'])

        new_cell_gene = None
        candidate_cells = [cell for cell in genotype.genotype_dict[chosen_organ]
                           if cell[2] == chosen_cell_type and len(cell[-1]) < 4]

        if len(candidate_cells) > 0:
            new_cell_gene = deepcopy(choice(candidate_cells))
        else:
            if chosen_cell_type == 'Conv':
                out_channels = evolution_config['conv_out_channels']
                new_cell_gene = [-1, 1, 'Conv', [out_channels, 3, 2]]

            elif chosen_cell_type == 'C3':
                out_channels = evolution_config['c3_out_channels']
                shortcut = True if chosen_organ == 'backbone' else False
                new_cell_gene = [-1, 1, 'C3', [out_channels, shortcut]]

            elif chosen_cell_type == 'C2f':
                out_channels = evolution_config['c2f_out_channels']
                shortcut = True if chosen_organ == 'backbone' else False
                new_cell_gene = [-1, 1, 'C2f', [out_channels, shortcut]]

            elif chosen_cell_type == 'SCDown':
                out_channels = evolution_config['scdown_out_channels']
                new_cell_gene = [-1, 1, 'SCDown', [out_channels, 3, 2]]

            elif chosen_cell_type == 'C2fCIB':
                out_channels = evolution_config['c2fcib_out_channels']
                new_cell_gene = [-1, 1, 'C2fCIB', [out_channels, True]]

            elif chosen_cell_type == 'nn.Upsample':
                new_cell_gene = [-1, 1, 'nn.Upsample', [None, 2, 'nearest']]

        if new_cell_gene is not None:
            genotype.genotype_dict[chosen_organ].insert(chosen_location, new_cell_gene)

            genotype.genotype_dict['chains'][chosen_organ][chosen_chain].insert(insert_loc, chosen_location)

            for i in range(chain_num):
                if i == chosen_chain:
                    for j in range(len(genotype.genotype_dict['chains'][chosen_organ][i])):
                        if j > insert_loc:
                            genotype.genotype_dict['chains'][chosen_organ][i][j] += 1
                elif i > chosen_chain:
                    for j in range(len(genotype.genotype_dict['chains'][chosen_organ][i])):
                        genotype.genotype_dict['chains'][chosen_organ][i][j] += 1

            # update the relation according to the chians dict
            for cell in genotype.genotype_dict['backbone']:
                cell[0] = -1

            for cell in genotype.genotype_dict['head']:
                cell[0] = -1

            for i in range(chain_num):
                b_start = genotype.genotype_dict['chains']['backbone'][i][0]
                genotype.genotype_dict['backbone'][b_start][0] = 0

                if len(genotype.genotype_dict['chains']['head'][i]):
                    h_start = genotype.genotype_dict['chains']['head'][i][0]
                    h_start_value = genotype.genotype_dict['chains']['backbone'][i][-1]
                    genotype.genotype_dict['head'][h_start][0] = h_start_value

            # update detector input index
            genotype.genotype_dict['head'][-1][0] = []
            backbone_length = len(genotype.genotype_dict['backbone'])
            for i in range(chain_num):
                head_length = len(genotype.genotype_dict['chains']['head'][i])
                if head_length > 0:
                    head_end = genotype.genotype_dict['chains']['head'][i][-1]
                    genotype.genotype_dict['head'][-1][0].append(backbone_length + head_end)
                else:
                    backbone_end = genotype.genotype_dict['chains']['backbone'][i][-1]
                    genotype.genotype_dict['head'][-1][0].append(backbone_end)

            print(f'add cell in {chosen_organ} - '
                  f'cell location: {chosen_location}, cell type: {chosen_cell_type}')
            print()

        genotype.genotype_dict['fitness'] = 0.0
        genotype.genotype_dict['final_fitness'] = 0.0

    print()


def modify_cell(offsprings, evolution_config, dynamic_state):
    for genotype in offsprings.values():
        for _ in range(floor(dynamic_state['modify_cell_freq'][-1])):
            chosen_organ = choice(['backbone', 'head'])
            chosen_cell_list = genotype.genotype_dict[chosen_organ][:-2] \
                if chosen_organ == 'backbone' else genotype.genotype_dict[chosen_organ][:-1]

            if len(chosen_cell_list) > 0:
                chosen_cell = choice(chosen_cell_list)

                if chosen_cell[2] in ['Conv', 'CSDown']:
                    if chosen_cell[2] == 'Conv':
                        cell_attr_prob = 'conv_attribution_prob'
                        cell_attr_factor = 'conv_cell_attr_factor'
                    else:
                        cell_attr_prob = 'csdown_attribution_prob'
                        cell_attr_factor = 'csdown_cell_attr_factor'

                    chosen_cell_attribution = choices(['out_channels', 'kernel', 'stride'],
                                                      weights=evolution_config[cell_attr_prob], k=1)[0]

                    if chosen_cell_attribution == 'out_channels':
                        chosen_cell[-1][0] += evolution_config[cell_attr_factor][0]
                    elif chosen_cell_attribution == 'kernel':
                        chosen_cell[-1][1] = choice(evolution_config[cell_attr_factor][1])
                    elif chosen_cell_attribution == 'stride':
                        chosen_cell[-1][2] = choice(evolution_config[cell_attr_factor][2])

                    print(f'genotype {genotype.genotype_key}')
                    print(f'modify cell {chosen_cell} - {chosen_cell_attribution}')

                    # update the out channels of last two cell
                    if chosen_organ == 'backbone' and 'v10' in evolution_config['init_genotype']:
                        genotype.genotype_dict[chosen_organ][-2][-1][0] = (
                            genotype.genotype_dict)[chosen_organ][-3][-1][0]
                        genotype.genotype_dict[chosen_organ][-1][-1][0] = (
                            genotype.genotype_dict)[chosen_organ][-2][-1][0]
                    elif chosen_organ == 'backbone':
                        genotype.genotype_dict[chosen_organ][-1][-1][0] = (
                            genotype.genotype_dict)[chosen_organ][-2][-1][0]

                elif chosen_cell[2] in ['C3', 'C2f', 'C2fCIB']:
                    if chosen_cell[2] == 'C2f':
                        chosen_cell[-1][0] += evolution_config['c2f_cell_attr_factor']
                    elif chosen_cell[2] == 'C3':
                        chosen_cell[-1][0] += evolution_config['c3_cell_attr_factor']
                    else:
                        chosen_cell[-1][0] += evolution_config['c2fcib_cell_attr_factor']

                    print(f'genotype {genotype.genotype_key}')
                    print(f'modify cell {chosen_cell}')

                    if chosen_organ == 'backbone' and 'v10' in evolution_config['init_genotype']:
                        genotype.genotype_dict[chosen_organ][-2][-1][0] = (
                            genotype.genotype_dict)[chosen_organ][-3][-1][0]
                        genotype.genotype_dict[chosen_organ][-1][-1][0] = (
                            genotype.genotype_dict)[chosen_organ][-2][-1][0]
                    elif chosen_organ == 'backbone':
                        genotype.genotype_dict[chosen_organ][-1][-1][0] = \
                                genotype.genotype_dict[chosen_organ][-2][-1][0]

                genotype.genotype_dict['fitness'] = 0.0
                genotype.genotype_dict['final_fitness'] = 0.0

    print()


def crossover_organ(offsprings, dynamic_state):
    chosen_genotypes = sample(list(offsprings.values()),
                              k=ceil(len(offsprings.values()) * (dynamic_state['crossover_ratio'][-1] / 100.0)))

    crossover_list_length = floor(len(chosen_genotypes) / 2)

    crossover_list = [[chosen_genotypes[i], chosen_genotypes[i + crossover_list_length]]
                      for i in range(crossover_list_length)]

    for parent in crossover_list:
        chosen_organ = choice(['backbone', 'head'])
        parent0_organ = deepcopy(parent[0].genotype_dict[chosen_organ])
        parent1_organ = deepcopy(parent[1].genotype_dict[chosen_organ])
        parent[0].genotype_dict[chosen_organ] = parent1_organ
        parent[1].genotype_dict[chosen_organ] = parent0_organ

        parent0_chains = deepcopy(parent[0].genotype_dict['chains'][chosen_organ])
        parent1_chains = deepcopy(parent[1].genotype_dict['chains'][chosen_organ])
        parent[0].genotype_dict['chains'][chosen_organ] = parent1_chains
        parent[1].genotype_dict['chains'][chosen_organ] = parent0_chains

        for genotype in parent:
            chain_num = genotype.genotype_dict['chain_num']
            for cell in genotype.genotype_dict['backbone']:
                cell[0] = -1

            for cell in genotype.genotype_dict['head']:
                cell[0] = -1

            for i in range(chain_num):
                b_start = genotype.genotype_dict['chains']['backbone'][i][0]
                genotype.genotype_dict['backbone'][b_start][0] = 0

                if len(genotype.genotype_dict['chains']['head'][i]):
                    h_start = genotype.genotype_dict['chains']['head'][i][0]
                    h_start_value = genotype.genotype_dict['chains']['backbone'][i][-1]
                    genotype.genotype_dict['head'][h_start][0] = h_start_value

            # update detector input index
            genotype.genotype_dict['head'][-1][0] = []
            backbone_length = len(genotype.genotype_dict['backbone'])
            for i in range(chain_num):
                head_length = len(genotype.genotype_dict['chains']['head'][i])
                if head_length > 0:
                    head_end = genotype.genotype_dict['chains']['head'][i][-1]
                    genotype.genotype_dict['head'][-1][0].append(backbone_length + head_end)
                else:
                    backbone_end = genotype.genotype_dict['chains']['backbone'][i][-1]
                    genotype.genotype_dict['head'][-1][0].append(backbone_end)

        print(f'crossover parents: {parent[0].genotype_key, parent[1].genotype_key}')

        parent[0].genotype_dict['fitness'] = 0.0
        parent[0].genotype_dict['final_fitness'] = 0.0
        parent[1].genotype_dict['fitness'] = 0.0
        parent[1].genotype_dict['final_fitness'] = 0.0

    print()


def select_individuals(generation, specieset, population, evolution_config, state_file):
    # Aging selection
    species_pop_list = []
    for species in specieset.species_dict.values():
        genotype_pop_list = []
        for genotype in species.members.values():
            if genotype.genotype_dict['age'] > evolution_config['age_threshold']:
                genotype_pop_list.append(genotype.genotype_key)
        for genotype_key in genotype_pop_list:
            species.members.pop(genotype_key)
        if len(species.members) == 0:
            species_pop_list.append(species.species_key)
    for species_key in species_pop_list:
        specieset.species_dict.pop(species_key)

    specieset.update_species_state(generation, state_file)

    # select species
    if len(specieset.species_dict) > evolution_config['species_threshold']:
        specieset.species_dict = dict(islice(specieset.species_dict.items(), evolution_config['species_threshold']))

    # select from species
    population_number = sum([len(species.members) for species in specieset.species_dict.values()])
    if population_number > evolution_config['population_threshold']:
        for species_key, species in specieset.species_dict.items():
            reserved_number = ceil(len(species.members) / population_number * evolution_config['population_threshold'])
            choice_weights = [member.genotype_dict['fitness'] if member.genotype_dict['fitness'] > 0.0 else 1e-6
                              for member in species.members.values()]
            total_weights = sum(choice_weights)
            norm_choice_weights = [w / total_weights for w in choice_weights]
            choice_list = list(species.members.keys())
            member_list = npchoice(choice_list, size=reserved_number, replace=False, p=norm_choice_weights)
            new_members = {member.genotype_key: member
                           for member in species.members.values() if member.genotype_key in member_list}
            species.members = new_members

    specieset.update_species_state(generation, state_file)

    # refresh genotypes of population
    next_genotypes = {}
    for species in specieset.species_dict.values():
        for genotype in species.members.values():
            genotype.genotype_dict['age'] += 1
            next_genotypes[genotype.genotype_key] = genotype
    population.genotypes = next_genotypes

    population.update_population_state(generation, state_file)

