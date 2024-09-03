from random import sample
from math import ceil
from ultralytics import YOLO
from numpy.random import choice as npchoice


def estimate_population(species_dict, train_config, dynamic_state):
    # get training individuals from each species
    training_list = []
    for species in species_dict.values():
        member_list = list(species.members.values())
        sample_list = sample(member_list, k=ceil(len(member_list) * (train_config['train_ratio'] / 100.0)))
        print(f'species {species.species_key} - sampled list {[s.genotype_key for s in sample_list]}\n')
        training_list += sample_list

    print('training...')
    for genotype in training_list:
        try:
            print(f'genotype {genotype.genotype_key}({training_list.index(genotype) + 1}-{len(training_list)})')
            if genotype.genotype_dict['fitness'] == 0.0:
                estimate_individual(genotype, is_pretrain=True, train_config=train_config, dynamic_state=dynamic_state)
            else:
                print('individual ', genotype.genotype_key, 'fitness:', genotype.genotype_dict['fitness'], '\n')
        except Exception as e:
            print(f'An error occurred: {e}')


def estimate_individual(genotype, is_pretrain, train_config, dynamic_state=None, val=False):
    if is_pretrain:
        train_epoch = dynamic_state['train_epoch'][-1]
    else:
        train_epoch = train_config['complete_train_epoch']

    phenotype = YOLO(genotype.genotype_yaml)

    if len(train_config['train_device']) > 1:
        phenotype.train(data=train_config['dataset'], epochs=train_epoch, batch=train_config['train_batch'],
                        workers=train_config['train_workers'], device=train_config['train_device'],
                        project=train_config['project'], name=train_config['name'],
                        patience=train_config['train_patience'],
                        imgsz=640, val=val, cache=True, exist_ok=True, plots=False)

        val_model = YOLO('./' + train_config['project'] + '/' + train_config['name'] + '/weights/best.pt')
        metrics = val_model.val(project=train_config['project'], name=train_config['name']+'_val',
                                batch=train_config['train_batch'], workers=train_config['train_workers'],
                                device=train_config['train_device'], cache=True, exist_ok=True)

    else:
        metrics = phenotype.train(data=train_config['dataset'], epochs=train_epoch, batch=train_config['train_batch'],
                                  workers=train_config['train_workers'], device=train_config['train_device'],
                                  project=train_config['project'], name=train_config['name'],
                                  patience=train_config['train_patience'],
                                  imgsz=640, val=val, cache=True, exist_ok=True, plots=False)

    fitness = metrics.box.map50

    fitness = round(float(fitness), 4) if float(fitness) > 0.0 else float(fitness) + 1e-6

    if is_pretrain:
        genotype.genotype_dict['fitness'] = fitness
    else:
        genotype.genotype_dict['final_fitness'] = fitness

    print(f'individual {genotype.genotype_key} fitness: {fitness}\n')
    genotype.genotype_dict['fitness_history'].append(fitness)
    genotype.save_genotype()


def complete_estimation(population, train_config):
    estimation_list = [genotype for genotype in population.genotypes.values()
                       if genotype.genotype_dict['fitness'] >=
                       train_config['fitness_threshold'] * train_config['fitness_ratio'] and
                       genotype.genotype_dict['final_fitness'] == 0.0]
    if len(estimation_list) > 10:
        choice_weights = [genotype.genotype_dict['fitness'] for genotype in estimation_list]
        total_weights = sum(choice_weights)
        norm_choice_weights = [w / total_weights for w in choice_weights]
        estimation_list = npchoice(estimation_list, size=10, replace=False, p=norm_choice_weights)

    satisfied_list = []
    if len(estimation_list) > 0:
        print('estimation list', [genotype.genotype_key for genotype in estimation_list])
        for genotype in estimation_list:
            estimate_individual(genotype, is_pretrain=False, train_config=train_config, val=True)
            if genotype.genotype_dict['final_fitness'] >= train_config['fitness_threshold']:
                satisfied_list.append(genotype)

    return satisfied_list
