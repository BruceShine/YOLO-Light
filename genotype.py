import yaml
import os


class Genotype(object):
    def __init__(self, genotype_key, generation, genotype_file, evolution_config, use_old_key=False):
        with open(genotype_file, 'r') as file:
            self.genotype_dict = yaml.safe_load(file)

        self.genotype_key = genotype_key
        if use_old_key:
            self.genotype_key = self.genotype_dict['genotype_key']

        self.generation = generation
        self.genotype_dir = evolution_config['genotype_dir']
        self.genotype_yaml = None

    def save_genotype(self):
        save_dir = os.path.join(self.genotype_dir)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        genotype_file_name = save_dir + 'genotype-' + str(self.generation) + '-' + str(self.genotype_key) + '.yaml'
        self.genotype_dict['genotype_key'] = self.genotype_key
        with open(genotype_file_name, 'w') as file:
            yaml.safe_dump(self.genotype_dict, file, sort_keys=False, default_flow_style=False)

        self.genotype_yaml = genotype_file_name

    def print_architecture(self, output_file, print_fitness=False):
        output_file.write(f'genotype key - {self.genotype_key}\n')
        output_file.write('genotype dict:\n')
        for cell_gene in self.genotype_dict['backbone']:
            output_file.write(f'{cell_gene}\n')
        for cell_gene in self.genotype_dict['head']:
            output_file.write(f'{cell_gene}\n')
        if print_fitness:
            output_file.write(f"fitness: {self.genotype_dict['fitness']}\n")
            if self.genotype_dict['final_fitness'] > 0.0:
                output_file.write(f"final fitness: {self.genotype_dict['final_fitness']}\n")
        output_file.write('\n')
