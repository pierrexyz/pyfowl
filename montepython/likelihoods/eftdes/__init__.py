from montepython.likelihood_class import Likelihood

import yaml
import os, sys
import numpy as np
from pyfowl.likelihood import Likelihood as Likelihood_eft

class eftdes(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        self.c = yaml.full_load(open(os.path.join(self.data_directory, self.config_file), 'r'))
        self.L = Likelihood_eft(self.c)
        self.need_cosmo_arguments(data, self.L.class_settings)
        self.first_evaluation = True

    def loglkl(self, cosmo, data):

        # if we run with zero varying cosmological parameter, we evaluate the model only once
        if self.first_evaluation: 
            data.update_cosmo_arguments() 
            data.need_cosmo_update = True
            self.first_evaluation = False 

        free_b_name = self.use_nuisance # we should use this since sometimes we fix some parameters but they are still needed in the likelihood
        # free_b_name = data.get_mcmc_parameters(['nuisance', 'varying'])
        free_b = np.array([data.mcmc_parameters[fbn]['current'] * data.mcmc_parameters[fbn]['scale'] for fbn in free_b_name])

        if self.c['jeffrey']:
            free_c_name = data.get_mcmc_parameters(['cosmo', 'varying'])
            free_c = np.array([data.mcmc_parameters[fbn]['current'] * data.mcmc_parameters[fbn]['scale'] for fbn in free_c_name])
            return self.L.loglkl_(free_c, free_c_name, free_b, free_b_name, cosmo, data.need_cosmo_update)
        
        else: return self.L.loglkl(free_b, free_b_name, cosmo, data.need_cosmo_update)         
