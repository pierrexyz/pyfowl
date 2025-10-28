import os, sys
import yaml
import numpy as np
from copy import copy

from scipy import stats
def pvalue(minchi2, dof): return 1. - stats.chi2.cdf(minchi2, dof)

class hiddenprint():
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from classy import Class
from iminuit import minimize as minuit

from pyfowl.likelihood import Likelihood

class Tools(object):
    
    def __init__(self, config, verbose=True): 

        self.c = config
        self.class_settings = {'output': 'mPk', 'z_max_pk': 5, 'P_k_max_1/Mpc': 1.}

    def update_cosmo(self, cosmo):
        self.cosmo = cosmo
        self.M = Class()
        self.M.set(self.class_settings) #, 'use_nn': 'yes', 'workspace_path': '/cluster/work/senatore/class_public-classnet/classnet_workspace'}) 
        self.M.set(cosmo)
        self.M.compute()
        return

    def get_bestfit(self, cosmo=None, nuisance=None, marg=True, return_lkl=False): # posterior mode / maximum aposteriori 
        if cosmo is not None: self.update_cosmo(cosmo)
        self.c.update({'marg': marg, 'marg_chi2': marg})
        with hiddenprint(): L = Likelihood(self.c) # instantiate likelihood
        if nuisance is None: nuisance_array = np.ones(shape=(L.N)) # initial positions
        else: nuisance_array = np.array([v for n, v in nuisance.items() if n in L.nuisance_name_nonmarg]) # unmarginalised nuisance values
        init = L.loglkl(nuisance_array, None, self.M, need_cosmo_update=True) # initialise on cosmology
        if nuisance is None: 
            def chi2(params): return -2. * L.loglkl(params, None, self.M, need_cosmo_update=False)
            minimum = minuit(chi2, nuisance_array) # minimisation
            chi2, nuisance_array = minimum['fun'], minimum['x'] # final positions
        else: chi2 = -2. * init 
        if marg: nuisance_array = np.concatenate((nuisance_array, L.bg)) # nonmarg + marg parameters
        ndata, dof = L.ydata.shape[0], L.ydata.shape[0] - len(nuisance_array) # number of data points - parameters
        print ('min chi2 (+ prior): %.2f (+ %.2f), ndata: %d, dof: %d, p-value: %.3f' % (L.chi2, chi2-L.chi2, ndata, dof, pvalue(L.chi2, dof)))
        print ('bestfit:')
        L.print_bestfit(nuisance_array) 
        nuisance = {n: p for n, p in zip(L.nuisance_name, nuisance_array)}
        if return_lkl: return nuisance, L
        else: return nuisance

    def get_fake_data(self, cosmo=None, nuisance=None, nnlo=False):
        if cosmo is not None: self.update_cosmo(cosmo)
        if nuisance is None: nuisance = self.get_bestfit()
        original_cut_file = copy(self.c['cut_file'])
        self.c.update({'cut_file': 'cuts/nocuts.ini', 'marg': False, 'marg_chi2': False, 'synth': False})
        if nnlo: self.c['nnlo'] = True
        path_to_cut_file = os.path.join(self.c["data_path"], self.c["cut_file"])
        if not os.path.exists(path_to_cut_file): raise Exception('%s do not exist, cannot create fake data') % path_to_cut_file
        with hiddenprint(): L = Likelihood(self.c) # instantiate likelihood with no data cuts
        nuisance_array = np.array([v for n, v in nuisance.items() if n in L.nuisance_name_nonmarg])
        if nnlo: 
            pad = 2*L.e.Ng # (b1, c2)
            if L.extended: pad += 2*L.e.Ng # independent set of (b1, c2) for gs 
            if L.mag: pad += L.e.Ng
            if L.baryons: pad += L.e.Ng # b1i
            nuisance_array = np.insert(nuisance_array, pad, np.ones(4))
        loglkl = L.loglkl(nuisance_array, None, self.M, need_cosmo_update=True) 
        fssp, fssm, fgs, fgg = L.get_bestfit(nuisance_array)
        # assp, assm, ags, agg = L.angle_list
        # xssp, xssm, xgs, xgg = L.correlator_data_list
        # essp, essm, egs, egg = L.correlator_error_list
        fake_data_vector = np.concatenate([np.reshape(fssp, -1), np.reshape(fssm, -1), np.reshape(fgs, -1), np.reshape(fgg, -1)])
        header = "fit | chi2 = %.2f | chi2+prior = %.2f | parameters: " % (L.chi2, -2*loglkl)
        for key, value in self.cosmo.items(): header += "%s: %.5e, " % (key, value)
        for key, value in nuisance.items(): header += "%s: %.4e, " % (key, value)
        path_to_fake_file = os.path.join(self.c["data_path"], self.c["synth_file"])
        np.savetxt(path_to_fake_file, fake_data_vector, header=header)
        self.c['cut_file'] = original_cut_file # putting back the original scale cuts
        if nnlo: self.c['nnlo'] = False
        return nuisance

    def get_fisher(self, free_cosmo=None, cosmo=None, nuisance=None, free_nuisance=None, marg=True, derivatives=False): # Fisher
        is_synth = os.path.exists(os.path.join(self.c['data_path'], self.c['synth_file']))
        if not is_synth or cosmo is not None or nuisance is None: nuisance = self.get_fake_data(cosmo, nuisance)
        self.c.update({'marg': marg, 'marg_chi2': False, 'drop_logdet': marg, 'synth': True}) # we put the data on the model
        with hiddenprint(): L = Likelihood(self.c)
        _nuisance = {key: val for (key, val) in nuisance.items() if key in L.nuisance_name_nonmarg} if marg else nuisance
        if free_nuisance is None: free_nuisance = list(_nuisance.keys())
        delta = {key: 0.001 * nuisance[key] for key in free_nuisance} # 0.1% delta for nuisance
        if free_cosmo is not None: delta.update({key: 0.01 * self.cosmo[key] for key in free_cosmo}) # 1% for cosmological parameters
        def loglkl(nuisance, L=L, M=None, need_cosmo_update=False): 
            nuisance_array = np.array(list(nuisance.values()))
            return L.loglkl(nuisance_array, None, M, need_cosmo_update=need_cosmo_update)
        init = loglkl(_nuisance, L, self.M, True)
        
        def get_hessian(delta, nuisance, cosmo, free_cosmo=None, free_nuisance=None, derivatives=False, prior=False):
            # The matrix of second derivatives will be split in cosmo-cosmo, cosmo-nuisance, nuisance-nuisance.
            # H_{ij} = (𝐸(𝐱+ℎ𝑒̂𝑖+ℎ𝑒̂𝑗)−𝐸(𝐱−ℎ𝑒̂𝑖+ℎ𝑒̂𝑗)−𝐸(𝐱+ℎ𝑒̂𝑖−ℎ𝑒̂𝑗)+𝐸(𝐱−ℎ𝑒̂𝑖−ℎ𝑒̂𝑗) ) / 4ℎ^2
            # We also need to do an additional derivative of the Fisher! So 3rd derivative
            # vi = (f(x + hi*ei) - f(x - hi*ei))/(2 hi)
            # vij = (vi(x + hj*ej) - vi(x - hj*ej))/(2 hj)
            # = (f(x + hi*ei + hj*ej) - f(x - hi*ei + hj*ej) - f(x + hi*ei - hj*ej) + f(x - hi*ei - hj*ej))/(4 hi*hj)
            # vijk = (vij(x + hk * ek) - vij(x - hk * ek))/(2 hk)
            # = (f(x + hi*ei + hj*ej + hk*ek) - f(x - hi*ei + hj*ej + hk*ek) - f(x + hi*ei - hj*ej + hk*ek) + f(x - hi*ei - hj*ej + hk*ek)
            #    - f(x + hi*ei + hj*ej - hk*ek) + f(x - hi*ei + hj*ej - hk*ek) + f(x + hi*ei - hj*ej - hk*ek) - f(x - hi*ei - hj*ej - hk*ek)) )/(8 * hi*hj*hk)

            # nuisance-nuisance(-nuisance)
            der_nn = np.zeros((len(free_nuisance), len(free_nuisance)))
            if derivatives: der_nnn = np.zeros((len(free_nuisance), len(free_nuisance), len(free_nuisance)))

            for i, key1 in enumerate(free_nuisance):
                n_p1 = copy(nuisance); n_p1[key1] = nuisance[key1] + delta[key1]
                n_m1 = copy(nuisance); n_m1[key1] = nuisance[key1] - delta[key1]
                for j, key2 in enumerate(free_nuisance):
                    if j <= i:
                        n_p1p2 = copy(n_p1); n_p1p2[key2] = n_p1[key2] + delta[key2]
                        n_m1p2 = copy(n_m1); n_m1p2[key2] = n_m1[key2] + delta[key2]
                        n_p1m2 = copy(n_p1); n_p1m2[key2] = n_p1[key2] - delta[key2]
                        n_m1m2 = copy(n_m1); n_m1m2[key2] = n_m1[key2] - delta[key2]
                        loglklp1p2, loglklm1p2, loglklp1m2, loglklm1m2 = loglkl(n_p1p2), loglkl(n_m1p2), loglkl(n_p1m2), loglkl(n_m1m2)
                        der_nn[i][j] = 0.25 * (loglklp1p2 - loglklp1m2 - loglklm1p2 + loglklm1m2) / (delta[key1] * delta[key2])
                        der_nn[j][i] = 1. * der_nn[i][j]
                        if derivatives: 
                            for k, key3 in enumerate(free_nuisance):
                                if k <= j:
                                    n_p1p2p3 = copy(n_p1p2); n_p1p2p3[key3] = n_p1p2[key3] + delta[key3]
                                    n_p1m2p3 = copy(n_p1m2); n_p1m2p3[key3] = n_p1m2[key3] + delta[key3]
                                    n_m1p2p3 = copy(n_m1p2); n_m1p2p3[key3] = n_m1p2[key3] + delta[key3]
                                    n_m1m2p3 = copy(n_m1m2); n_m1m2p3[key3] = n_m1m2[key3] + delta[key3]
                                    n_p1p2m3 = copy(n_p1p2); n_p1p2m3[key3] = n_p1p2[key3] - delta[key3]
                                    n_p1m2m3 = copy(n_p1m2); n_p1m2m3[key3] = n_p1m2[key3] - delta[key3]
                                    n_m1p2m3 = copy(n_m1p2); n_m1p2m3[key3] = n_m1p2[key3] - delta[key3]
                                    n_m1m2m3 = copy(n_m1m2); n_m1m2m3[key3] = n_m1m2[key3] - delta[key3]
                                    # L.prior = False
                                    loglklp1p2p3, loglklp1m2p3, loglklm1p2p3, loglklm1m2p3, loglklp1p2m3, loglklp1m2m3, loglklm1p2m3, loglklm1m2m3 = loglkl(n_p1p2p3), loglkl(n_p1m2p3), loglkl(n_m1p2p3), loglkl(n_m1m2p3), loglkl(n_p1p2m3), loglkl(n_p1m2m3), loglkl(n_m1p2m3), loglkl(n_m1m2m3)
                                    L.prior = True
                                    der_nnn[i][j][k] = 0.125 * (loglklp1p2p3 - loglklp1m2p3 - loglklm1p2p3 + loglklm1m2p3 - loglklp1p2m3 + loglklp1m2m3 + loglklm1p2m3 - loglklm1m2m3) / (delta[key1] * delta[key2] * delta[key3])
                                    der_nnn[j][k][i] = 1. * der_nnn[i][j][k]
                                    der_nnn[k][i][j] = 1. * der_nnn[i][j][k]
                                    der_nnn[j][i][k] = 1. * der_nnn[i][j][k]
                                    der_nnn[i][k][j] = 1. * der_nnn[i][j][k]
                                    der_nnn[k][j][i] = 1. * der_nnn[i][j][k]

            if free_cosmo is not None: 
                # cosmo-cosmo/nuisance(-cosmo/nuisance)
                with hiddenprint(): 
                    M = Class(); M.set(self.class_settings); 
                    Mp = Class(); Mp.set(self.class_settings); Lp = Likelihood(self.c)
                    Mm = Class(); Mm.set(self.class_settings); Lm = Likelihood(self.c)
                    Mpp = Class(); Mpp.set(self.class_settings); Lpp = Likelihood(self.c)
                    Mmm = Class(); Mmm.set(self.class_settings); Lmp = Likelihood(self.c)
                    Mpm = Class(); Mpm.set(self.class_settings); Lpm = Likelihood(self.c)
                    Mmp = Class(); Mmp.set(self.class_settings); Lmm = Likelihood(self.c)

                # cosmo-cosmo and cosmo-nuisance
                der_cc = np.zeros((len(free_cosmo), len(free_cosmo)))
                der_cn = np.zeros((len(free_cosmo), len(free_nuisance)))
                if derivatives: 
                    der_ccc = np.zeros((len(free_cosmo), len(free_cosmo), len(free_cosmo)))
                    der_ccn = np.zeros((len(free_cosmo), len(free_cosmo), len(free_nuisance)))
                    der_cnn = np.zeros((len(free_cosmo), len(free_nuisance), len(free_nuisance)))

                for i, ckey1 in enumerate(free_cosmo):
                    c_p1 = copy(cosmo); c_p1[ckey1] = cosmo[ckey1] + delta[ckey1]
                    Mp.set(c_p1); Mp.compute(); loglkl(nuisance, Lp, Mp, True)
                    c_m1 = copy(cosmo); c_m1[ckey1] = cosmo[ckey1] - delta[ckey1]
                    Mm.set(c_m1); Mm.compute(); loglkl(nuisance, Lm, Mm, True)
                    # Do cosmo-nuisance
                    for j, key1 in enumerate(free_nuisance):
                        n_p1 = copy(nuisance); n_p1[key1] = nuisance[key1] + delta[key1]
                        n_m1 = copy(nuisance); n_m1[key1] = nuisance[key1] - delta[key1]
                        loglklp1p2, loglklm1p2, loglklp1m2, loglklm1m2 = loglkl(n_p1, Lp), loglkl(n_p1, Lm), loglkl(n_m1, Lp), loglkl(n_m1, Lm)
                        der_cn[i][j] = 0.25 * (loglklp1p2 - loglklp1m2 - loglklm1p2 + loglklm1m2) / (delta[ckey1] * delta[key1])
                        if derivatives: 
                            # cosmo-nuisance-nuisance
                            for k, key2 in enumerate(free_nuisance):
                                if k <= j:
                                    n_p1p2 = copy(n_p1); n_p1p2[key2] = n_p1[key2] + delta[key2]
                                    n_m1p2 = copy(n_m1); n_m1p2[key2] = n_m1[key2] + delta[key2]
                                    n_p1m2 = copy(n_p1); n_p1m2[key2] = n_p1[key2] - delta[key2]
                                    n_m1m2 = copy(n_m1); n_m1m2[key2] = n_m1[key2] - delta[key2]
                                    # Lp.prior, Lm.prior = False, False
                                    loglklp1p2p3, loglklp1m2p3, loglklm1p2p3, loglklm1m2p3, loglklp1p2m3, loglklp1m2m3, loglklm1p2m3, loglklm1m2m3 = loglkl(n_p1p2, Lp), loglkl(n_m1p2, Lp), loglkl(n_p1p2, Lm), loglkl(n_m1p2, Lm), loglkl(n_p1m2, Lp), loglkl(n_m1m2, Lp), loglkl(n_p1m2, Lm), loglkl(n_m1m2, Lm)
                                    Lp.prior, Lm.prior = True, True
                                    der_cnn[i][j][k] = 0.125 * (loglklp1p2p3 - loglklp1m2p3 - loglklm1p2p3 + loglklm1m2p3 - loglklp1p2m3 + loglklp1m2m3 + loglklm1p2m3 - loglklm1m2m3) / (delta[ckey1] * delta[key1] * delta[key2])
                                    der_cnn[i][k][j] = 1. * der_cnn[i][j][k]
                    # cosmo-cosmo            
                    for j, ckey2 in enumerate(free_cosmo):
                        if j <= i:
                            c_p1p2 = copy(c_p1); c_p1p2[ckey2] = c_p1p2[ckey2] + delta[ckey2]
                            c_m1p2 = copy(c_m1); c_m1p2[ckey2] = c_m1p2[ckey2] + delta[ckey2]
                            c_p1m2 = copy(c_p1); c_p1m2[ckey2] = c_p1m2[ckey2] - delta[ckey2]
                            c_m1m2 = copy(c_m1); c_m1m2[ckey2] = c_m1m2[ckey2] - delta[ckey2]
                            Mpp.set(c_p1p2); Mpp.compute(); loglklp1p2 = loglkl(nuisance, Lpp, Mpp, True) 
                            Mmp.set(c_m1p2); Mmp.compute(); loglklm1p2 = loglkl(nuisance, Lmp, Mmp, True) 
                            Mpm.set(c_p1m2); Mpm.compute(); loglklp1m2 = loglkl(nuisance, Lpm, Mpm, True) 
                            Mmm.set(c_m1m2); Mmm.compute(); loglklm1m2 = loglkl(nuisance, Lmm, Mmm, True) 
                            der_cc[i][j] = 0.25 * (loglklp1p2 - loglklp1m2 - loglklm1p2 + loglklm1m2) / (delta[ckey1] * delta[ckey2])
                            der_cc[j][i] = 1. * der_cc[i][j]
                            if derivatives: 
                                # cosmo-cosmo-nuisance
                                for k, key in enumerate(free_nuisance):
                                    n_p1 = copy(nuisance); n_p1[key] = n_p1[key] + delta[key]
                                    n_m1 = copy(nuisance); n_m1[key] = n_m1[key] - delta[key]
                                    # Lpp.prior, Lmp.prior, Lpm.prior, Lmm.prior = False, False, False, False
                                    loglklp1p2p3, loglklp1m2p3, loglklm1p2p3, loglklm1m2p3, loglklp1p2m3, loglklp1m2m3, loglklm1p2m3, loglklm1m2m3 = loglkl(n_p1, Lpp), loglkl(n_p1, Lpm), loglkl(n_p1, Lmp), loglkl(n_p1, Lmm), loglkl(n_m1, Lpp), loglkl(n_m1, Lpm), loglkl(n_m1, Lmp), loglkl(n_m1, Lmm), 
                                    Lpp.prior, Lmp.prior, Lpm.prior, Lmm.prior = True, True, True, True
                                    der_ccn[i][j][k] = 0.125 * (loglklp1p2p3 - loglklp1m2p3 - loglklm1p2p3 + loglklm1m2p3 - loglklp1p2m3 + loglklp1m2m3 + loglklm1p2m3 - loglklm1m2m3) / (delta[ckey1] * delta[ckey2] * delta[key])
                                    der_ccn[j][i][k] = 1. * der_ccn[i][j][k]
                                # cosmo-cosmo-cosmo
                                for k, ckey3 in enumerate(free_cosmo):
                                    if k <= j:
                                        c_p1p2p3 = copy(c_p1p2); c_p1p2p3[ckey3] = c_p1p2[ckey3] + delta[ckey3]
                                        c_p1m2p3 = copy(c_p1m2); c_p1m2p3[ckey3] = c_p1m2[ckey3] + delta[ckey3]
                                        c_m1p2p3 = copy(c_m1p2); c_m1p2p3[ckey3] = c_m1p2[ckey3] + delta[ckey3]
                                        c_m1m2p3 = copy(c_m1m2); c_m1m2p3[ckey3] = c_m1m2[ckey3] + delta[ckey3]
                                        c_p1p2m3 = copy(c_p1p2); c_p1p2m3[ckey3] = c_p1p2[ckey3] - delta[ckey3]
                                        c_p1m2m3 = copy(c_p1m2); c_p1m2m3[ckey3] = c_p1m2[ckey3] - delta[ckey3]
                                        c_m1p2m3 = copy(c_m1p2); c_m1p2m3[ckey3] = c_m1p2[ckey3] - delta[ckey3]
                                        c_m1m2m3 = copy(c_m1m2); c_m1m2m3[ckey3] = c_m1m2[ckey3] - delta[ckey3]
                                        # L.prior = False
                                        M.set(c_p1p2p3); M.compute(); loglklp1p2p3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_p1m2p3); M.compute(); loglklp1m2p3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_m1p2p3); M.compute(); loglklm1p2p3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_m1m2p3); M.compute(); loglklm1m2p3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_p1p2m3); M.compute(); loglklp1p2m3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_p1m2m3); M.compute(); loglklp1m2m3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_m1p2m3); M.compute(); loglklm1p2m3 = loglkl(nuisance, L, M, True) 
                                        M.set(c_m1m2m3); M.compute(); loglklm1m2m3 = loglkl(nuisance, L, M, True) 
                                        L.prior = True
                                        der_ccc[i][j][k] = 0.125 * (loglklp1p2p3 - loglklp1m2p3 - loglklm1p2p3 + loglklm1m2p3 - loglklp1p2m3 + loglklp1m2m3 + loglklm1p2m3 - loglklm1m2m3) / (delta[ckey1] * delta[ckey2] * delta[ckey3])
                                        der_ccc[j][k][i] = 1. * der_ccc[i][j][k]
                                        der_ccc[k][i][j] = 1. * der_ccc[i][j][k]
                                        der_ccc[j][i][k] = 1. * der_ccc[i][j][k]
                                        der_ccc[i][k][j] = 1. * der_ccc[i][j][k]
                                        der_ccc[k][j][i] = 1. * der_ccc[i][j][k]

            hessian = {'nn': der_nn}
            if derivatives: hessian['nnn'] = der_nnn
            if free_cosmo is not None: 
                hessian.update({'cn': der_cn, 'cc': der_cc, 'nc': der_cn.T})
                if derivatives: hessian.update({'ccc': der_ccc, 'cnn': der_cnn, 'ccn': der_ccn, 'cnc': np.swapaxes(der_ccn, 2,1), 'ncc': np.swapaxes(der_ccn, 2, 0), 'nnc': np.swapaxes(der_cnn, 0, 2), 'ncn': np.swapaxes(der_cnn, 0, 1)})
            return hessian

        fiducial = {}
        if free_cosmo is not None: fiducial.update({key: self.cosmo[key] for key in free_cosmo})
        fiducial.update({key: nuisance[key] for key in free_nuisance})
        F = {'fiducial': fiducial, 'cosmo': self.cosmo, 'nuisance': nuisance} 
        h = get_hessian(delta, _nuisance, self.cosmo, free_cosmo=free_cosmo, free_nuisance=free_nuisance, derivatives=derivatives)
        if free_cosmo is None: F['ij'] = -2. * h['nn']
        else: F['ij'] = -2. * np.block([[h['cc'], h['cn']], [h['nc'], h['nn']]])
        with np.printoptions(precision=3): print('Fisher eigvals: ', np.linalg.eigvalsh(F['ij']))

        if derivatives: 
            if free_cosmo is None: 
                F['ijk'] = -2. * h['nnn']
            else: 
                block_c = np.block([[h['ccc'], h['ccn']], [h['cnc'], h['cnn']]])
                block_n = np.block([[h['ncc'], h['ncn']], [h['nnc'], h['nnn']]])
                F['ijk'] = -2. * np.concatenate([block_c, block_n], axis=0) # Block works for the innermost dimensions. So the k of the derivative F_{ij,k} is actually the first index here; think of this as \de_k F_{ij}

        return F

    def get_fisher_and_jeffreys(self, free_cosmo=None, cosmo=None, nuisance=None, free_nuisance=None, marg=True, config=None):
        F = self.get_fisher(free_cosmo=free_cosmo, cosmo=cosmo, nuisance=nuisance, free_nuisance=None, marg=marg, derivatives=True)
        shifts = np.einsum('ij,kij->k', np.linalg.inv(F['ij']), F['ijk'])
        names = free_cosmo + list(F['nuisance'].keys()) if free_cosmo is not None else list(F['nuisance'].keys())
        shift_dict = {n: np.round(s, 4) for n, s in zip(names, shifts)}
        if config is not None: pass
        return F, shift_dict

    def get_gaussian_samples(self, fisher, size=1e4):
        names = [key for key in fisher['fiducial'].keys()]
        means = [val for val in fisher['fiducial'].values()]
        samples = np.random.multivariate_normal(means, np.linalg.inv(fisher['ij']), size=int(size))
        return names, samples

                        


