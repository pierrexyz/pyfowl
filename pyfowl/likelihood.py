import numpy as np
import scipy.constants as conts
import yaml
import os
from astropy.io import fits
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.linalg import block_diag

from pyfowl.extract import TwoPointExtraction
from pyfowl.correlator import Correlator
from pyfowl.utils import *


class Likelihood(object):
    """EFT Likelihood"""
    
    def __init__(self, config, verbose=True):

        # Load config file
        self.c = config
        self.__load_data_and_config()

        # Loading correlator engine
        self.e = Correlator(self.t, self.zz, self.nlens, self.nsource, limber=self.limber, limber_2=self.limber_2, limber_gs=self.limber_gs, shear_beyond_limber=self.shear_beyond_limber, curved_sky=self.curved_sky, halofit=self.halofit, 
            rsd=self.rsd, mag=self.mag, baryons=self.baryons, nnlo=self.nnlo, photoz=self.photoz, zlens_mean=self.zlens_mean, intrinsic_alignments=self.intrinsic_alignments, shear_calibration=self.shear_calibration,
            binning=self.binning, theta_min=self.tmin, theta_max=self.tmax,
            fftpad=self.fftpad, 
            extended=self.extended, load=False, save=False, knl=0.7, km=0.7) 

        # redefine precision matrix from point mass analytic marginalisation
        if self.point_mass or self.point_mass_marg: self.set_point_mass(marg=self.point_mass_marg)

        # define optipath for einsum for acceleration
        self.__load_boost()

        if self.prior:
            # Loading EFT inverse correlation prior matrix
            self.__load_prior_inv_corr_mat()

            # Loading prior mean for Gaussian parameters
            self.__set_prior_mean_gauss()

            # Loading priors for marginalised Gaussian parameters
            if self.marg: self.__load_prior_gauss_marg()

        # number of unmarginalised, names of unmarginalised, and names of all, nuisance parameters
        self.N, self.nuisance_name_nonmarg, self.nuisance_name = self.__count_and_name_params()

        # CLASSy settingss
        log10kmax = 0
        if self.halofit: log10kmax = 1 # slower, but required to get good high-k asymptote of pk halofit
        if self.fftpad: log10kmax = 2 # slower, but required to get good high-k asymptote when we pad with 0 the input pk to FFTLog instead of extrapolating
        self.kk = np.logspace(-5, log10kmax, 300) 
        self.class_settings = {'output': 'mPk,dTk', 'z_max_pk' : self.zmax, 'P_k_max_1/Mpc': 10.**log10kmax, 'non linear':'halofit'}

        self.first_evaluation = True

    def __load_data_and_config(self):

        # path to data .fits
        path_to_data = os.path.join(self.c["data_path"], self.c["data_file"])
        path_to_cut = os.path.join(self.c["data_path"], self.c["cut_file"])

        # Load data, covariance and mask
        extract = TwoPointExtraction(path_to_data, path_to_cut)
        self.ydata = extract.data_y # ['xip', 'xim', 'gammat', 'wtheta']
        self.mask = np.concatenate(extract.two_point_data.masks)
        if "synth" in self.c: 
            if self.c["synth"]: 
                path_to_synth = os.path.join(self.c["data_path"], self.c["synth_file"])
                self.ydata = np.loadtxt(path_to_synth, usecols=0)[self.mask] # synthetic data for testing
        self.cov = extract.cov
        if "diagcov" in self.c: 
            if self.c['diagcov']: 
                self.cov = np.diag(np.diag(self.cov))
        self.invcov = np.linalg.inv(self.cov)
        
        self.chi2data = np.dot(self.ydata, np.dot(self.invcov, self.ydata))
        self.invcovdata = np.dot(self.ydata, self.invcov)
        
        # Load angles theta radial selection functions nlens, nsource
        des = fits.open(path_to_data)

        tam = np.empty(shape=(20)); tmin = 1.*tam; tmax = 1.*tam # angle theta in arcmin
        for i, line in enumerate(des['wtheta'].data):
            if "21" or "buzzard" in self.c["data_file"]: bin1, bin2, angbin, _, angmin, angmax, ang, npairs = line # 2021 = DESY3
            else: bin1, bin2, angbin, val, ang, npairs = line
            if i < 20: 
                tam[i] = ang; tmin[i] = angmin; tmax[i] = angmax
        self.t = tam * np.pi/(60. * 180.); self.tmin = tmin * np.pi/(60. * 180.); self.tmax = tmax * np.pi/(60. * 180.); # angle theta in radians
        # print ('tmin, t, tmax', np.vstack([self.tmin, self.t, self.tmax]).T)

        if "maglim" in self.c["data_file"]: Bl = 6
        else: Bl = 5
        self.Bl = Bl
        Nl = des['nz_lens'].data.shape[0]
        zl = np.empty(shape=(Nl))
        nl = np.empty(shape=(Bl,Nl))
        zlens_mean = np.empty(shape=(Bl))
        for i, line in enumerate(des['nz_lens'].data):
            if Bl == 5: zlow, zmid, zhigh, bin1, bin2, bin3, bin4, bin5 = line
            elif Bl == 6: zlow, zmid, zhigh, bin1, bin2, bin3, bin4, bin5, bin6 = line
            zl[i] = zmid
            for j in range(Bl): nl[j,i] = line[3+j]/(zhigh-zlow)
        for j in range(Bl): 
            nl[j] /= np.trapz(nl[j], x=zl)
            zlens_mean[j] = np.trapz(zl * nl[j], x=zl)
        
        Ns = des['nz_source'].data.shape[0]
        zs = np.empty(shape=(Ns))
        ns = np.empty(shape=(4,Ns))
        zsource_mean = np.empty(shape=(4))
        for i, line in enumerate(des['nz_source'].data):
            zlow, zmid, zhigh, bin1, bin2, bin3, bin4 = line
            zs[i] = zmid
            for j in range(4): ns[j,i] = line[3+j]/(zhigh-zlow)
        for j in range(4): 
            ns[j] /= np.trapz(ns[j], x=zs)
            zsource_mean[j] =  np.trapz(zs * ns[j], x=zs)
        
        self.zfid = 0.525 # DES 'central' redshift
        self.zmax = np.min([np.max(zl), np.max(zs)])
        if self.zmax > 4: self.zmax = 4. # DES 'maximal' redshift

        zz = zs[zs<self.zmax] 
        self.zz = zz[1:] 
        self.nlens = interp1d(zl, nl, kind='cubic', axis=-1)(self.zz)
        self.nsource = interp1d(zs, ns, kind='cubic', axis=-1)(self.zz)
        
        # Load config
        self.marg = self.c["marg"] if "marg" in self.c else False
        self.marg_chi2 = self.c["marg_chi2"] if "marg_chi2" in self.c else False
        # self.with_derived_bias = self.c["with_derived_bias"]

        self.binning = self.c["binning"] if "binning" in self.c else False
        self.fftpad = self.c["fftpad"] if "fftpad" in self.c else False
        self.extended = self.c["extended"] if "extended" in self.c else False
        self.limber = self.c["limber"] if "limber" in self.c else False
        self.limber_2 = self.c["limber_2"] if "limber_2" in self.c else False
        self.limber_gs = self.c["limber_gs"] if "limber_gs" in self.c else False
        self.shear_beyond_limber = self.c["shear_beyond_limber"] if "shear_beyond_limber" in self.c else False
        if "buzzard" in self.c["data_file"]: 
            print ('Buzzard: setting limber in galaxy-shear')
            self.limber_gs = True
        self.curved_sky = self.c["curved_sky"] if "curved_sky" in self.c else False
        self.halofit = self.c["halofit"] if "halofit" in self.c else False # halofit only for linear power spectra

        self.rsd = self.c["rsd"] if "rsd" in self.c else False
        self.mag = self.c["mag"] if "mag" in self.c else False
        self.baryons = self.c["baryons"] if "baryons" in self.c else False
        self.nnlo = self.c["nnlo"] if "nnlo" in self.c else False
        self.photoz = self.c["photoz"] if "photoz" in self.c else False
        self.intrinsic_alignments = self.c["intrinsic_alignments"] if "intrinsic_alignments" in self.c else False
        self.shear_calibration = self.c["shear_calibration"] if "shear_calibration" in self.c else False
        self.point_mass = self.c["point_mass"] if "point_mass" in self.c else False
        self.point_mass_marg = self.c["point_mass_marg"] if "point_mass_marg" in self.c else False
        if self.point_mass_marg: self.point_mass = False # can't do both the same time yo

        self.prior = self.c["prior"]
        self.nongaussian_prior = self.c["prior"] 
        self.redcorr = self.c["redcorr"] 
        self.prior_sigma = self.c["prior_sigma"]
        self.prior_mean_xssp = self.c["prior_mean_xssp"] if "prior_mean_xssp" in self.c else None
        self.prior_mean_xssm = self.c["prior_mean_xssm"] if "prior_mean_xssm" in self.c else None
        self.prior_mean_bmag = self.c["prior_mean_bmag"] if "prior_mean_bmag" in self.c else 0.
        self.prior_mean_gal_on_lagrangian_bias = self.c["prior_mean_gal_on_lagrangian_bias"] if "prior_mean_gal_on_lagrangian_bias" in self.c else False
        if self.prior: 
            self.c2_prior_mean = 0. # will be set dynamically to lagrangian bias relation as function of b1 if self.prior_mean_gal_on_lagrangian_bias is True
            if self.redcorr: 
                self.corr_sigma = self.c["corr_sigma"] if "corr_sigma" in self.c else 0.
                self.corr_sigma_ss = self.c["corr_sigma_ss"] if "corr_sigma_ss" in self.c else self.corr_sigma
            if self.extended: self.corr_sigma_gg_gs = self.c["corr_sigma_gg_gs"] if "corr_sigma_gg_gs" in self.c else 0.
        self.perturbativity = self.c["perturbativity"] if "perturbativity" in self.c else False

        self.drop_logdet = self.c['drop_logdet'] if 'drop_logdet' in self.c else False
        self.jeffrey = self.c["jeffrey"] if "jeffrey" in self.c else False 
        if self.jeffrey: 
            self.drop_logdet = True
            self.jeffrey_param_shift = self.c["jeffrey_param_shift"] # dictionary
            self.jeffrey_param_shift_values = np.array([v for n, v in self.jeffrey_param_shift.items()])

        if self.photoz:
            self.photoz_source_dz_mean = np.array(self.c["photoz_source_dz_mean"])
            self.photoz_source_dz_sigma = np.array(self.c["photoz_source_dz_sigma"])
            self.photoz_lens_dz_mean = np.array(self.c["photoz_lens_dz_mean"])
            self.photoz_lens_dz_sigma = np.array(self.c["photoz_lens_dz_sigma"])
            self.photoz_lens_sz_mean = np.array(self.c["photoz_lens_sz_mean"])
            self.photoz_lens_sz_sigma = np.array(self.c["photoz_lens_sz_sigma"])
            self.zlens_mean = zlens_mean
        else:
            self.zlens_mean = None

        if self.shear_calibration:
            self.shear_calibration_mean = np.array(self.c["shear_calibration_mean"])
            self.shear_calibration_sigma = np.array(self.c["shear_calibration_sigma"])

        # for plotting 
        self.multimask = extract.two_point_data.masks
        self.indexdict = extract.indexdict
        self.indexbin = [list(zip(extract.two_point_data.spectra[i].bin1, extract.two_point_data.spectra[i].bin2)) for i in range(4)]
        self.binpairs = [extract.two_point_data.spectra[i].bin_pairs for i in range(4)]
        self.angle = extract.angle
        
        self.correlator_data = self.ydata#[spectrum.value for spectrum in extract.two_point_data.spectra]
        self.correlator_data_list = self.create_list(self.correlator_data)
        self.correlator_error_list = self.create_list(np.sqrt(np.diag(self.cov)))
        self.angle_list = self.create_list(self.angle)
        # self.angle_min_clustering = np.array([angle[0] for angle in self.angle_list[-1]])

    def set_point_mass(self, marg=True): 
        import astropy.units as u
        import astropy.constants as const
        
        zz, nlens, nsource = self.e.z_thin_arr, self.e.nlens, self.e.nsource

        Hubble_fac = 1 / (const.c.to(u.km/u.s)).value
        def Hubble(z, Omega_m=0.3, H0=67): return H0 * Hubble_fac * (Omega_m * (1+z)**3 + (1-Omega_m))**0.5
        zvec = np.arange(0., 4., 0.001)
        def comoving_distance(z): return np.trapz(np.heaviside(z-zvec, 0) / Hubble(zvec), x=zvec, axis=-1)
        rz = np.array([comoving_distance(z) for z in zz]); rz[0] = rz[1] # to avoid nan and inf
        sigma_crit_fac = const.c**2 / (4 * np.pi * const.G)
        sigma_crit_fac_Mpc_Msun = (sigma_crit_fac.to(u.M_sun/u.Mpc)).value
        rl, rs = np.meshgrid(rz, rz, indexing='ij')
        def beta_integrand(rl, rs): return sigma_crit_fac_Mpc_Msun**-1 * (rs - rl) / rs / rl
        beta_integrand_mesh = beta_integrand(rl, rs)
        def beta(nl, ns): return np.trapz(np.trapz(nl * beta_integrand_mesh, axis=0), axis=-1)
        beta_ls = np.array([[beta(nl, ns) for ns in nsource] for nl in nlens])

        def t(beta, j_l=-1, sigma_B=1e17): # 2105.13545, p. 19: sigma_B = 10000 of 2.5 x 1e13 / pi M_solar / h (see app. A), so ~ 1e17 M_solar / h in total, OK!
            ugs = np.zeros(self.multimask[2].shape[0]).reshape(self.e.Ng, self.e.Ns, -1)
            if j_l == -1: # for nonmarg, here beta = beta_ls
                for j_l, beta_s in enumerate(beta): 
                    for i_s, b in enumerate(beta_s): ugs[j_l, i_s] = sigma_B * b / self.t**2
                return ugs
            else: # for marg, here beta = beta_s
                for i_s, b in enumerate(beta): ugs[j_l, i_s] = sigma_B * b / self.t**2 
                return ugs.reshape(-1)

        def u(beta_ls): 
            ussp, ussm, ugg = np.zeros(self.multimask[0].shape[0]), np.zeros(self.multimask[1].shape[0]), np.zeros(self.multimask[3].shape[0]) 
            u_ = np.array([np.concatenate([ussp, ussm, t(beta_s, j_l), ugg])[self.mask] for j_l, beta_s in enumerate(beta_ls)])
            if "noz56" in self.c["cut_file"]: u_ = u_[:4]
            return u_.T

        def marg_precision_matrix(P, U): # P: precision matrix ~ C^-1, U: marginal errors
            PU = np.einsum('ij,jl->il', P, U)
            UPU = np.einsum('li,im->lm', U.T, PU)
            return P - np.einsum('il,lm,mj->ij', PU, np.linalg.inv(np.eye(UPU.shape[0])+UPU), PU.T)

        if marg: self.invcov = marg_precision_matrix(self.invcov, u(beta_ls))
        else: self.pm, self.sigma_pm = t(beta_ls, sigma_B=1e8), 1e9 # shape (Ng x Ns, N_theta): somehow this seems to make the parameter order 1, so we use this to stablise the minimisation
        return 

    def __load_prior_inv_corr_mat(self):
        if not self.redcorr: # default, no correlation
            corrgg = np.eye(self.e.Ng)
            self.prior_nongauss_inv_corr_mat = corrgg
            # Gaussian parameters are organized as [css+, css-] (block1), [cgs] (block2), [cgg, b3] (block3)
            block1, block2, block3 = np.eye(2*self.e.Nss), np.eye(self.e.Ng * self.e.Ns), np.eye(2*self.e.Ng)
        
        else: # redshift bin correlation
            rhog = np.array([1-(1-self.corr_sigma)**2/2.] * (self.e.Ng-1))
            rhos_gs = np.array([1-(1-self.corr_sigma)**2/2.] * (self.e.Ns-1)) 
            rhos_ss = np.array([1-(1-self.corr_sigma_ss)**2/2.] * (self.e.Ns-1))
            
            corrgg = corrmat(rhog) # (b_z1, b_z2, ...) where b is a galaxy bias

            # NonGaussian (b1_z1, b1_z2, ...) / (c2_z1, c2_z2, ...)
            self.prior_nongauss_inv_corr_mat = np.linalg.inv(corrgg)  # This is only 1 as then it's applied to b1, c2 separately
            
            # ss ct corr
            #self.corr_sigma_ct = 0.8
            #rho_ct= 1-(1-self.corr_sigma_ct)**2/2.
            rho_ct = 0. 

            # Gaussian
            corrss, corrgs = corrmatss(rhos_ss), corrmatgs(rhog, rhos_gs)
            block1, block2, block3 = np.block([[corrss, rho_ct*corrss], [rho_ct*corrss, corrss]]), corrgs, np.block([[corrgg, 0.*corrgg], [0.*corrgg, corrgg]])
        
        if self.extended: # (b1, c2, b3) correlation between gg and gs
            rho_gg_gs = 1-(1-self.corr_sigma_gg_gs)**2/2.
            
            # NonGaussian (b1_gg, b1_gs) / (c2_gg, c2_gs)
            self.prior_nongauss_inv_crosscorr_mat = np.linalg.inv(np.block([[corrgg, rho_gg_gs*corrgg],
                                                                            [rho_gg_gs*corrgg, corrgg]]))  # This correlates b1 - b1gs
            # Gaussian (cgg, b3_gg, b3_gs)
            block3 = np.block([[corrgg, 0.*corrgg, 0.*corrgg],
                                [0.*corrgg, corrgg, rho_gg_gs*corrgg],
                                [0.*corrgg, rho_gg_gs*corrgg, corrgg]]) 
        
        # Gaussian parameters are organized as [css+, css-] (block1), [cgs] (block2), [cgg, b3] (block3)
        self.prior_gauss_inv_corr_mat = np.linalg.inv(np.block([[block1, np.zeros((len(block1), len(block2)+len(block3)))],
                                                      [np.zeros((len(block2), len(block1))), block2, np.zeros((len(block2), len(block3)))],
                                                      [np.zeros((len(block3), len(block1) + len(block2))), block3]]))

        if self.nnlo: self.prior_nnlo_inv_corr_mat = np.eye(4)

    def __set_prior_mean_gauss(self):
        self.bg_prior_mean = np.zeros(shape=(self.e.Nmarg))
        if self.prior_mean_xssp is not None: self.bg_prior_mean[:self.e.Nss] = self.prior_mean_xssp
        if self.prior_mean_xssm is not None: self.bg_prior_mean[self.e.Nss:2*self.e.Nss] = self.prior_mean_xssm

    def __set_prior_mean_gauss_marg(self):
        self.F1_bg_prior_mean = np.einsum('a,ab->b', self.bg_prior_mean, self.F2_bg_prior_matrix, optimize=self.optipath_bg) 
        self.chi2_bg_prior_mean = np.einsum('a,b,ab->', self.bg_prior_mean, self.bg_prior_mean, self.F2_bg_prior_matrix, optimize=self.optipath_chi2)
        return

    def __load_prior_gauss_marg(self):            
        self.F2_bg_prior_matrix = self.prior_sigma**-2 * self.prior_gauss_inv_corr_mat
        self.__set_prior_mean_gauss_marg()
        return

    def __set_prior_mean_gal_on_lagrangian_bias(self, b1=2., b1_gs=2.): # see e.g. (3.17) in https://arxiv.org/pdf/2404.07272
        def set_c2(b1): return 1 - 7/5. * (b1 - 1) # c2 = (b2 + b4)/2 (= b2 = b4 since c4 = b2-b4 = 0)
        def set_b3(b1): return (294 - 1015 * (b1 - 1)) / 441.
        if self.prior: 
            if self.extended: 
                b1_gg_gs = np.concatenate([b1, b1_gs])
                self.c2_prior_mean = set_c2(b1_gg_gs)
                self.bg_prior_mean[-2*self.e.Ngg:] = set_b3(b1_gg_gs)
            else: 
                self.c2_prior_mean = set_c2(b1)
                self.bg_prior_mean[-self.e.Ngg:] = set_b3(b1)
        if self.marg: self.__set_prior_mean_gauss_marg() 
        return

    def __load_prior_perturbativity(self, tol_frac=1.):
        self.e._get_xi_loop_scaling()
        xi_1loop_scaling = np.concatenate([self.e.xi_ssp_1loop_scaling.reshape(-1), self.e.xi_ssm_1loop_scaling.reshape(-1), self.e.xi_gs_1loop_scaling.reshape(-1), self.e.xi_gg_1loop_scaling.reshape(-1)])[self.mask]
        xi_2loop_scaling = np.concatenate([self.e.xi_ssp_2loop_scaling.reshape(-1), self.e.xi_ssm_2loop_scaling.reshape(-1), self.e.xi_gs_2loop_scaling.reshape(-1), self.e.xi_gg_2loop_scaling.reshape(-1)])[self.mask]
        delta_chi2_2loop = self.__chi2(xi_2loop_scaling, self.invcov)

        _, _, name_params = self.__count_and_name_params()
        ndata = len(self.ydata)
        dof = ndata # - len(name_params) # number of degrees of freedom
        norm_factor = (tol_frac * 2 * dof / delta_chi2_2loop)**.5 # since A^2 \Delta \chi^2 ~ X * Var(\chi^2) with Var(\chi^2) = 2 dof

        self.xi_1loop_size = norm_factor * xi_1loop_scaling * ndata**.5 # we put ndata^0.5 so that the chi2 penalisation is 1 overall when the loop saturates this expectation: prior ~ \sum_{data} (loop/expectation)^2 / ndata
        self.xi_1loop_size_invcov = np.diag(self.xi_1loop_size**-2)
        return

    def __load_boost(self): 
        self.optipath_chi2 = np.einsum_path('a,b,ab->', self.ydata, self.ydata, self.invcov, optimize='optimal')[0]
        if self.marg: 
            dummy_ak = np.zeros(shape=(self.e.Nmarg, self.ydata.shape[0]))
            self.optipath_F2 = np.einsum_path('ak,bp,kp->ab', dummy_ak, dummy_ak, self.invcov, optimize='optimal')[0]
            self.optipath_F1 = np.einsum_path('ak,p,kp->a', dummy_ak, self.ydata, self.invcov, optimize='optimal')[0]
            self.optipath_bg = np.einsum_path('a,ab->b', self.ydata, self.invcov, optimize='optimal')[0] 
        return

    def __count_and_name_params(self):
        name = ['b1_%s' % (i+1) for i in range(self.e.Ng)] + ['c2_%s' % (i+1) for i in range(self.e.Ng)]
        if self.extended: name += ['b1gs_%s' % (i+1) for i in range(self.e.Ng)] + ['c2gs_%s' % (i+1) for i in range(self.e.Ng)]
        if self.mag: name += ['bmag_%s' % (i+1) for i in range(self.e.Ng)]
        if self.baryons: name += ['bai_%s' % (i+1) for i in range(self.e.Ng)]
        if self.nnlo: name += ['bnnlo_gg', 'bnnlo_gs', 'bnnlo_ss+', 'bnnlo_ss-']
        if self.photoz: name += ['dzs_%s' % (i+1) for i in range(self.e.Ns)] + ['dzl_%s' % (i+1) for i in range(self.e.Ng)] + ['szl_%s' % (i+1) for i in range(self.e.Ng)]
        if self.intrinsic_alignments: name += ['A', 'alpha']
        if self.shear_calibration: name += ['ms_%s' % (i+1) for i in range(self.e.Ns)]
        if self.point_mass: name += ['pm_%s' % (i+1) for i in range(self.e.Ng)]
        if self.marg: N = len(name)
        name += ['css+_%s' % (i+1) for i in range(self.e.Nss)] + ['css-_%s' % (i+1) for i in range(self.e.Nss)] + ['cgs_%s' % (i+1) for i in range(self.e.Ngs)] + ['cgg_%s' % (i+1) for i in range(self.e.Ng)] + ['b3_%s' % (i+1) for i in range(self.e.Ng)] 
        if self.extended: name += ['b3gs_%s' % (i+1) for i in range(self.e.Ng)] 
        if not self.marg: N = len(name)
        return N, name[:N], name 
        
    def set_engine(self, bval, bg=None, loop_only=False):
        self.e.set_3x2pt(bval, bg=bg, marg=self.marg, loop_only=loop_only)

    def get_correlator_readable(self, bval):
        correlator = self.get_correlator(bval)
        return self.create_list(correlator)

    def get_correlator(self, bval, bg=None, loop_only=False):
        self.set_engine(bval, bg, loop_only=loop_only)
        Xgs = 1. * self.e.Xgs
        if self.point_mass and not loop_only: Xgs += np.einsum('l,lst->lst', self.B_pm, self.pm).reshape(self.e.Ng * self.e.Ns, -1)
        correlator = np.concatenate([self.e.Xssp.reshape(-1), self.e.Xssm.reshape(-1), Xgs.reshape(-1), self.e.Xgg.reshape(-1)])
        return correlator[self.mask]

    def get_correlator_marg(self, bval):
        correlator_marg = self.e.get_marg(bval)
        return correlator_marg[:,self.mask]

    def get_1loop(self, bval, bg=None, marg=False): 
        xi_1loop = self.get_correlator(bval, bg=bg, loop_only=True)
        if not marg: return xi_1loop
        else:
            xi_1loop_marg = self.get_correlator_marg(bval) # so for now, all marg parameters enters only in the loop, so we don't need to ask anything special with respect to the standard case
            return xi_1loop, xi_1loop_marg

    def __chi2(self, T_k, P):
        chi2 = np.einsum('k,p,kp->', T_k, T_k, P, optimize=self.optipath_chi2) # 
        return chi2

    # def __chi2_marg(self, Tng_k, Tg_bk, P): 
    #     """Marginalized chi2"""
    #     F2 = np.einsum('ak,bp,kp->ab', Tg_bk, Tg_bk, P, optimize=self.optipath_F2) 
    #     F1 = np.einsum('ak,p,kp->a', Tg_bk, Tng_k, P, optimize=self.optipath_F1)
    #     F0 = self.__chi2(Tng_k, P)

    #     if self.prior:
    #         F1 -= self.F1_bg_prior_mean
    #         F2 += self.F2_bg_prior_matrix

    #     invF2 = np.linalg.inv(F2)

    #     chi2 = F0 - np.einsum('a,b,ab->', F1, F1, invF2, optimize=self.optipath_chi2) 
    #     if not self.drop_logdet: chi2 += np.linalg.slogdet(F2)[1] 
    #     if self.prior: chi2 += self.chi2_bg_prior_mean
    #     bg = - np.einsum('a,ab->b', F1, invF2, optimize=self.optipath_bg) 

    #     return chi2, bg
    def __chi2_marg(self, Tng_k, Tg_bk, P): 

        def get_F(Tng_k, Tg_bk, P):
            F2 = np.einsum('ak,bp,kp->ab', Tg_bk, Tg_bk, P, optimize=self.optipath_F2)
            F1 = np.einsum('ak,p,kp->a', Tg_bk, Tng_k, P, optimize=self.optipath_F1)
            F0 = self.__chi2(Tng_k, P) 
            return F2, F1, F0

        for i, (Tng_i, Tg_i, P_i) in enumerate(zip(Tng_k, Tg_bk, P)):
            F2_i, F1_i, F0_i = get_F(Tng_i, Tg_i, P_i)
            if i == 0: F2 = F2_i; F1 = F1_i; F0 = F0_i
            else: F2 += F2_i; F1 += F1_i; F0 += F0_i

        if self.prior:
            F1 -= self.F1_bg_prior_mean
            F2 += self.F2_bg_prior_matrix

        invF2 = np.linalg.inv(F2)
        chi2 = F0 - np.einsum('a,b,ab->', F1, F1, invF2, optimize=self.optipath_chi2) 
        if not self.drop_logdet: chi2 += np.linalg.slogdet(F2)[1] 
        if self.prior: chi2 += self.chi2_bg_prior_mean
        bg = - np.einsum('a,ab->b', F1, invF2, optimize=self.optipath_bg) 
        
        return chi2, bg

    def get_chi2(self, bval, bg=None):
        correlator = self.get_correlator(bval, bg)
        chi2 = self.__chi2(correlator-self.ydata, self.invcov)
        self._chi2 = 1. * chi2
        if self.perturbativity:
            xi_1loop = self.get_1loop(bval, bg=bg) # 1loop, bin-weighted diagonal precision matrix of inverse linear squared 
            chi2 += np.sum((xi_1loop/self.xi_1loop_size)**2)
        return chi2

    def get_chi2_marg(self, bval, bg=None):
        correlator = self.get_correlator(bval, bg)
        correlator_marg = self.get_correlator_marg(bval) # bval[:5] = [b1_1, ..., b1_5]
        Tng_k, Tg_bk, p = [correlator-self.ydata], [correlator_marg], [self.invcov]

        if self.perturbativity: 
            xi_1loop, xi_1loop_marg = self.get_1loop(bval, bg=bg, marg=True) # 1loop, bin-weighted diagonal precision matrix of inverse linear squared 
            Tng_k.append(xi_1loop); Tg_bk.append(xi_1loop_marg); p.append(self.xi_1loop_size_invcov)
        
        chi2, bg = self.__chi2_marg(Tng_k, Tg_bk, p) 
        # chi2, bg = self.__chi2_marg(correlator-self.ydata, correlator_marg, self.invcov)
        return chi2, bg

    def loglkl_(self, cosmo_param, cosmo_name, nuisance_param, nuisance_name, class_engine, need_cosmo_update=True):
        loglkl = self.loglkl(nuisance_param, nuisance_name, class_engine, need_cosmo_update=need_cosmo_update)
        if self.jeffrey: 
            p = np.array([c for c, n in zip(cosmo_param, cosmo_name) if n in self.jeffrey_param_shift] + 
                         [c for c, n in zip(nuisance_param, nuisance_name) if n in self.jeffrey_param_shift])
            loglkl += 0.5 * np.sum(self.jeffrey_param_shift_values * p)
        return loglkl

    def loglkl(self, bval, free_b_name, class_engine, need_cosmo_update=True, fast=False): 
        # if self.with_derived_bias: data.derived_lkl = {}
        id_0 = 2*self.e.Ng # (b1, c2)
        if self.extended: id_0 += 2*self.e.Ng # independent set of (b1, c2) for gs 
        if need_cosmo_update or self.intrinsic_alignments or self.photoz: 
        # if need_cosmo_update: 
            self.pk_l, ilogpk_nl, rz, dz_by_dr, Dz, Dfid, fz, ffid, h, Omega0_m, pai = self.set_cosmo(module='class', engine=class_engine)
            for o, pad in zip([self.mag, self.baryons, self.nnlo], [self.e.Ng, self.e.Ng, 4]): 
                if o: id_0 += pad
            if self.photoz:
                pad = self.e.Ns; dzs = bval[id_0:id_0+pad]; id_0 += pad
                pad = self.e.Ng; dzl = bval[id_0:id_0+pad]; id_0 += pad
                pad = self.e.Ng; szl = bval[id_0:id_0+pad]; id_0 += pad
            else:
                dzs, dzl, szl = [None, None, None]
            if self.intrinsic_alignments:
                pad = 2; A, alpha = bval[id_0:id_0+pad]; id_0 += pad
            else:
                A, alpha = [0., 1.]
            if not fast: self.e.compute(self.kk, self.pk_l, ilogpk_nl, rz, dz_by_dr, Dz, Dfid, fz, ffid, h, Omega0_m, dzs=dzs, dzl=dzl, szl=szl, Pai=pai, A=A, alpha=alpha)
        else: pass
        if self.shear_calibration: # do this externally of self.e.compute() since these are fast parameters
            if not (need_cosmo_update or self.intrinsic_alignments or self.photoz): 
                for o, pad in zip([self.mag, self.baryons, self.nnlo, self.photoz, self.intrinsic_alignments], [self.e.Ng, self.e.Ng, 4, self.e.Ns + 2*self.e.Ng, 2]): 
                    if o: id_0 += pad
            pad = self.e.Ns; ms = bval[id_0:id_0+pad]; id_0 += pad
            self.e.do_shear_calibration(ms)
        if self.point_mass: 
            if not (need_cosmo_update or self.intrinsic_alignments or self.photoz or self.shear_calibration): 
                for o, pad in zip([self.mag, self.baryons, self.nnlo, self.photoz, self.intrinsic_alignments, self.shear_calibration], [self.e.Ng, self.e.Ng, 4, self.e.Ns + 2*self.e.Ng, 2, self.e.Ns]): 
                    if o: id_0 += pad
            pad = self.e.Ng; self.B_pm = bval[id_0:id_0+pad]; id_0 += pad

        if self.first_evaluation and self.perturbativity: 
            self.__load_prior_perturbativity()
            self.first_evaluation = False

        if self.prior and self.prior_mean_gal_on_lagrangian_bias: 
            b1 = bval[:self.e.Ng]
            b1_gs = bval[2*self.e.Ng:3*self.e.Ng] if self.extended else b1
            self.__set_prior_mean_gal_on_lagrangian_bias(b1, b1_gs)
        if self.marg:
            self.chi2, self.bg = self.get_chi2_marg(bval)
            if self.marg_chi2: self.chi2 = self.get_chi2(bval, bg=self.bg)
            # if self.with_derived_bias:
            #     for i, elem in enumerate(data.get_mcmc_parameters(['derived_lkl'])):
            #         if 'chi2' in elem: data.derived_lkl[elem] = self.get_chi2(bval, self.bg)
            #         else: data.derived_lkl[elem] = self.bg[i]
        else: self.chi2 = self.get_chi2(bval)

        prior = 0.
        # if self.c["with_bbn"]: prior += -0.5 * ((data.cosmo_arguments['omega_b'] - self.c["omega_b_BBNcenter"]) / self.c["omega_b_BBNsigma"])**2
        # if self.c["with_n_s_prior"]: prior += -0.5 * ((data.cosmo_arguments['n_s'] - self.c["n_s_center"]) / self.c["n_s_sigma"])**2
        if self.prior: # Prior for Gaussian parameters
            if self.marg_chi2: bval = np.concatenate([bval, self.bg])
            if not self.marg or self.marg_chi2: prior += self.__set_prior_gauss_nonmarg(bval)
            prior += self.__set_prior_nongauss(bval) # Prior for non-Gaussian parameters: b1, c2, mag, baryons, photoz, nnlo
        return -0.5 * self.chi2 + prior

    def __prior(self, bs, prior_mean, prior_sigma, prior_inv_corr_mat, prior_type='gauss'):
        if 'gauss' in prior_type: 
            prior = - 0.5 * prior_sigma**-2 * np.einsum( 'n...,nm,m...->...', bs - prior_mean, prior_inv_corr_mat, bs - prior_mean )
        elif 'lognormal'in prior_type:
            if any(b <= 0. for b in bs): prior = - 0.5 * np.inf
            else:
                prior = - 0.5 * prior_sigma**-2 * np.einsum( 'n...,nm,m...->...', np.log(bs) - prior_mean, prior_inv_corr_mat, np.log(bs) - prior_mean ) 
                if self.extended: prior -= 0.5*np.sum(np.log(bs), axis=0)
                else: prior -= np.sum(np.log(bs), axis=0)
        return prior

    def __set_prior_gauss_nonmarg(self, bval): # Gaussian prior for Gaussian parameters when they are not not marginalized
        pad = 2*self.e.Ng # (b1, c2)
        if self.extended: pad += 2*self.e.Ng # independent set of (b1, c2) for gs 
        if self.mag: pad += self.e.Ng
        if self.baryons: pad += self.e.Ng # b1i
        if self.nnlo: pad += 4 # gg, gs, ss+, ss-
        if self.photoz: pad += self.e.Ns + 2 * self.e.Ng
        if self.intrinsic_alignments: pad += 2 # A, alpha
        if self.shear_calibration: pad += self.e.Ns
        if self.point_mass: pad += self.e.Ng
        bg = bval[pad:pad+self.e.Nmarg]
        prior = self.__prior(bg, self.bg_prior_mean, self.prior_sigma, self.prior_gauss_inv_corr_mat)
        return prior 

    def __set_prior_nongauss(self, bval, log_b1_mean=0.8, log_b1_sigma=0.8944, b1type='lognormal'): 
        '''
        Prior for non-Gaussian parameters
            Log-normal prior on b1 (linear biases)
            Gaussian prior on c2 (quadratic biases)
            Gaussian prior on bmag (magnification biases)
            Gaussian prior for bia (adia x iso biases in presence of baryons)
            Gaussian prior on dz (photo-z error biases)
            Gaussian prior on nnlo (nnlo biases)
        '''
        id_0 = 0; prior = 0.
        if self.nongaussian_prior: # (b1, c2)
            pad = self.e.Ng; b1 = bval[id_0:id_0+pad]; id_0 += pad
            pad = self.e.Ng; c2 = bval[id_0:id_0+pad]; id_0 += pad
            if not self.extended: 
                prior += self.__prior(b1, log_b1_mean, log_b1_sigma, self.prior_nongauss_inv_corr_mat, prior_type='lognormal')
                prior += self.__prior(c2, self.c2_prior_mean, self.prior_sigma, self.prior_nongauss_inv_corr_mat)
            else: # independent set of (b1, c2) for gs 
                pad = self.e.Ng; b1_gs = bval[id_0:id_0+pad]; id_0 += pad
                pad = self.e.Ng; c2_gs = bval[id_0:id_0+pad]; id_0 += pad
                prior += self.__prior(np.concatenate((b1, b1_gs)), log_b1_mean, log_b1_sigma, self.prior_nongauss_inv_crosscorr_mat, prior_type='lognormal')
                prior += self.__prior(np.concatenate((c2, c2_gs)), self.c2_prior_mean, self.prior_sigma, self.prior_nongauss_inv_crosscorr_mat)
        if self.mag: 
            pad = self.e.Ng; bmag = bval[id_0:id_0+pad]; id_0 += pad
            prior += self.__prior(bmag, self.prior_mean_bmag, self.prior_sigma, self.prior_nongauss_inv_corr_mat) 
        if self.baryons: 
            pad = self.e.Ng; bia = bval[id_0:id_0+pad]; id_0 += pad
            prior += self.__prior(bia, 0., self.prior_sigma, self.prior_nongauss_inv_corr_mat)
        if self.nnlo: 
            pad = 4; bnnlo = bval[id_0:id_0+pad]; id_0 += pad # gg, gs, ss+, ss-
            prior += self.__prior(bnnlo, 0., 1., self.prior_nnlo_inv_corr_mat)
        if self.photoz: 
            pad = self.e.Ns; dzs = bval[id_0:id_0+pad]; id_0 += pad
            prior += - 0.5 * np.sum((dzs - self.photoz_source_dz_mean)**2 / self.photoz_source_dz_sigma**2)
            pad = self.e.Ng; dzl = bval[id_0:id_0+pad]; id_0 += pad
            prior += - 0.5 * np.sum((dzl - self.photoz_lens_dz_mean)**2 / self.photoz_lens_dz_sigma**2)
            pad = self.e.Ng; szl = bval[id_0:id_0+pad]; id_0 += pad
            prior += - 0.5 * np.sum((szl - self.photoz_lens_sz_mean)**2 / self.photoz_lens_sz_sigma**2)
        if self.intrinsic_alignments: 
            pad = 2; id_0 += pad # A, alpha
        if self.shear_calibration:
            pad = self.e.Ns; ms = bval[id_0:id_0+pad]; id_0 += pad
            prior += - 0.5 * np.sum((ms - self.shear_calibration_mean)**2 / self.shear_calibration_sigma**2)
        if self.point_mass:
            pad = self.e.Ng; pm = bval[id_0:id_0+pad]; id_0 += pad
            prior += - 0.5 * np.sum(pm**2/self.sigma_pm**2) 
        return prior

    def create_list(self, correlator):
        if len(correlator) != 4:
            correlator = [correlator[self.indexdict['2pt_xip_startind']:self.indexdict['2pt_xip_endind']+1],
                          correlator[self.indexdict['2pt_xim_startind']:self.indexdict['2pt_xim_endind']+1],
                          correlator[self.indexdict['2pt_gammat_startind']:self.indexdict['2pt_gammat_endind']+1],
                          correlator[self.indexdict['2pt_wtheta_startind']:self.indexdict['2pt_wtheta_endind']+1],
                          ]
        
        xssp = [ [correlator[0][s] for s, ij in enumerate(self.indexbin[0]) if ij == binpair] for binpair in self.binpairs[0] ]
        xssm = [ [correlator[1][s] for s, ij in enumerate(self.indexbin[1]) if ij == binpair] for binpair in self.binpairs[1] ]
        xgs = [ [correlator[2][s] for s, ij in enumerate(self.indexbin[2]) if ij == binpair] for binpair in self.binpairs[2] ]
        xgg = [ [correlator[3][s] for s, ij in enumerate(self.indexbin[3]) if ij == binpair] for binpair in self.binpairs[3] ]
        return [xssp, xssm, xgs, xgg]

    def get_bestfit(self, bval):
        # _, bg = self.get_chi2_marg(bval) # Gaussian parameters
        # correlator = self.get_correlator(bval, bg)
        correlator = self.get_correlator(bval)
        return self.create_list(correlator)

    def print_bestfit(self, bval):
        if self.marg:
            self.e.print_bias(bval, self.bg, marg=True)
        else:
            self.e.print_bias(bval, None, marg=False)
    
    def set_cosmo(self, cosmo_dict=None, module='class', engine=None): 
        
        if not engine:
            from classy import Class
            M = Class()
            M.set(cosmo_dict)
            M.set(self.class_settings)
            M.compute()
        else: M = engine
        
        h = M.h()
        Omega0_m = M.Omega0_m()

        self.pk = np.array([M.pk_lin(ki, self.zfid) for ki in self.kk]) # linear P(k) in (Mpc)**3
        def deriv(x, func, dx=0.001): return 0.5*(func(x+dx)-func(x-dx))/dx
        def comoving_distance(z): return M.angular_distance(z)*(1+z)
        rz = np.array([comoving_distance(z) for z in self.zz])
        dr_by_dz = np.array([deriv(z, comoving_distance) for z in self.zz]) # PZ: is dr_by_dz ~ H(z)/c?
        Dz = np.array([M.scale_independent_growth_factor(zi) for zi in self.zz])
        Dfid = M.scale_independent_growth_factor(self.zfid)
        if self.rsd:
            ffid = 1. # debugged
            fz = np.array([M.scale_independent_growth_factor_f(zi) for zi in self.zz])
            # ffid = M.scale_independent_growth_factor_f(self.zfid)
            if "buzzard" in self.c["data_file"]: # 2105.13547, Appendix H
                # print ('Buzzard: rescaling redshift by 1/(1+z)')
                fz /= np.array([(1.+zi) for zi in self.zz])
                # ffid /= (1.+self.zfid) 
        else: fz, ffid = None, None
        
        if self.baryons: 
            Tk = M.get_transfer(z=self.zfid) # transfer functions
            Ta = interp1d(Tk['k (h/Mpc)'], ( M.Omega0_cdm()*Tk['d_cdm'] + M.Omega_b()*Tk['d_b'] ) / M.Omega0_m(), kind='cubic')(self.kk / M.h()) # adiabatic
            Ti = interp1d(Tk['k (h/Mpc)'], Tk['d_cdm'] - Tk['d_b'], kind='cubic')(self.kk / M.h()) # isocurvature
            kpivot = 0.05
            A_s = M.get_current_derived_parameters(['A_s'])['A_s']
            pk_primordial = 2*np.pi**2 / (self.kk)**3 * A_s * (self.kk/0.05)**(M.n_s()-1)
            pai = Ta * Ti * pk_primordial
        else: pai = None 

        if self.halofit: 
            self.zz_thick = np.linspace(self.zz[0], self.zz[-1], 100) 
            self.rz_thick = np.array([comoving_distance(z) for z in self.zz_thick]) 
            pk_nl = np.array([[M.pk(ki, zi) for zi in self.zz_thick] for ki in self.kk]) # halofit P(k,z) in (Mpc)**3
            ilogpk_nl = RegularGridInterpolator((np.log(self.kk), self.rz_thick), np.log(pk_nl), method='linear', bounds_error=False, fill_value=None)
        else: 
            ilogpk_nl = None
        
        return self.pk, ilogpk_nl, rz, 1/dr_by_dz, Dz, Dfid, fz, ffid, h, Omega0_m, pai

