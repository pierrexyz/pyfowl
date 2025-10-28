import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from scipy.special import legendre
from scipy.integrate import quad, dblquad, simps
import scipy.constants as conts

from pyfowl.fftlog import FFTLog, MJ, MJ0, MPC
from pyfowl.nonlinear import M13a, M22a
from pyfowl.fftlog_fang import fftlog_fang

class Correlator(object):
    
    def __init__(self, theta, z, nlens, nsource, limber=False, limber_2=False, 
        reduced=True, load=True, save=True, path='./', NFFT=256, Nsum=400, knl=1., km=1., 
        nnlo=False, baryons=False, rsd=False, mag=False):
        
        self.limber = limber
        self.limber_2 = limber_2
        self.baryons = baryons
        self.rsd = rsd
        self.mag = mag
        self.nnlo = nnlo
        self.knl = knl
        self.km = km

        self.z = z
        self.Nsum = Nsum
        self.z_thin_arr = np.linspace(z[0], z[-1], Nsum) # linspace is better than logspace # np.logspace(np.log10(z[0]), np.log10(z[-1]), num=self.Nsum, endpoint=True) 
        self.nlens = self._interp1d(np.asarray(nlens))
        self.nsource = self._interp1d(np.asarray(nsource))

        self.theta, _ = np.meshgrid(theta, self.z_thin_arr, indexing='ij')
        self.Nz = len(self.z_thin_arr)
        self.Nt = len(theta)
        
        self.Ng = self.nlens.shape[0]
        self.Ns = self.nsource.shape[0]
        self.Nss = self.Ns*(self.Ns+1)//2
        self.Ngs = self.Ns*self.Ng
        self.Ngg = self.Ng
        self.N = max([self.Nss, self.Ngs])
        
        self.Nbin = 2*self.Nss + self.Ngs + self.Ngg
        
        self.reduced = reduced
        if self.reduced: self.Nmarg = 2+3*self.Ng # number of gaussian parameters: 2 [ss counterterms] + 2*Ng [gs + gg counterterms] + Ng [b3]
        else: self.Nmarg = self.Nbin+self.Ng # Nbin [counterterms] + Ng [b3]
        
        self.kdeep = np.logspace(-4, 5, 400)
        self.fft0settings = dict(Nmax=NFFT, xmin=1.5e-4, xmax=1.e4, bias=-1.01)
        self.fft0 = FFTLog(**self.fft0settings)

        self.fft1settings = dict(Nmax=NFFT, xmin=1.5e-4, xmax=1.e4, bias=-2.01)
        self.fft1 = FFTLog(**self.fft1settings)

        self.fft2settings = dict(Nmax=NFFT, xmin=1.5e-4, xmax=1.e5, bias=-2.6)
        self.fft2 = FFTLog(**self.fft2settings)

        self.pyegg = os.path.join(path, 'pyegg%s_limber.npz') % (NFFT)

        if load is True:
            try:
                L = np.load( self.pyegg )
                if (self.fft0.Pow - L['Pow0']).any() or (self.fft1.Pow - L['Pow1']).any() or (self.fft2.Pow - L['Pow2']).any():
                    print ('Loaded loop matrices do not correspond to asked FFTLog configuration. \n Computing new matrices.')
                    load = False
                else:
                    self.M0, self.M22, self.M13, self.W2, self.W22, self.W13, self.Mrsd, self.Mmag = L['M0'], L['M22'], L['M13'], L['W2'], L['W22'], L['W13'], L['Mrsd'], L['Mrsd']
                    save = False
            except:
                print ('Can\'t load loop matrices at %s. \n Computing new matrices.' % path)
                load = False

        if load is False:
            self.setM0()
            self.setM1()
            self.setM13()
            self.setM22()
            self.setW2()
            self.setW13()
            self.setW22()
            self.setMrsd()
            self.setMmag()

        if save is True:
            try: np.savez(self.pyegg, Pow0=self.fft0.Pow, Pow1=self.fft1.Pow, Pow2=self.fft2.Pow, 
                M0=self.M0, M22=self.M22, M13=self.M13, W2=self.W2, W22=self.W22, W13=self.W13)
            except: print ('Can\'t save loop matrices at %s.' % path)

        self.setsPow()

        # To speed-up matrix multiplication:
        self.optipath13 = np.einsum_path('ns,ms,bnm->bs', self.sPow, self.sPow, self.M22, optimize='optimal')[0]
        self.optipath22 = np.einsum_path('ns,ms,bnm->bs', self.sPow, self.sPow, self.M13, optimize='optimal')[0]

        # Beyond Limber for linear part
        # self.ls =  np.concatenate([[0.1], np.arange(1, 500, 1), np.geomspace(500, 1e4, 20)])
        self.ls = np.concatenate([[0.1], np.arange(1, 5, 1), np.geomspace(5, 1e4, 50)]) # fast
        self.t = theta
        self.fftsettings = dict(Nmax=256, xmin=1e-1, xmax=1e4, bias=-1.01) ### ell-to-theta FFTLog
        self.fft = FFTLog(**self.fftsettings)
        self.Mb = np.empty(shape=(3, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(3): self.Mb[l] = MJ(2 * l, -0.5 * self.fft.Pow)
        self.tPow = np.exp(np.einsum('n,s->ns', -self.fft.Pow - 2., np.log(self.t)))

    def _interp1d(self, fz):
        return interp1d(self.z, fz, axis=-1, kind='cubic', fill_value='extrapolate')(self.z_thin_arr) # extrapolate to prevent rounding errors

    def _get_cls_limber(self, t1, t2, ls, d1=0, d2=0):
        """
        Attributes
        ----------
        d1, d2 : 0, 1, 2
            placeholders of transfer function t1 / t2 standing for 0: density, 1: magification, or 2: redshift-space distortion
        """
        ell, chi = np.meshgrid(ls, self.chi_logspace_arr, indexing='ij')
        k = (ell+0.5) / chi
        def get_transfer(tfunc, ell, d=0):
            if d == 0: return tfunc # here no need to interpolate on the mesh
            elif d == 1: return ell*(ell+1)/k**2 * tfunc # mag
            elif d == 2: # rsd: we interpolate on the mesh as we need the transfer function evaluated at two places, (ell+0.5)/k and (ell+0.5+1)/k .
                itfunc = interp1d(self.chi_logspace_arr, - tfunc, kind='cubic', bounds_error=False, fill_value=0.) # 1. add minus sign given chosen convention in definition of tgal_rsd | 2. fill with 0 as the transfer function goes to 0 at high chi, i.e. at high z
                return (8*ell+1.)/(2*ell+1.)**2 * itfunc((ell+0.5)/k) - 4.*(2*ell+1.)**.5/(2*ell+3.)**1.5 * itfunc((ell+1.5)/k)
        cls_limber = np.trapz(get_transfer(t1, ell, d=d1) * get_transfer(t2, ell, d=d2) / self.chi_logspace_arr**2 * np.exp(self.pklin_interp(np.log(k))), x=self.chi_logspace_arr, axis=-1)
        return cls_limber
    
    def _get_cls_nonlimber(self, fchi1, fchi2, ls, d1=0, d2=0):
        nu = 1.01
        myfftlog1 = fftlog_fang(self.chi_logspace_arr, fchi1 * self.chi_logspace_arr, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=112)
        myfftlog2 = fftlog_fang(self.chi_logspace_arr, fchi2 * self.chi_logspace_arr, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=112)
        Nell = ls.size
        cls_nonlimber = np.empty(Nell)
        def get_fftlog(myfftlog, ell, d=0):
            if d == 0: k, fk = myfftlog.fftlog(ell)
            elif d == 1: 
                k, fk = myfftlog.fftlog(ell)
                fk *= ell*(ell+1)/k**2
            elif d == 2: k, fk = myfftlog.fftlog_ddj(ell)
            return k, fk
        for i in np.arange(Nell):
            ell = ls[i]
            k, fk1 = get_fftlog(myfftlog1, ell, d=d1)
            k, fk2 = get_fftlog(myfftlog2, ell, d=d2)
            cls_nonlimber[i] = np.sum(fk1 * fk2 * k**3 * np.exp(self.pklin_interp(np.log(k))) ) * self.dlnr * 2./np.pi
        return cls_nonlimber
    
    def _get_cls(self, t1, t2, d1=0, d2=0, l_split=400):
        if self.limber_2: # for debugging
            cls_limber = self._get_cls_limber(t1, t2, self.ls, d1=d1, d2=d2) 
            return cls_limber
        else:
            cls_nonlimber = self._get_cls_nonlimber(t1, t2, self.ls[self.ls<l_split], d1=d1, d2=d2)
            cls_limber = self._get_cls_limber(t1, t2, self.ls[self.ls>=l_split], d1=d1, d2=d2)
            return np.hstack((cls_nonlimber, cls_limber)) 
            
    
    def _get_xi(self, cl, cl_type='gg'):
        Coef = self.fft.Coef(self.ls, cl, extrap='extrap', window=None)
        CoeftPow = np.einsum('n,nt->nt', Coef, self.tPow)
        if 'gg' in cl_type: M = self.Mb[0]
        elif 'gs' in cl_type: M = self.Mb[1]
        elif 'ssp' in cl_type: M = self.Mb[0]
        elif 'ssm' in cl_type: M = self.Mb[2]
        return np.real(np.einsum('nt,n->t', CoeftPow, M))

    def _get_xi_lin_nonlimber(self):            
        self.xi_gg_lin_nonlimber = np.array([self._get_xi(self._get_cls(t1, t1), cl_type='gg') for t1 in self.tgal])
        self.xi_gs_lin_nonlimber = np.array([self._get_xi(self._get_cls(t1, t2), cl_type='gs') for t1 in self.tgal for t2 in self.tshear])
        # self.xi_ssp_lin_nonlimber = np.array([self._get_xi(self._get_cls(t1, t2), cl_type='ssp') for i, t1 in enumerate(self.tshear) for j, t2 in enumerate(self.tshear) if i <= j])
        # self.xi_ssm_lin_nonlimber = np.array([self._get_xi(self._get_cls(t1, t2), cl_type='ssm') for i, t1 in enumerate(self.tshear) for j, t2 in enumerate(self.tshear) if i <= j])
        if self.rsd: 
            self.xi_gg_lin_rsd_nonlimber = np.array([
                [self._get_xi(self._get_cls(t1, t2, d1=0, d2=2), cl_type='gg') for t1, t2 in zip(self.tgal, self.tgal_rsd)], 
                [self._get_xi(self._get_cls(t1, t1, d1=2, d2=2), cl_type='gg') for t1 in self.tgal_rsd], 
                    ])
        if self.mag:
            self.xi_gg_lin_mag_nonlimber = np.array([
                [self._get_xi(self._get_cls(t1, t2, d1=0, d2=1), cl_type='gg') for t1, t2 in zip(self.tgal, self.tgal_mag)], 
                [self._get_xi(self._get_cls(t1, t1, d1=1, d2=1), cl_type='gg') for t1 in self.tgal_mag], 
                    ])
            self.xi_gs_lin_mag_nonlimber = np.array([self._get_xi(self._get_cls(t1, t2, d1=1, d2=0), cl_type='gs') for t1 in self.tgal_mag for t2 in self.tshear])

    def setsPow(self):
        """ Compute the r's to the powers of the FFTLog to evaluate the loop correlation function. Called at the instantiation of the class. """
        self.s = np.geomspace(1.e-3, 1.e4, 256) # DES s_min = min ( chi x theta ) ~ 0.07
        self.sPow = exp(np.einsum('n,s->ns', -self.fft0.Pow - 2., log(self.s)))   # linear
        self.sPow1 = exp(np.einsum('n,s->ns', -self.fft1.Pow - 2.5, log(self.s))) # 1loop 
        self.sPow2 = exp(np.einsum('n,s->ns', -self.fft2.Pow - 3.5, log(self.s))) # k^2 1loop
        self.sPow_rsd = np.array([exp(np.einsum('n,s->ns', -self.fft0.Pow - p - 1., log(self.s))) for p in [1, -1, -3]]) # rsd
        self.sPow_mag = np.array([exp(np.einsum('n,s->ns', -self.fft0.Pow - p - 1., log(self.s))) for p in [2, 0, 3, 1, -1]]) # mag

    def setM0(self):
        """ Compute the linear matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M0 = np.empty(shape=(3, self.fft0.Pow.shape[0]), dtype='complex')
        for l in range(3): self.M0[l] = MJ(2 * l, -0.5 * self.fft0.Pow)

    def setM1(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M1 = np.empty(shape=(3, self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex')
        for l in range(3):
            for u, n1 in enumerate(-0.5 * self.fft1.Pow):
                for v, n2 in enumerate(-0.5 * self.fft1.Pow):
                    self.M1[l, u, v] = MJ(2 * l, n1 + n2 - 1.5)

    def setM22(self):
        """ Compute the 22-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mbb22 = np.empty(shape=(6, self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex')
        self.Mbm22 = np.empty(shape=(3, self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex')
        self.Mmm22 = np.empty(shape=(2, self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex')
        Ma = np.empty(shape=(self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex') # common piece of M22
        Mmm = np.empty(shape=(self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex') # matter-matter M22
        for u, n1 in enumerate(-0.5 * self.fft1.Pow):
            for v, n2 in enumerate(-0.5 * self.fft1.Pow):
                Ma[u, v] = M22a(n1, n2)
                Mmm[u, v] = M22mm[0](n1, n2)
        for i in range(6):
            Mbb = np.empty(shape=(self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex')
            Mbm = np.empty(shape=(self.fft1.Pow.shape[0], self.fft1.Pow.shape[0]), dtype='complex')
            for u, n1 in enumerate(-0.5 * self.fft1.Pow):
                for v, n2 in enumerate(-0.5 * self.fft1.Pow):
                    Mbb[u, v] = M22bb[i](n1, n2)
                    if i < 3: Mbm[u, v] = M22bm[i](n1, n2)
            self.Mbb22[i] = Mbb
            if i < 3: self.Mbm22[i] = Mbm
        self.Mbb22 = np.einsum('nm,nm,bnm->bnm', self.M1[0], Ma, self.Mbb22)
        self.Mbm22 = np.einsum('nm,nm,bnm->bnm', self.M1[1], Ma, self.Mbm22)
        self.Mmm22 = np.einsum('lnm,nm,nm->lnm', self.M1[[0,2]], Ma, Mmm)
        self.M22 = np.vstack([self.Mmm22, self.Mbm22, self.Mbb22])

    def setM13(self):
        """ Compute the 13-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        Ma = M13a(-0.5 * self.fft1.Pow)
        Mmm = M13mm[0](-0.5 * self.fft1.Pow)
        Mbm = np.empty(shape=(2, self.fft1.Pow.shape[0]), dtype='complex')
        Mbb = np.empty(shape=(2, self.fft1.Pow.shape[0]), dtype='complex')
        for i in range(2): 
            Mbb[i] = M13bb[i](-0.5 * self.fft1.Pow)
            Mbm[i] = M13bm[i](-0.5 * self.fft1.Pow)
        self.Mbb13 = np.einsum('nm,n,bn->bnm', self.M1[0], Ma, Mbb)
        self.Mbm13 = np.einsum('nm,n,bn->bnm', self.M1[1], Ma, Mbm)
        self.Mmm13 = np.einsum('lnm,n,n->lnm', self.M1[[0,2]], Ma, Mmm)
        self.M13 = np.vstack([self.Mmm13, self.Mbm13, self.Mbb13])

    def setW2(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.W2 = np.empty(shape=(3, self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex')
        for l in range(3):
            for u, n1 in enumerate(-0.5 * self.fft2.Pow):
                for v, n2 in enumerate(-0.5 * self.fft2.Pow):
                    self.W2[l, u, v] = MJ(2 * l, n1 + n2 - 2.5)

    def setW22(self):
        """ Compute the 22-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Wbb22 = np.empty(shape=(6, self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex')
        self.Wbm22 = np.empty(shape=(3, self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex')
        self.Wmm22 = np.empty(shape=(2, self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex')
        Wa = np.empty(shape=(self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex') # common piece of W22
        Wmm = np.empty(shape=(self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex') # matter-matter W22
        for u, n1 in enumerate(-0.5 * self.fft2.Pow):
            for v, n2 in enumerate(-0.5 * self.fft2.Pow):
                Wa[u, v] = M22a(n1, n2)
                Wmm[u, v] = M22mm[0](n1, n2)
        for i in range(6):
            Wbb = np.empty(shape=(self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex')
            Wbm = np.empty(shape=(self.fft2.Pow.shape[0], self.fft2.Pow.shape[0]), dtype='complex')
            for u, n1 in enumerate(-0.5 * self.fft2.Pow):
                for v, n2 in enumerate(-0.5 * self.fft2.Pow):
                    Wbb[u, v] = M22bb[i](n1, n2)
                    if i < 3: Wbm[u, v] = M22bm[i](n1, n2)
            self.Wbb22[i] = Wbb
            if i < 3: self.Wbm22[i] = Wbm
        self.Wbb22 = np.einsum('nm,nm,bnm->bnm', self.W2[0], Wa, self.Wbb22)
        self.Wbm22 = np.einsum('nm,nm,bnm->bnm', self.W2[1], Wa, self.Wbm22)
        self.Wmm22 = np.einsum('lnm,nm,nm->lnm', self.W2[[0,2]], Wa, Wmm)
        self.W22 = np.vstack([self.Wmm22, self.Wbm22, self.Wbb22])

    def setW13(self):
        """ Compute the 13-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        Wa = M13a(-0.5 * self.fft2.Pow)
        Wmm = M13mm[0](-0.5 * self.fft2.Pow)
        Wbm = np.empty(shape=(2, self.fft2.Pow.shape[0]), dtype='complex')
        Wbb = np.empty(shape=(2, self.fft2.Pow.shape[0]), dtype='complex')
        for i in range(2): 
            Wbb[i] = M13bb[i](-0.5 * self.fft2.Pow)
            Wbm[i] = M13bm[i](-0.5 * self.fft2.Pow)
        self.Wbb13 = np.einsum('nm,n,bn->bnm', self.W2[0], Wa, Wbb)
        self.Wbm13 = np.einsum('nm,n,bn->bnm', self.W2[1], Wa, Wbm)
        self.Wmm13 = np.einsum('lnm,n,n->lnm', self.W2[[0,2]], Wa, Wmm)
        self.W13 = np.vstack([self.Wmm13, self.Wbm13, self.Wbb13])

    def setMrsd(self):
        """ Compute the linear redshift-space distortion matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mrsd = np.empty(shape=(3, self.fft0.Pow.shape[0]), dtype='complex')
        for i, p in enumerate([1, -1, -3]): self.Mrsd[i] = MJ0(p, -0.5 * self.fft0.Pow)

    def setMmag(self):
        """ Compute the linear magnification matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mmag = np.empty(shape=(5, self.fft0.Pow.shape[0]), dtype='complex')
        for i, p in enumerate([2, 0, 3, 1, -1]): self.Mmag[i] = MJ0(p, -0.5 * self.fft0.Pow) # (dm, dm), (mm, mm, mm)

    def _get_bessel_transforms11(self, CoefsPow):
        """ Perform the linear correlation function matrix multiplications """
        A11 = np.real(np.einsum('ns,ln->ls', CoefsPow, self.M0))
        return np.array([A11[0], A11[2], A11[1], A11[0]])

    def _get_bessel_transformsct(self, CoefsPow):
        """ Perform the counterterm correlation function matrix multiplications """
        Act = np.real(np.einsum('ns,ln->ls', CoefsPow, self.M0))
        return np.array([Act[0], Act[2], Act[1], Act[0]])

    def _get_bessel_transforms22(self, CoefsPow):
        """ Perform the 22-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.M22, optimize=self.optipath22))

    def _get_bessel_transforms13(self, CoefsPow):
        """ Perform the 13-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.M13, optimize=self.optipath13))

    def _get_bessel_transforms_nnloct(self, CoefsPow):
        """ Perform the nnlo counterterm correlation function matrix multiplications """
        Vct = np.real(np.einsum('ns,ln->ls', CoefsPow, self.M0))
        return np.array([Vct[0], Vct[2], Vct[1], Vct[0]])

    def _get_bessel_transforms_nnlo22(self, CoefsPow):
        """ Perform the 22-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.W22, optimize=self.optipath22))

    def _get_bessel_transforms_nnlo13(self, CoefsPow):
        """ Perform the 13-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.W13, optimize=self.optipath13))

    def get_bessel_transforms(self, kin, Pin, window=None):
        
        Pk_interp = interp1d(np.log(kin), np.log(Pin), fill_value='extrapolate')
        Pk = np.exp(Pk_interp(np.log(self.kdeep)))

        coef11 = self.fft0.Coef(self.kdeep, Pk, window=.2)
        coefsPow11 = np.einsum('n,ns->ns', coef11, self.sPow)
        coefct = self.fft0.Coef(self.kdeep, self.kdeep**2/(1.+self.kdeep**2) * Pk, window=.2)
        coefsPowct = np.einsum('n,ns->ns', coefct, self.sPow)
        coef1 = self.fft1.Coef(self.kdeep, Pk, window=.2)
        coefsPow1 = np.einsum('n,ns->ns', coef1, self.sPow1)

        A11 = self._get_bessel_transforms11(coefsPow11)
        Act = self._get_bessel_transformsct(coefsPowct)
        A22 = self._get_bessel_transforms22(coefsPow1)
        A13 = self._get_bessel_transforms13(coefsPow1)

        return A11, Act, A13, A22

    def get_bessel_transforms_rsd(self, kin, Pin, window=None):
        
        Pk_interp = interp1d(np.log(kin), np.log(Pin), fill_value='extrapolate')
        Pk = np.exp(Pk_interp(np.log(self.kdeep)))

        coef = self.fft0.Coef(self.kdeep, Pk, window=.2)
        coefsPow = np.einsum('n,pns->pns', coef, self.sPow_rsd)

        Add, Adr, Arr = np.real(np.einsum('pns,pn->ps', coefsPow, self.Mrsd))

        return Adr, Arr

    def get_bessel_transforms_mag(self, kin, Pin, window=None): 

        Pk_interp = interp1d(np.log(kin), np.log(Pin), fill_value='extrapolate')
        Pk = np.exp(Pk_interp(np.log(self.kdeep)))

        coef = self.fft0.Coef(self.kdeep, Pk, window=.2)
        coefsPow = np.einsum('n,pns->pns', coef, self.sPow_mag)

        Adm2, Adm0, Amm3, Amm1, Amm_1 = np.real(np.einsum('pns,pn->ps', coefsPow, self.Mmag)) 

        return Adm2, Adm0, Amm3, Amm1, Amm_1

    def get_bessel_transforms_baryons(self, kin, Pai, window=None):

        coef11b = self.fft0.Coef(kin, Pai, window=.2)
        coefsPow11b = np.einsum('n,ns->ns', coef11b, self.sPow)
        B11 = self._get_bessel_transforms11(coefsPow11b)

        return B11

    def get_bessel_transforms_nnlo(self, kin, Pin, window=None):
        
        Pk_interp = interp1d(np.log(kin), np.log(Pin), fill_value='extrapolate')
        Pk = np.exp(Pk_interp(np.log(self.kdeep)))

        coefnnlo = self.fft0.Coef(self.kdeep, self.kdeep**4/(1.+self.kdeep**4) * Pk, window=.2)
        CoefsPownnlo = np.einsum( 'n,ns->ns', coefnnlo, self.sPow )
        coef2 = self.fft2.Coef(self.kdeep, 1/(1.+self.kdeep) * Pk, window=.2)
        coefsPow2 = np.einsum('n,ns->ns', coef2, self.sPow2)

        Vct = self._get_bessel_transforms_nnloct(CoefsPownnlo)
        V22 = self._get_bessel_transforms_nnlo22(coefsPow2)
        V13 = self._get_bessel_transforms_nnlo13(coefsPow2)

        return Vct, V13, V22
        
    def compute(self, kin, Pin, rz, dz_by_dr, Dz, Dfid, fz, ffid, h, Omega0_m, A=0., alpha=1., dzl=None, dzs=None, Pai=None):
        
        self._set_time_kernels(kin, Pin, rz, dz_by_dr, Dz, Dfid, fz, ffid, h, Omega0_m, A=A, alpha=alpha, dzl=dzl, dzs=dzs)

        def time_integral(qq, DD, A): # line-of-sight integration 
            A1 = interp1d(self.s, A, kind='cubic', axis=-1)(self.theta * self.rz)
            # return np.trapz( np.einsum('biz,z,btz->bitz', qq, DD, A1), x=self.rz, axis=-1 ) 
            # b: [ss+, ss-, gs, gg] ; i: ss/gs/gg bins ; z: redshift bins ; t: theta bins
            return np.trapz( np.einsum('...iz,z,...tz->...itz', qq, DD, A1), x=self.rz, axis=-1 ) 
        
        neat_indent = True
        if neat_indent:
            A11, Act, A13, A22 = self.get_bessel_transforms(kin, Pin) 

            A11 = time_integral(self.qq11, self.Dp2, A11)
            Act = time_integral(self.qq11, self.Dp2, Act)
            A13 = time_integral(self.qq13, self.Dp4, A13)
            A22 = time_integral(self.qq22, self.Dp4, A22)

            self.Assp = np.array([A11[0], Act[0], A13[0], A22[0]])[:,:self.Nss]
            self.Assm = np.array([A11[1], Act[1], A13[1], A22[1]])[:,:self.Nss]
            self.Ags = np.array([A11[2], Act[2], A13[2], A13[3], A22[2], A22[3], A22[4]])[:,:self.Ngs]
            self.Agg = np.array([A11[3], Act[3], A13[4], A13[5], A22[5], A22[6], A22[7], A22[8], A22[9], A22[10]])[:,:self.Ngg]

        if self.rsd:
            Arsd = self.get_bessel_transforms_rsd(kin, Pin) 
            self.Agg_rsd = time_integral(self.qqrsd, self.Dp2, Arsd)

        if self.mag:
            Adm2, Adm0, Amm3, Amm1, Amm_1 = self.get_bessel_transforms_mag(kin, Pin) # PZ: to add the gs zeta(r) function
            Amm = time_integral(self.qmqm, self.Dp2 * self.rz**6, Amm3) -1/2. * time_integral(self.qmqm, self.Dp2 * self.rz**4, Amm1) + 1/16. * time_integral(self.qmqm, self.Dp2 * self.rz**2, Amm_1)
            Agm = time_integral(self.qgqm, self.Dp2 * self.rz**3, Adm2) -1/4. * time_integral(self.qgqm, self.Dp2 * self.rz**1, Adm0)
            self.Agg_mag = np.array([Agm, Amm])
            # self.Ags_mag = time_integral(self.qsqm, self.Dp2, Asm)  # PZ

            
        
        if not self.limber:
            self._get_xi_lin_nonlimber()
            self.Agg[0] = self.xi_gg_lin_nonlimber
            self.Ags[0] = self.xi_gs_lin_nonlimber
            if self.rsd: 
                self.Agg_rsd = self.xi_gg_lin_rsd_nonlimber
            if self.mag: 
                self.Agg_mag = self.xi_gg_lin_mag_nonlimber
                # self.Ags_mag = self.xi_gs_lin_mag_nonlimber

        if self.baryons: 
            B11 = self.get_bessel_transforms_baryons(kin, Pai)
            self.B11 = time_integral(self.qq11, self.Dp2, B11) # baryons

        if self.nnlo: 
            Vct, V13, V22 = self.get_bessel_transforms_nnlo(kin, Pin)
            
            Vct = time_integral(self.qq11, self.Dp2, Vct) # nnlo counterterm k^4 P11
            V13 = time_integral(self.qq13, self.Dp4, V13) # nnlo higher-derivative k^2 P1loop
            V22 = time_integral(self.qq22, self.Dp4, V22) # nnlo higher-derivative k^2 P1loop

            self.Vct = Vct
            
            self.Vssp = np.array([0.*A11[0], 0.*Vct[0], V13[0], V22[0]])[:,:self.Nss]
            self.Vssm = np.array([0.*A11[1], 0.*Vct[1], V13[1], V22[1]])[:,:self.Nss]
            self.Vgs = np.array([0.*A11[2], 0.*Vct[2], V13[2], V13[3], V22[2], V22[3], V22[4]])[:,:self.Ngs]
            self.Vgg = np.array([0.*A11[3], 0.*Vct[3], V13[4], V13[5], V22[5], V22[6], V22[7], V22[8], V22[9], V22[10]])[:,:self.Ngg]

    def _set_time_kernels(self, kin, Pin, rz, dz_by_dr, Dz, Dfid, fz, ffid, h, Omega0_m, A=0., alpha=1., dzl=None, dzs=None):

        ### line-of-sight functions
        self.rz = self._interp1d(rz)
        self.dz_by_dr = self._interp1d(dz_by_dr)
        self.Dz = self._interp1d(Dz)
        self.Dp2 = ( self.Dz / Dfid )**2
        self.Dp4 = self.Dp2**2
        
        if self.rsd: 
            self.fz = self._interp1d(fz)
            self.Dp2f = - self.Dp2 * self.fz / ffid   # minus sign
            self.Dp2fp2 = self.Dp2 * (self.fz / ffid)**2


        lensing_factor = 1.5/conts.c**2 * h**2 * 1e10 * Omega0_m
        self.r1, _ = np.meshgrid(self.rz, self.z_thin_arr, indexing='ij')
        
        def lensing_efficiency(nz):
            return lensing_factor * self.rz * (1+self.z_thin_arr) * np.trapz(np.heaviside(self.rz-self.r1, 0.) * nz * (self.rz-self.r1)/self.rz, x=self.z_thin_arr, axis=-1)
        
        def intrinsic_alignments(A=A, alpha=alpha, z0=0.62, C1_rho_crit=0.0134):
            return - A * ( (1+self.z_thin_arr)/(1+z0) )**alpha * C1_rho_crit * Omega0_m / self.Dz
        
        def interp(zz, nz): return interp1d(zz, nz, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)
        if dzs is None: nsource = self.nsource
        else: nsource = np.array([interp(self.z_thin_arr-dz, self.nsource[i])(self.z_thin_arr) for i, dz in enumerate(dzs)])
        if dzl is None: nlens = self.nlens
        else: nlens = np.array([interp(self.z_thin_arr-dz, self.nlens[i])(self.z_thin_arr) for i, dz in enumerate(dzl)])
        
        qshear = np.array([lensing_efficiency(ns) + intrinsic_alignments(A, alpha)*self.dz_by_dr*ns for ns in nsource])
        qgal = np.array([self.dz_by_dr * nl for nl in nlens])
        
        ### line-of-sight kernels
        qsqs = np.zeros(shape=(self.N, self.Nz)) # we pad with zeros such that all arrays, shear-shear, 
        qsqs[:self.Nss] = np.array([qi*qj for i, qi in enumerate(qshear) for j, qj in enumerate(qshear) if i <= j])
        
        qgqs = np.zeros(shape=(self.N, self.Nz)) # galaxy-shear, etc. have same size for the einsum in the time integral, see in compute(self, ...)
        qgqs[:self.Ngs] = np.array([qi*qj for qi in qgal for qj in qshear])
        
        qgqg = np.zeros(shape=(self.N, self.Nz)) # it makes the code much more concise.
        qgqg[:self.Ng] = np.array([qi**2 for qi in qgal])

        self.qq11 = np.array([qsqs, qsqs, qgqs, qgqg]) 
        self.qq13 = np.array([qsqs, qsqs, qgqs, qgqs, qgqg, qgqg])
        self.qq22 = np.array([qsqs, qsqs, qgqs, qgqs, qgqs, qgqg, qgqg, qgqg, qgqg, qgqg, qgqg])

        if self.rsd:
            qrsd = np.array([ self.dz_by_dr * Dfid/self.Dz * UnivariateSpline(self.rz, nl * self.Dz/Dfid * self.fz/ffid, k=3, s=400).derivative(n=2)(self.rz) for nl in nlens ]) # spline: k=3 (cubic), s=200 (smoothing factor)...
            qgqr = np.array([qi*qj for qi, qj in zip(qgal, qrsd)])
            qrqr = np.array([qi**2 for qi in qrsd])
            self.qqrsd = np.array([qgqr, qrqr])

        if self.mag:
            qmag = np.array([1/self.rz**2 * lensing_efficiency(nl) for nl in nlens]) # 1/r^2 given the definition of the lensing efficiency; here it is the convergence (which is dimensionless)
            self.qmqm = np.array([qi**2 for qi in qmag])
            self.qgqm = np.array([qi*qj for qi, qj in zip(qgal, qmag)])
            # self.qsqm = np.array([qi*qj for qi in qmag for qj in qshear])

        if not self.limber:
            self.pklin_interp = interp1d(np.log(kin), np.log(Pin), fill_value='extrapolate')
            self.chi_logspace_arr = np.logspace(np.log10(rz[0]), np.log10(rz[-1]), num=self.Nsum, endpoint=True) # logspace in chi is not logspaced in z!
            self.dlnr = np.log(rz[-1]/rz[0])/(self.Nsum-1.)
            self.tgal = interp1d(self.rz, qgal * self.Dz / Dfid, kind='cubic', fill_value='extrapolate')(self.chi_logspace_arr) # extrapolate to prevent rounding errors
            self.tshear = interp1d(self.rz, qshear * self.Dz / Dfid, kind='cubic', fill_value='extrapolate')(self.chi_logspace_arr) # extrapolate to prevent rounding errors
            
            if self.rsd: self.tgal_rsd = interp1d(self.rz, - qgal * self.Dz / Dfid * self.fz / ffid, kind='cubic', fill_value='extrapolate')(self.chi_logspace_arr) # minus sign
            if self.mag: self.tgal_mag = interp1d(self.rz, qmag * self.Dz / Dfid, kind='cubic', fill_value='extrapolate')(self.chi_logspace_arr) 

    def format_bias(self, bng, bg=None, marg=False):
        
        if self.reduced: 
            if bg is None: 
                if marg: bg = np.zeros(shape=(self.Nmarg))
                else:
                    # if self.baryons: bg = bng[3*self.Ngg:]
                    # elif self.nnlo: bg = bng[2*self.Ngg+4:] # starting after (b1, c2)_i, i=1, ..., Ng, and 4 NNLO biases, one for each xi: xi_+, xi_-, gamma_t, w
                    pad = 2*self.Ngg # starting after (b1, c2)_i, i=1, ..., Ng
                    if self.mag: pad += self.Ngg # # starting after (b1, c2, b1mag)_i, i=1, ..., Ng
                    if self.baryons: pad += self.Ngg  # starting after (b1a, c2, b1i)_i, i=1, ..., Ng
                    if self.nnlo: pad += 4
                    bg = bng[pad:] 
            bgg = np.vstack([[bng[i] for i in range(self.Ngg)], 
                             [bng[i] for i in np.arange(self.Ngg, 2*self.Ngg)], 
                             [bg[i] for i in np.arange(2+self.Ng+self.Ngg, 2+self.Ng+2*self.Ngg)], 
                             [bng[i] for i in np.arange(self.Ngg, 2*self.Ngg)]]) 
            cssp = bg[0] * np.ones(self.Nss)
            cssm = bg[1] * np.ones(self.Nss)
            cgs = np.array([bg[i+2] for i in range(self.Ng) for j in range(self.Ns) ]) 
            cgg = np.array([bg[i] for i in np.arange(2+self.Ng, 2+self.Ng+self.Ngg)]) 
        else: # not maintainted
            if bg is None: 
                if marg: bg = np.zeros(shape=(self.Nbin+self.Ngg))
                else: bg = bng[2*self.Ngg:] # starting after (b1, c2)_i, i=1, ..., Ngg
            bgg = np.vstack([[bng[i] for i in range(self.Ngg)], 
                             [bng[i] for i in np.arange(self.Ngg, 2*self.Ngg)], 
                             [bg[i] for i in np.arange(self.Nbin, self.Nbin+self.Ngg)], 
                             [bng[i] for i in np.arange(self.Ngg, 2*self.Ngg)]])
            cgg = np.array([bg[i] for i in np.arange(2*self.Nss+self.Ngs, self.Nbin)])
            cgs = np.array([bg[i] for i in np.arange(2*self.Nss, 2*self.Nss+self.Ngs)])
            cssp = np.array([bg[i] for i in range(self.Nss)])
            cssm = np.array([bg[i] for i in np.arange(self.Nss, 2*self.Nss)])

        pad = 0
        if self.mag: 
            bmag = bng[(2+pad)*self.Ngg:(3+pad)*self.Ngg] # magnification biases
            pad += 1
        else: bmag = np.zeros(shape=(self.Ngg)) # dummy

        if self.baryons: 
            b1i = bng[(2+pad)*self.Ngg:(2+pad)*self.Ngg] # isocurvature linear biases
            pad += 1
        else: b1i = np.zeros(shape=(self.Ngg)) # dummy

        if self.nnlo: bnnlo = bng[(2+pad)*self.Ngg:(2+pad)*self.Ngg+4] # nnlo biases
        else: bnnlo = np.zeros(shape=(4)) # dummy

        return bgg, cgg, cgs, cssp, cssm, bmag, b1i, bnnlo

    def print_bias(self, bval, bg=None, marg=False):
        bgg, cgg, cgs, cssp, cssm, bmag, b1i, bnnlo = self.format_bias(bval, bg, marg=marg)
        for i, (b, tag) in enumerate(zip(bgg, ['b1', 'b2', 'b3'])):
            with np.printoptions(precision=3, suppress=True): print (tag, b)
        with np.printoptions(precision=3, suppress=True): print ('cgg', cgg)
        with np.printoptions(precision=3, suppress=True): print ('cgs', cgs.reshape(self.Ng, self.Ns)[:,0])
        print ('cssp', '%.3f' % cssp[0], 'cssm', '%.3f' % cssm[0])
        if self.mag:
            with np.printoptions(precision=3, suppress=True): print ('bmag', bmag)
        if self.baryons: 
            with np.printoptions(precision=3, suppress=True): print ('b1i', b1i)
        if self.nnlo: 
            with np.printoptions(precision=3, suppress=True): print ('bnnlo', bnnlo)

    def get_bias_arrays(self, bgg, cgg, cgs, cssp, cssm):

        bbssp = np.ones(shape=(self.Nss, 4))
        bbssm = np.ones(shape=(self.Nss, 4))
        bbssp[:,1] = 2./self.knl**2 * cssp
        bbssm[:,1] = 2./self.knl**2 * cssm
        
        cgs = 2./self.km**2 * cgs.reshape(self.Ng, self.Ns)
        bbgs = np.array([[bs[0], ci, bs[0], bs[2], bs[0], bs[1], bs[3]] for bs, cs in zip(bgg.T, cgs) for ci in cs]) # b1*11, c*ct, b1*13, b3*13, b1*22, b2*22, b4*22

        b1, b2, b3, b4 = bgg
        bbgg = np.array([b1**2, 2.*b1*cgg/self.km**2, b1**2, b1*b3, b1**2, b1*b2, b1*b4, b2**2, b2*b4, b4**2]).T
        
        return bbssp, bbssm, bbgs, bbgg, b1

    def set_3x2pt(self, bng, bg=None, marg=False):

        bgg, cgg, cgs, cssp, cssm, bmag, b1i, bnnlo = self.format_bias(bng, bg=bg, marg=marg)
        bssp, bssm, bgs, bgg, b1 = self.get_bias_arrays( bgg, cgg, cgs, cssp, cssm )
        def xi(bias_arrays, xi_arrays): return np.einsum('ib,bit->it', bias_arrays, xi_arrays)
        self.Xssp, self.Xssm, self.Xgs, self.Xgg = xi(bssp, self.Assp), xi(bssm, self.Assm), xi(bgs, self.Ags), xi(bgg, self.Agg)

        if self.rsd: 
            self.Xgg += np.einsum('i,it->it', 2.*b1, self.Agg_rsd[0]) + self.Agg_rsd[1] # + 2 b1 f ... + f^2 ...

        if self.mag: # we neglect the correlations with the loops
            bgg_mag = np.array([2.*bmag*b1, bmag**2]).T
            self.Xgg += xi(bgg_mag, self.Agg_mag)
            # self.Xgs += np.einsum('i,it->it', 2.*bmag, self.Ags_mag) # PZ
        
        if self.baryons: # adding adiabatic x isocurvature linear correction. b1a : adiabatic linear bias ; b1i : isocurvature linear bias .
            self.Xssp += self.B11[0,:self.Nss]
            self.Xssm += self.B11[1,:self.Nss]
            b1gs = np.concatenate([ self.Ns*[b1[i] + b1i[i]] for i in range(self.Ng) ]) 
            self.Xgs += np.einsum('i,it->it', b1gs, self.B11[2,:self.Ngs]) 
            self.Xgg_ia = np.einsum('i,it->it', b1 * b1i, self.B11[3,:self.Ngg])
            self.Xgg += self.Xgg_ia

        if self.nnlo:
            self.Yssp, self.Yssm, self.Ygs, self.Ygg = xi(bssp, self.Vssp), xi(bssm, self.Vssm), xi(bgs, self.Vgs), xi(bgg, self.Vgg)   
            self.Xssp += bnnlo[0] / self.knl**2 * self.Yssp
            self.Xssm += bnnlo[1] / self.knl**2 * self.Yssm
            self.Xgs += bnnlo[2] / self.km**2 * self.Ygs
            self.Xgg += bnnlo[3] / self.km**2 * self.Ygg

    def get_marg(self, bng, external_gg_counterterm=None, external_gg_b3=None): 

        bgg, _, _, _, _, _, _, bnnlo = self.format_bias(bng, marg=True) # getting bg, and bnnlo, if any
        b1, _, _, _ = bgg # getting b1
        
        if self.reduced:
            marg = np.zeros(shape=(self.Nmarg, self.Nt*self.Nbin))
            
            # counterterms
            for i in range(self.Nss): 
                marg[0 , (i)*self.Nt : (i+1)*self.Nt] = self.Assp[1,i] * 2/self.knl**2 # ss+: one cssp for all ss+ bins
                marg[1 , (self.Nss+i)*self.Nt : (self.Nss+i+1)*self.Nt] = self.Assm[1,i] * 2/self.knl**2 # ss-: one cssm for all ss- bins
            for i in range(self.Ngs): # gs: one cgs per lens
                j = i // self.Ns
                marg[2+j , (2*self.Nss+i)*self.Nt : (2*self.Nss+i+1)*self.Nt] = self.Ags[1,i] * 2/self.km**2 
            for i in range(self.Ngg): # gg
                if external_gg_counterterm is None: marg[i+2+self.Ng , (2*self.Nss+self.Ngs+i)*self.Nt : (2*self.Nss+self.Ngs+i+1)*self.Nt] = self.Agg[1,i] * 2*b1[i]/self.km**2
                else: marg[2+self.Ng+i, (2*self.Nss+self.Ngs+i)*self.Nt : (2*self.Nss+self.Ngs+i+1)*self.Nt] = external_gg_counterterm[i]

            # b3 
            for i in range(self.Ng):
                u = 2 + 2*self.Ng + i # id_marg_b3
                for j in range(self.Ns): 
                    v = 2*self.Nss + i*self.Ns + j # id_bin_gs
                    marg[u, v*self.Nt:(v+1)*self.Nt] = self.Ags[3,i*self.Ns + j]
                    if self.nnlo: marg[u, v*self.Nt:(v+1)*self.Nt] += self.Vgs[3,i*self.Ns + j] * bnnlo[2] / self.km**2
                w = 2*self.Nss + self.Ngs + i # id_bin_gg
                if external_gg_b3 is None: 
                    marg[u, w*self.Nt:(w+1)*self.Nt] = b1[i] * self.Agg[3,i]
                    if self.nnlo: marg[u, w*self.Nt:(w+1)*self.Nt] += b1[i] * self.Vgg[3,i] * bnnlo[3] / self.km**2
                else: marg[u, w*self.Nt:(w+1)*self.Nt] = external_gg_b3[i]
            
        else:
            marg = np.zeros(shape=(self.Nbin+self.Ng, self.Nt*self.Nbin))
            
            # counterterms: one per bin pair (i, j)
            for i in range(self.Nss): marg[i, i*self.Nt:(i+1)*self.Nt] = self.Assp[1,i] * 2/self.knl**2 # ss+
            for i in np.arange(self.Nss, 2*self.Nss): marg[i, i*self.Nt:(i+1)*self.Nt] = self.Assm[1,i-self.Nss] * 2/self.knl**2 # ss-
            for i in np.arange(2*self.Nss, 2*self.Nss+self.Ngs): marg[i, i*self.Nt:(i+1)*self.Nt] = self.Ags[1,i-2*self.Nss] * 2/self.km**2 # gs
            for i in np.arange(2*self.Nss+self.Ngs, self.Nbin): # gg
                if external_gg_counterterm is None: marg[i, i*self.Nt:(i+1)*self.Nt] = self.Agg[1,i-(2*self.Nss+self.Ngs)] * 2*b1[i-(2*self.Nss+self.Ngs)]/self.km**2
                else: marg[i, i*self.Nt:(i+1)*self.Nt] = external_gg_counterterm[i-(2*self.Nss+self.Ngs)]

            # b3 
            for i in range(self.Ng):
                u = self.Nbin + i  # gs
                for j in range(self.Ns):
                    v = 2*self.Nss + i*self.Ns + j
                    marg[u, v*self.Nt:(v+1)*self.Nt] = self.Ags[3,i*self.Ns + j]
                w = 2*self.Nss+self.Ngs + i # gg
                if external_gg_b3 is None: marg[u, w*self.Nt:(w+1)*self.Nt] = b1[i]*self.Agg[3,i]
                else: marg[u, w*self.Nt:(w+1)*self.Nt] = external_gg_b3[i]

        return marg




M22bb = { # galaxy-galaxy
    0: lambda n1, n2: (6 + n1**4 * (4 - 24 * n2) - 7 * n2 + 8 * n1**5 * n2 - 13 * n2**2 + 4 * n2**3 + 4 * n2**4 + n1**2 * (-13 + 38 * n2 + 12 * n2**2 - 8 * n2**3) + 2 * n1**3 * (2 - 5 * n2 - 4 * n2**2 + 8 * n2**3) + n1 * (-7 - 6 * n2 + 38 * n2**2 - 10 * n2**3 - 24 * n2**4 + 8 * n2**5)) / (4. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (-18 + n1**2 * (1 - 11 * n2) - 12 * n2 + n2**2 + 10 * n2**3 + 2 * n1**3 * (5 + 7 * n2) + n1 * (-12 - 38 * n2 - 11 * n2**2 + 14 * n2**3)) / (7. * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: (-3 * n1 + 2 * n1**2 + n2 * (-3 + 2 * n2)) / (n1 * n2),
    3: lambda n1, n2: (-4 * (-24 + n2 + 10 * n2**2) + 2 * n1 * (-2 + 51 * n2 + 21 * n2**2) + n1**2 * (-40 + 42 * n2 + 98 * n2**2)) / (49. * n1 * (1 + n1) * n2 * (1 + n2)),
    4: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7. * n1 * n2),
    5: lambda n1, n2: 2.
} # b1**2, b1*b2, b1*b4, b2**2, b2*b4, b4**2

M13bb = { # galaxy-galaxy
    0: lambda n1: 1.125,
    1: lambda n1: -(1 / (1. + n1))
} # b1**2, b1*b3

M13bm = { # galaxy-matter
    0: lambda n1: (5 + 9*n1)/(8. + 8*n1),
    1: lambda n1: -(1/(2. + 2*n1))
} # b1, b3

M22bm = { # galaxy-matter
    0: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + n1*(-1 + (13 - 6*n1)*n1) - n2 + 2*n1*(-3 + 2*n1)*(-9 + n1*(3 + 7*n1))*n2 + (13 + 2*n1*(-27 + 14*(-1 + n1)*n1))*n2**2 + 2*(-3 + n1*(-15 + 14*n1))*n2**3 + 28*n1*n2**4))/(28.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    1: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-22 + 7*n1**2*(5 + 7*n2) + n2*(16 + 35*n2) + n1*(16 + 7*n2*(6 + 7*n2))))/(98.*n1*(1 + n1)*n2*(1 + n2)),
    2: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-4 + 7*n1 + 7*n2))/(14.*n1*n2)
} # b1, b2, b4

M22mm = { # matter-matter
    0: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(58 + 98*n1**3*n2 + (3 - 91*n2)*n2 + 7*n1**2*(-13 - 2*n2 + 28*n2**2) + n1*(3 + 2*n2*(-73 + 7*n2*(-1 + 7*n2)))))/(196.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2))
}

M13mm = { # matter-matter
    0: lambda n1: 1.125 - 1./(1. + n1)
}
