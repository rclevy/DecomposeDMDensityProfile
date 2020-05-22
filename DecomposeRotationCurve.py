def decompose_rotation_curve(gal_name,path_dir,suffix,Rrot,Vrot,eVrot,Rstar,Vstar,eVstar,Ratom,Vatom,Rmol,Vmol):
	
	r'''
	Decompose rotation curve and density profile into stellar and DM components.
	Based on method in Relatores et el. (2019b).
	Assumes a generalized NFW profile for the DM.
	Minimizes chi squared for likelihood function.
	Calculates inner slope of DM density profile (beta_*), based on Relatores et al. (2019b).

	Parameters
	----------
	gal_name : str
		name of galaxy to fit
	path_dir : string
		path to directory to save output plots, if sub-directories don't exist they will be created
	suffix : string
		suffic to append to plot names
	Rrot : array
		numpy array containing the radii for the input rotation curve, in kpc 
	Vrot : array
		numpy array containing the rotation curve to fit and decompose, in km/s
	eVrot : array
		numpy array containing the uncertainties on Vrot, in km/s
	Rstar : array
		numpy array containing the radii for the stellar rotation curve, in kpc 
	Vstar : array
		numpy array containing the stellar rotation curve, in km/s
	eVstar : array
		numpy array containing the uncertainties on Vstar, in km/s
	Ratom : array
		numpy array containing the radii for the atomic gas rotation curve, if NaN ignores this component, if NaN ignores this component, in kpc 
	Vatom : array
		numpy array containing the atomic gas rotation curve, if NaN ignores this component, if NaN ignores this component, in km/s
	Rmol : array
		numpy array containing the radii for the molecular gas rotation curve, if NaN ignores this component, in kpc 
	Vmol : array
		numpy array containing the molecular gas rotation curve, if NaN ignores this component, in km/s
	
		
	Returns
	-------
	rho_o : float
		fitted DM density normalization, in g/cm^3
	erho_o : array
		numpy array containing the upper and lower uncertainties on rho_o, in g/cm^3
	Rs : float
		fitted radial DM scale length, in kpc
	eRs : array
		numpy array containing the upper and lower uncertainties on Rs, in kpc
	beta : float
		fitted shape parameter for DM density profile
	ebeta : array
		numpy array containing the upper and lower uncertainties on beta
	beta_star : float
		inner slope of the DM density profile
	ebeta_star : array
		numpy array containing the upper and lower uncertainties on beta_star
	R : array
		numpy array containing the radii of the fitted models, in kpc
	Vtot : array
		numpy array containing the fitted model circular velocity curve, in km/s
	eVtot : array
		numpy array containing the uncertainty on Vtot, in km/s
	Vdm : array
		numpy array containting the DM velocity curve, in km/s
	eVdm : array
		numpy array containing the uncertainty on Vdm
	rho_dm : array
		numpy array containing the fitted DM density profile, in g/cm^3
	erho_dm : array
		numpy array containing the uncertainty on rho_dm, in g/cm^3

	
	Notes
	-----
	Required modules: astropy, corner, emcee, matplotlib, numpy, os, scipy, time
	Author: R. C. Levy (rlevy.astro@gmail.com)
	Last updated: 2020-01-06
	Change log:
		2019-07-30 : file created, RCL
		2019-07-31 : finished writing core of file, RCL
		2019-08-01 : added functionality to support atomic and molecular components and plot suffix, RCL
		2020-01-06 : added functionality to handle uncertainties on stellar rotation velocity

	Examples
	--------
	>>> from DecomposeRotationCurve import decompose_rotation_curve
	>>> rho_o, erho_o, Rs, eRs, beta, ebeta, R, Vtot, eVtot, Vdm, eVdm, rho_dm, erho_dm
		= decompose_rotation_curve(gal_name, path_dir, suffix, Rrot, Vrot, eVrot, Rstar, Vstar, eVstar, Ratom, Vatom, Rmol, Vmol)
	'''

	#import modules
	import numpy as np
	import matplotlib.pyplot as plt
	import emcee
	import corner
	from scipy.interpolate import interp1d
	import astropy.units as u
	import os
	import time
	plt.rcParams['font.family'] = 'serif'

	print(gal_name)

	#check if sub-directories exist for saving plots
	#if not, create the subdirectories
	if not os.path.exists(path_dir+'mcmc_chains'):
		os.makedirs(path_dir+'mcmc_chains')
	if not os.path.exists(path_dir+'mcmc_correlations'):
		os.makedirs(path_dir+'mcmc_correlations')
	if not os.path.exists(path_dir+'rotcurve_decomp'):
		os.makedirs(path_dir+'rotcurve_decomp')

	#interpolate onto the same radial gridding
	r = Rrot.copy()
	v_int = interp1d(Rrot,Vrot,bounds_error=False,fill_value='extrapolate')
	v = v_int(r)
	ev_int = interp1d(Rrot,eVrot,bounds_error=False,fill_value='extrapolate')
	ev = ev_int(r)
	vs_int = interp1d(Rstar,Vstar,bounds_error=False,fill_value='extrapolate')
	vs = vs_int(r)
	evs_int = interp1d(Rstar,eVstar,bounds_error=False,fill_value='extrapolate')
	evs = evs_int(r)
	if np.all(np.isnan(Ratom))==False:
		va_int = interp1d(Ratom,Vatom,bounds_error=False,fill_value='extrapolate')
		va = va_int(r)
	else:
		va = np.zeros(r.shape)
	if np.all(np.isnan(Rmol))==False:
		vm_int = interp1d(Rmol,Vmol,bounds_error=False,fill_value='extrapolate')
		vm = vm_int(r)
	else:
		vm = np.zeros(r.shape)

	#package up data to be passed into the functions
	data = np.array([r,v,ev,vs,evs, va,vm])

	#define function to make dm density profile and get velocity
	def calc_DM_density(r,po,rs,b):
		#get DM density profile from generalized NFW
		rho_dm = po/((r/rs)**b*(1+r/rs)**(3-b))
		return rho_dm #g/cm^3

	def calc_DM_velocity(r,rho_dm):
		#convert density profile units 
		rho_dm = (rho_dm*u.g*u.cm**-3).to(u.g/u.kpc**3)
		#convert to DM velocity
		G = (6.67E-8*u.cm**3/u.g/u.s**2).to(u.km**2*u.kpc/u.g/u.s**2) #km^2 kpc g^-1 s^-2
		V_dm = np.sqrt(4*np.pi*G.value/r*np.trapz(r**2*rho_dm.value,x=r))
		#account for constant of integration
		V_dm = np.nanmax(V_dm)-V_dm 
		return V_dm

	def measure_beta_star(po,rs,b,epou,epol,ersu,ersl,ebu,ebl):
		#measure inner DM density slope
		#as in Relatores et al. (2019, submitted) Eq 6
		p_08=calc_DM_density(0.8,po,rs,b)
		p_03=calc_DM_density(0.3,po,rs,b)
		b_star = -np.log(p_08/p_03)/np.log(0.8/0.3)
		#do a monte carlo to get uncertainties on beta_*
		b_star_mc = np.zeros(500)
		for i in range(500):
			po_mc = po+np.random.uniform(-epol,epou)
			rs_mc = rs+np.random.uniform(-ersl,ersu)
			b_mc = b+np.random.uniform(-ebl,ebu)
			p_08_mc = calc_DM_density(0.8,po_mc,rs_mc,b_mc)
			p_03_mc = calc_DM_density(0.3,po_mc,rs_mc,b_mc)
			b_star_mc[i] = -np.log(p_08_mc/p_03_mc)/np.log(0.8/0.3)
		eb_star = np.std(b_star_mc)
		return b_star,eb_star


	#use emcee to minimize the -log(likelihood)
	#set up the following functions to be used by emcee
	def log_likelihood(theta,data):
		#unpack parameters
		po,rs,b=theta
		#unpack data
		r=data[0]
		v=data[1]
		ev=data[2]
		vs=data[3]
		evs=data[4]
		va=data[5]
		vm=data[6]
		#get DM density profile
		rho_dm=calc_DM_density(r,po,rs,b)
		#get DM rotation curve
		V_dm=calc_DM_velocity(r,rho_dm)
		#make model velocity
		m = np.sqrt(vs**2+V_dm**2+va**2+vm**2)
		log_like = -0.5*np.nansum(((v-m)/np.sqrt(ev**2+evs**2))**2)
		return log_like

	def log_prior(theta):
		#set priors on parameters
		po,rs,b=theta
		#if po > 0, rs > 0, b > 0, return 0 as log(prior)
		#also restrict rs < 20 kpc and b < 2.0 to help with the fits
		if po > 0.0 and rs > 0.0 and rs < 20.0 and b > 0.0 and b < 2.0:
			return 0.0
		#else return -inf as log(prior)
		return -np.inf

	def log_probability(theta,data):
		lp=log_prior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp+log_likelihood(theta,data)

	nll = lambda *args: -log_likelihood(*args)

	#get intial parameter guesses
	po_init = (0.005*u.solMass/u.pc**3).to(u.g/u.cm**3).value
	rs_init = np.max(r)
	b_init = 0.5
	init = np.array([po_init,rs_init,b_init])

	#set up the emcee
	ndim = 3 #number of dimensions (parameters)
	nwalkers = 200 #number of walkers to use
	nsteps = 200 #number of steps to take

	#get the starting positions for the walkers
	#small offset (1E-4) from intial values
	params = init.copy()
	poff = 10**(np.round(np.log10(params))-2.)
	pos = params+poff*np.random.rand(nwalkers,ndim)
	sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=[data])
	print('Running the MCMC...')
	ts = time.time()
	sampler.run_mcmc(pos,nsteps);
	te = time.time()
	t_mcmc = te-ts
	print('\t...took %.1f s' %t_mcmc)

	#discard burn in
	nburn=50
	samples=sampler.chain[:,nburn:,:].reshape((-1,ndim))
	
	#get best fit parameters and "1-sigma" uncertainties
	po_best,rs_best,b_best=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
												   zip(*np.percentile(samples, [16,50,84], axis=0)))
	params_best= np.array([po_best,rs_best,b_best])
	
	#unpack parameters and uncertainties
	po = po_best[0]
	epou = po_best[1]
	epol = po_best[2]
	epo = np.array([epou,epol])
	rs = rs_best[0]
	ersu = rs_best[1]
	ersl = rs_best[2]
	ers = np.array([ersu,ersl])
	b = b_best[0]
	ebu = b_best[1]
	ebl = b_best[2]
	eb = np.array([ebu,ebl])

	#look at chains
	fig,axes=plt.subplots(ndim, figsize=(10,7),sharex=True)
	chains=sampler.chain
	labels=['$\\rho_o$ (g cm$^{-3}$)','r$_{\mathrm{s}}$ (kpc)','$\\beta$']
	for j in range(ndim):
		ax=axes[j]
		ax.plot(chains[:,:,j].T,'gray',alpha=0.3)
		ax.set_xlim(0,chains.shape[1])
		ax.set_ylabel(labels[j])
		ax.yaxis.set_label_coords(-0.1,0.5)
		ax.minorticks_on()
		ax.axvline(nburn,color='k',linestyle='--')
		ax.axhline(params_best[j,0],color='b')
		ax.axhline(params_best[j,0]+params_best[j,1],color='b',alpha=0.5)
		ax.axhline(params_best[j,0]-params_best[j,2],color='b',alpha=0.5)
	axes[-1].set_xlabel('Step Number')
	plt.savefig(path_dir+'mcmc_chains/'+gal_name+'_chains'+suffix+'.pdf')

	#make a corner plot to show parameter posteriors and correlations
	fig=corner.corner(samples,labels=labels)
	axes=np.array(fig.axes).reshape((ndim,ndim))
	for j in range(ndim):
		ax=axes[j,j]
		ax.axvline(params_best[j,0],color='b')
		ax.axvline(params_best[j,0]+params_best[j,1],color='b',alpha=0.5)
		ax.axvline(params_best[j,0]-params_best[j,2],color='b',alpha=0.5)
		ax.minorticks_on()
	for yi in range(ndim):
		for xi in range(yi):
			ax=axes[yi,xi]
			ax.axvline(params_best[xi,0],color='b')
			ax.axvline(params_best[xi,0]+params_best[xi,1],color='b',alpha=0.5)
			ax.axvline(params_best[xi,0]-params_best[xi,2],color='b',alpha=0.5)     
			ax.axhline(params_best[yi,0],color='b')
			ax.axhline(params_best[yi,0]+params_best[yi,1],color='b',alpha=0.5)
			ax.axhline(params_best[yi,0]-params_best[yi,2],color='b',alpha=0.5)
			ax.plot(params_best[xi,0],params_best[yi,0],'bs')
			ax.minorticks_on()
	plt.savefig(path_dir+'mcmc_correlations/'+gal_name+'_paramcorrs'+suffix+'.pdf')

	#get rotation curve components from best fit
	rho_dm=calc_DM_density(r,po,rs,b)
	V_dm=calc_DM_velocity(r,rho_dm)
	V_tot = np.sqrt(vs**2+V_dm**2)

	#do a Monte Carlo to get the uncertainties on the DM density and velocity 
	rho_dm_mc = np.zeros((500,len(r)))
	V_dm_mc = np.zeros((500,len(r)))
	for i in range(500):
		po_mc = po+np.random.uniform(-epol,epou)
		rs_mc = rs+np.random.uniform(-ersl,ersu)
		b_mc = b+np.random.uniform(-ebl,ebu)
		rho_dm_mc[i,:]=calc_DM_density(r,po_mc,rs_mc,b_mc)
		V_dm_mc[i,:]=calc_DM_velocity(r,rho_dm_mc[i,:])
	erho_dm = np.std(rho_dm_mc,axis=0)
	eV_dm = np.std(V_dm_mc,axis=0)
	eV_tot = np.sqrt(eV_dm**2*(V_dm/V_tot)**2)

	#measure the inner slope of the DM density profile
	b_star,eb_star=measure_beta_star(po,rs,b,epou,epol,ersu,ersl,ebu,ebl)
	print('beta_* = %.2f +/- %.2f' %(b_star,eb_star))


	#plot rotation curve comparison
	plt.figure(1)
	plt.clf()
	plt.fill_between(r,vs+evs,vs-evs,facecolor='g',alpha=0.2,edgecolor=None)
	plt.fill_between(r,V_dm+eV_dm,V_dm-eV_dm,facecolor='b',alpha=0.2,edgecolor=None)
	plt.fill_between(r,V_tot+eV_tot,V_tot-eV_tot,facecolor='purple',alpha=0.2,edgecolor=None) 
	plt.errorbar(Rrot,Vrot,yerr=eVrot,color='k',fmt='o-',capsize=3,label='CO Rotation Curve') 
	xlim=plt.gca().get_xlim()
	ylim=plt.gca().get_ylim()
	plt.plot(r,V_tot,'-',color='purple',label='Total') 
	plt.plot(r,V_dm,'b-',label='Dark Matter')  
	plt.plot(Rstar,Vstar,'g-',label='Stellar') 
	if np.all(np.isnan(Ratom))==False:
		plt.plot(Ratom,Vatom,'r-',label='Atomic Gas') 
	if np.all(np.isnan(Rmol))==False:
		plt.plot(Rmol,Vmol,'-',color='orange',label='Molecular Gas')  
	plt.xlabel('Radius (kpc)')
	plt.ylabel('Velocity (km s$^{-1}$)')
	plt.minorticks_on()
	plt.xlim(left=0.,right=xlim[1])
	plt.ylim(bottom=0.,top=ylim[1])
	plt.legend(loc='lower right')
	plt.text(0.01,0.99,
		gal_name+'\n$\\beta^*=%.2f \\pm %.2f}$' %(b_star,eb_star),
		transform=plt.gca().transAxes,va='top',ha='left')
	plt.savefig(path_dir+'rotcurve_decomp/'+gal_name+'_rotcurve_decomp'+suffix+'.pdf')

	#return quantities listed in the header
	return po, epo, rs, ers, b, eb, b_star, eb_star, r, V_tot, eV_tot, V_dm, eV_dm, rho_dm, erho_dm


