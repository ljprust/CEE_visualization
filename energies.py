from __main__ import *
import yt
from yt import YTQuantity
import matplotlib.pyplot as pl

energies_dotsize = 10
fixaxes = 0
axes = [0.0, 4.5, 1.0e7, 1.3e7]
a_rad = 7.5657e-15

gamma = 5.0/3.0
G = 6.674e-8
R = 8.314e7 / G
normalizer = 1.0e47 / G

# preallocate
time = np.zeros(nframes)
KEtot = np.zeros(nframes)
enthalpytot = np.zeros(nframes)
internaltot = np.zeros(nframes)
PEtot = np.zeros(nframes)
Etot = np.zeros(nframes)
PEtotGas = np.zeros(nframes)
PEtotDM = np.zeros(nframes)
KEtotGas = np.zeros(nframes)
KEtotDM = np.zeros(nframes)
EtotGas = np.zeros(nframes)
EtotDM = np.zeros(nframes)

# calculate
for i in range(0,nframes):
	
	num = i*frameskip + 1000000 + startingset
	numstr = str(num)
	cut = numstr[1:7]
	print 'energies: ' + simname + ' Data Set ' + cut
	
	ds = yt.load(readpath + 'star.out.' + cut)
	ad = ds.all_data()

	cl = ds.arr(1.0, 'code_length')
	cm = ds.arr(1.0, 'code_mass')
	cv = ds.arr(1.0, 'code_velocity')
	K = YTQuantity(1.0,'K')

	posCM, velCM = getCM(ds)

	phiGas = ad[('Gas','Phi')]/cl
	phiDM = ad[('DarkMatter','Phi')]/cl
	vGas = ad[('Gas','Velocities')]/cv
	vDM = ad[('DarkMatter','Velocities')]/cv
	massGas = ad[('Gas','Mass')]/cm
	massDM = ad[('DarkMatter','Mass')]/cm
	temp = ad[('Gas','Temperature')]/K

	vnormGas = np.linalg.norm( vGas - velCM, axis=1 )
	KEGas = 0.5 * np.multiply( np.multiply(vnormGas,vnormGas) , massGas )
	enthalpy = gamma / (gamma-1.0) * R * np.multiply( temp, massGas )
	internal = 3.0/2.0 * R * np.multiply( temp, massGas )
	PEGas = np.multiply( phiGas, massGas )

	vnormDM = np.linalg.norm( vDM - velCM, axis=1 )
	KEDM = 0.5 * np.multiply( np.multiply(vnormDM,vnormDM) , massDM )
	PEDM = np.multiply( phiDM, massDM )

	KEtotGas[i] = KEGas.sum() / normalizer
	enthalpytot[i] = enthalpy.sum() / normalizer
	internaltot[i] = internal.sum() / normalizer
	PEtotGas[i] = PEGas.sum() / normalizer

	KEtotDM[i] = KEDM.sum() / normalizer
	PEtotDM[i] = PEDM.sum() / normalizer

	KEtot[i] = KEtotGas[i] + KEtotDM[i]
	PEtot[i] = PEtotGas[i] + PEtotDM[i]
	Etot[i] = KEtot[i] + PEtot[i] + internaltot[i]
	EtotGas[i] = KEtotGas[i] + PEtotGas[i] + internaltot[i]
	EtotDM[i] = KEtotDM[i] + PEtotDM[i]
	
	time[i], timelabel = getTime(ds, i)

# plot
pl.clf()
pl.plot(time, internaltot, c='r', label='internal')
pl.plot(time, KEtot, c='b', label='KE_tot')
pl.plot(time, KEtotDM, c='b', linestyle=':', label='KE_DM')
pl.plot(time, KEtotGas, c='b', linestyle='--', label='KE_Gas')
pl.plot(time, PEtot, c='g', label='PE_tot')
pl.plot(time, PEtotGas, c='g', linestyle='--', label='PE_Gas')
pl.plot(time, PEtotDM, c='g', linestyle=':', label='PE_DM')
pl.plot(time, Etot, c='k', label='E_tot')
pl.plot(time, EtotGas, c='y', linestyle='-', label = 'Gas_tot')
pl.plot(time, EtotDM, c='m', linestyle='-', label = 'DM_tot')
pl.legend()

if fixaxes:
	pl.axis(axes)
	
pl.xlabel('Time (' + timelabel + ')' )
pl.ylabel('Energy Budget (code units)')
pl.title(simname + ' Energies')
energies_saveas = writepath + 'energies_' + simname + '.pdf'
pl.savefig(energies_saveas)
print 'energies: Saved figure ' + energies_saveas
pl.clf()
