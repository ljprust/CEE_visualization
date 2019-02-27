from crawler import crawlRead
from crawler import splitData
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--unbound', action='store_true')
parser.add_argument('--orbel', action='store_true')
parser.add_argument('--mass', action='store_true')
parser.add_argument('--no_latex', action='store_true')
parser.add_argument('--nplots', nargs=1, type=int)
parser.add_argument('--py2', action='store_true')
parser.add_argument('--energy', action='store_true')
args = parser.parse_args()

if (args.nplots != None) :
	nplots = args.nplots[0]
else:
	nplots = 1

if (not args.no_latex) :
	import matplotlib
	matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def savePlot(fig,name):
    fig.savefig(name)
    print('Saved plot ' + name)

def getPaths(nplots,py2):
	paths = []
	labels = []
	if nplots > 1 :
		for i in range(0,nplots) :
			if py2 :
				newpath = raw_input('Path ' + str(i+1) + ': ')
				newlabel = raw_input('Label ' + str(i+1) + ': ')
			else:
				newpath = input('Path ' + str(i+1) + ': ')
				newlabel = input('Label ' + str(i+1) + ': ')
			paths = np.append( paths, newpath )
			labels = np.append( labels, newlabel )
	else :
		paths = np.append( paths, '' )
		labels = np.append( labels, '' )
	return paths, labels

def smoothData(data,smoothrange,time):
	newtime = time[smoothrange:len(time)-smoothrange]
	datalength = len(data) - 2*smoothrange
	newdata = np.zeros(datalength)
	for i in range(0,datalength):
		toSmooth = data[i:(i+2*smoothrange+1)]
		newdata[i] = np.mean(toSmooth)
	return newdata, newtime

def collectData(nplots,paths):

	numsets = []
	data = []
	setnums = []
	time = []
	posCMx = []
	posCMy = []
	posCMz = []
	vCMx = []
	vCMy = []
	vCMz = []
	fracunbound = []
	fracunbound_i = []
	sep = []
	velCMnorm = []
	posPrimx = []
	posPrimy = []
	posPrimz = []
	posCompx = []
	posCompy = []
	posCompz = []
	massGasTot = []
	ejeceff = []
	ejeceff_i = []
	ietot = []
	ie_idealtot = []
	gasKEtot = []
	gasPEtot = []
	DMKEtot = []
	DMPEtot = []

	for i in range(0,nplots) :
		numsetsN, dataN = crawlRead(paths[i])
		setnumsN, timeN, posCMxN, posCMyN, posCMzN, vCMxN, vCMyN, vCMzN, fracunboundN, fracunbound_iN, sepN, \
		velCMnormN, posPrimxN, posPrimyN, posPrimzN, posCompxN, posCompyN, posCompzN, \
		massGasTotN, ejeceffN, ejeceff_iN, ietotN, ie_idealtotN, gasKEtotN, gasPEtotN, DMKEtotN, DMPEtotN = splitData(dataN)

		# numsets.append(numsetsN)
		# data.append(dataN)
		setnums.append(setnumsN)
		time.append(timeN)
		posCMx.append(posCMxN)
		posCMy.append(posCMyN)
		posCMz.append(posCMzN)
		vCMx.append(vCMxN)
		vCMy.append(vCMyN)
		vCMz.append(vCMzN)
		fracunbound.append(fracunboundN)
		fracunbound_i.append(fracunbound_iN)
		sep.append(sepN)
		velCMnorm.append(velCMnormN)
		posPrimx.append(posPrimxN)
		posPrimy.append(posPrimyN)
		posPrimz.append(posPrimzN)
		posCompx.append(posCompxN)
		posCompy.append(posCompyN)
		posCompz.append(posCompzN)
		massGasTot.append(massGasTotN)
		ejeceff.append(ejeceffN)
		ejeceff_i.append(ejeceff_iN)
		ietot.append(ietotN)
		ie_idealtot.append(ie_idealtotN)
		gasKEtot.append(gasKEtotN)
		gasPEtot.append(gasPEtotN)
		DMKEtot.append(DMKEtotN)
		DMPEtot.append(DMPEtotN)

	return setnums, time, posCMx, posCMy, posCMz, vCMx, vCMy, vCMz, fracunbound, fracunbound_i, \
	sep, velCMnorm, posPrimx, posPrimy, posPrimz, posCompx, posCompy, \
	posCompz, massGasTot, ejeceff, ejeceff_i, ietot, ie_idealtot, gasKEtot, gasPEtot, DMKEtot, DMPEtot

def plotMass( time, massGasTot, nplots, labels ):
    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], massGasTot[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel('Time (days)', fontsize=25 )
    plt.ylabel('Total Gas Mass', fontsize=25 )
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    saveas = 'masstot.pdf'
    fig.savefig(saveas)
    print('Saved plot ' + saveas)
    plt.clf()

def plotUnbound( time, fracunbound, ejeceff, nplots, labels ):
    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], fracunbound[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel('Time (days)', fontsize=25 )
    plt.ylabel('Unbound Mass Fraction', fontsize=25 )
    plt.axis([0.,240.,0.,0.5])
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    saveas = 'unbound.pdf'
    fig.savefig(saveas)
    print('Saved plot ' + saveas)
    plt.clf()

    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], ejeceff[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel('Time (days)', fontsize=25 )
    plt.ylabel('Ejection Efficiency', fontsize=25 )
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    saveas = 'ejeceff.pdf'
    fig.savefig(saveas)
    print('Saved plot ' + saveas)
    plt.clf()

def plotUnbound_i( time, fracunbound, ejeceff, nplots, labels ):
    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], fracunbound[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel('Time (days)', fontsize=25 )
    plt.ylabel('Unbound Mass Fraction', fontsize=25 )
    plt.axis([0.,240.,0.,0.5])
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    saveas = 'unbound_i.pdf'
    fig.savefig(saveas)
    print('Saved plot ' + saveas)
    plt.clf()

    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], ejeceff[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel('Time (days)', fontsize=25 )
    plt.ylabel('Ejection Efficiency', fontsize=25 )
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    saveas = 'ejeceff_i.pdf'
    fig.savefig(saveas)
    print('Saved plot ' + saveas)
    plt.clf()

def plotSmoothSep( nplots, labels, time, sep ):
	fig = plt.figure()
	for i in range(0,nplots):
		smoothsep, smoothtime = smoothData(sep[i],25,time[i])
		plt.plot( smoothtime, smoothsep, lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel('Time (days)', fontsize=25 )
	plt.ylabel(r'Smoothed Separation ($R_{\odot}$)', fontsize=25 )
	# plt.axis([0.,240.,0.,53.])
	plt.axis([0.,240.,2.,53.])
	plt.hlines( 3., 0., 300. )
	plt.yscale('log')
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
	saveas = 'smoothsep.pdf'
	fig.savefig(saveas)
	print('Saved plot ' + saveas)
	plt.clf()

def plotOrbEl( nplots, labels, time, sep, a, ecc, boolArray, velCMnorm, posCMx, \
posCMy, posCMz, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz ):

	if nplots > 1 :
		fig = plt.figure()
		for i in range(0,nplots):
			plt.plot( time[i], sep[i], lw=2, label=labels[i] )
		if nplots > 1 :
			plt.legend()
		plt.xlabel('Time (days)', fontsize=25 )
		plt.ylabel(r'Separation ($R_{\odot}$)', fontsize=25 )
		# plt.axis([0.,240.,0.,53.])
		plt.axis([0.,240.,2.,53.])
		plt.hlines( 3., 0., 300. )
		plt.yscale('log')
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.grid(True)
		plt.tight_layout()
	    # saveas = writepath + 'unbound_' + simname + '.pdf'
		saveas = 'seplog.pdf'
		fig.savefig(saveas)
		print('Saved plot ' + saveas)
		plt.clf()

	else :

		fig = plt.figure(figsize=(9,9))

		Time = time[0]
		BoolArray = boolArray[0]
		A = a[0]
		Ecc = ecc[0]

		plt.subplot(2,2,1)
		plt.plot(time[0], sep[0], c='b', lw=2 )

		# plt.plot(time, periapse, c='g')
		# plt.plot(time, apoapse, c='c')
		# plt.plot(Time[BoolArray], A[BoolArray], c='r', lw=2 )
		# plt.axis([0., 120., 0., 54.])
		plt.xlabel('Time (days)', fontsize=20 )
		plt.ylabel(r'separation ($R_{\odot}$)', fontsize=20 )
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.axis([0.,240.,0.,10.])
		plt.hlines( 3., 0., 300. )
		plt.grid(True)
		plt.title('Separation', fontsize=20)

		plt.subplot(2,2,3)
		plt.plot( posPrimx[0], posPrimy[0], c='g', lw=2, label='Primary' )
		plt.plot( posCompx[0], posCompy[0], c='b', lw=2, label='Companion' )
		plt.plot( posCMx[0], posCMy[0], c='r', lw=2, label='CM' )
		plt.legend()
		plt.xlabel('x (cm)', fontsize=20)
		plt.ylabel('y (cm)', fontsize=20)
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.grid(True)
		plt.title('Positions',fontsize=20)

		plt.subplot(2,2,4)
		plt.plot(time[0], velCMnorm[0], lw=2)
		plt.xlabel('Time (days)',fontsize=20 )
		plt.ylabel('CM Velocity (km/s)',fontsize=20)
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.grid(True)
		plt.title('CM Velocity',fontsize=20)

		plt.subplot(2,2,2)
		plt.plot(Time[BoolArray], Ecc[BoolArray], lw=2)
		plt.xlabel('Time (days)',fontsize=20 )
		plt.ylabel('Eccentricity',fontsize=20)
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.grid(True)
		plt.title('Eccentricity',fontsize=20)

		plt.tight_layout()

		# saveas = writepath + 'orbel_' + simname + '.pdf'
		saveas = 'orbel.pdf'
		fig.savefig(saveas)
		print('Saved figure ' + saveas)

		plt.clf()

def findAE( sep ):
    sys.stdout.write('Getting semi-major axis & eccentricity ... ')
    sys.stdout.flush()

    nframes = len(sep)
    ecc = np.zeros(nframes)
    is_peri = np.full(nframes, False, dtype = bool)
    is_apo = np.full(nframes, False, dtype = bool)
    periapse = np.zeros(nframes)
    apoapse = np.zeros(nframes)
    peridomain = np.zeros(nframes)
    apodomain = np.zeros(nframes)
    a = np.zeros(nframes)

    periapse[0] = sep[0]
    apoapse[0] = sep[0]
    a[0] = sep[0]
    pericount = 0
    apocount = 0
    peridomain[0] = 0
    apodomain[0] = 0
    is_peri[0] = True
    is_apo[0] = True
    is_peri[nframes-1] = True
    is_apo[nframes-1] = True
    for k in range(1,nframes-1):
    	if( sep[k] > sep[k-1] and sep[k] > sep[k+1]):
    		is_apo[k] = True
    		apocount = apocount + 1
    		apodomain[apocount] = k
    	if( sep[k] < sep[k-1] and sep[k] < sep[k+1]):
    		is_peri[k] = True
    		pericount = pericount + 1
    		peridomain[pericount] = k
    apodomain[apocount+1] = nframes-1
    peridomain[pericount+1] = nframes-1

    apocount = 0
    pericount = 0
    boolArray = np.zeros( nframes, dtype=bool )
    for m in range(1,nframes):
    	apoapse[m] = np.interp(m, [apodomain[apocount], apodomain[apocount+1]],
    		[sep[int(apodomain[apocount])], sep[int(apodomain[apocount+1])]])
    	periapse[m] = np.interp(m, [peridomain[pericount], peridomain[pericount+1]],
    		[sep[int(peridomain[pericount])], sep[int(peridomain[pericount+1])]])
    	if (apodomain[apocount+1] == nframes-1) or (peridomain[pericount+1] == nframes-1) :
    		a[m] = 0.
    		ecc[m] = 0.
    	elif (apodomain[apocount] == 0 or peridomain[pericount] == 0) :
    		a[m] = 0.
    		ecc[m] = 0.
    	else :
    		a[m] = (apoapse[m] + periapse[m]) / 2.0
    		ecc[m] = (apoapse[m] - periapse[m]) / (apoapse[m] + periapse[m])
    		boolArray[m] = True
    	if is_apo[m]:
    		apocount = apocount + 1
    	if is_peri[m]:
    		pericount = pericount + 1

    print('done')
    return a, ecc, boolArray

def plotEnergy( time, ietot, ie_idealtot, gasKEtot, gasPEtot, DMKEtot, DMPEtot, nplots, labels ):
	fig = plt.figure()
	for i in range(0,nplots):
		G = 6.674e-8
		ietot = np.multiply(ietot,G)
		ie_idealtot = np.multiply(ie_idealtot,G)
		gasKEtot = np.multiply(gasKEtot,G)
		gasPEtot = np.multiply(gasPEtot,G)
		DMKEtot = np.multiply(DMKEtot,G)
		DMPEtot = np.multiply(DMPEtot,G)

		KEtot = gasKEtot[i] + DMKEtot[i]
		PEtot = gasPEtot[i] + DMPEtot[i]
		Etot = ietot[i] + KEtot[i] + PEtot[i]
		Etot_ideal = ie_idealtot[i] + KEtot + PEtot
		gasEtot = ietot[i] + gasKEtot[i] + gasPEtot[i]
		gasEtot_ideal = ie_idealtot[i] + gasKEtot[i] + gasPEtot[i]
		DMEtot = DMKEtot[i] + DMPEtot[i]

		# plt.plot(time[i], ietot[i], c='r', linestyle='-', label='ie', lw=2)
		# plt.plot(time[i], ie_idealtot[i], c='r', linestyle='--', label='ie ideal', lw=2)
		# plt.plot(time[i], KEtot, c='b', label='KE tot', lw=2)
		# plt.plot(time[i], DMKEtot[i], c='b', linestyle=':', label='KE DM', lw=2)
		# plt.plot(time[i], gasKEtot[i], c='b', linestyle='--', label='KE Gas', lw=2)
		# plt.plot(time[i], PEtot, c='g', label='PE tot', lw=2)
		# plt.plot(time[i], gasPEtot[i], c='g', linestyle='--', label='PE Gas', lw=2)
		# plt.plot(time[i], DMPEtot[i], c='g', linestyle=':', label='PE DM', lw=2)
		# plt.plot(time[i], Etot, c='k', linestyle='-', label='E tot', lw=2)
		# plt.plot(time[i], Etot_ideal, c='k', linestyle='--', label='E tot ideal', lw=2)
		# plt.plot(time[i], gasEtot, c='y', linestyle='-', label = 'Gas tot', lw=2)
		# plt.plot(time[i], gasEtot_ideal, c='y', linestyle='--', label = 'Gas tot ideal', lw=2)
		# plt.plot(time[i], DMEtot, c='m', linestyle='-', label = 'DM tot', lw=2)

		plt.plot(time[i], gasPEtot[i], c='r', linestyle='-', label='PE Gas', lw=2)
		plt.plot(time[i], gasKEtot[i], c='r', linestyle=':', label='KE Gas', lw=2)
		plt.plot(time[i], DMPEtot[i], c='b', linestyle='-', label='PE DM', lw=2)
		plt.plot(time[i], DMKEtot[i], c='b', linestyle=':', label='KE DM', lw=2)
		plt.plot(time[i], DMEtot, c='g', linestyle='-', label='Tot DM', lw=2)
		plt.hlines( 0., 0., 300. )

	plt.axis([0.,240.,-1.2e48,0.3e48])

	plt.legend()
	plt.yscale('linear')
	plt.xlabel('Time (days)', fontsize=25 )
	plt.ylabel('Mechanical Energy (ergs)', fontsize=25 )
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.tight_layout()

	savePlot(fig,'energies.pdf')
	plt.clf()

paths, labels = getPaths(nplots,args.py2)
setnums, time, posCMx, posCMy, posCMz, vCMx, vCMy, vCMz, fracunbound, fracunbound_i, \
sep, velCMnorm, posPrimx, posPrimy, posPrimz, posCompx, posCompy, \
posCompz, massGasTot, ejeceff, ejeceff_i, ietot, ie_idealtot, gasKEtot, gasPEtot, DMKEtot, DMPEtot = collectData(nplots,paths)

if args.unbound :
	plotUnbound( time, fracunbound, ejeceff, nplots, labels )
	plotUnbound_i( time, fracunbound_i, ejeceff_i, nplots, labels )
if args.mass :
    plotMass( time, massGasTot, nplots, labels )
if args.orbel :
	a = []
	ecc = []
	boolArray = []
	for i in range(0,nplots):
		aN, eccN, boolArrayN = findAE( sep[i] )
		a.append(aN)
		ecc.append(eccN)
		boolArray.append(boolArrayN)

	plotOrbEl( nplots, labels, time, sep, a, ecc, boolArray, velCMnorm, posCMx, \
	posCMy, posCMz, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz )
	plotSmoothSep( nplots, labels, time, sep )
if args.energy :
	plotEnergy( time, ietot, ie_idealtot, gasKEtot, gasPEtot, DMKEtot, DMPEtot, nplots, labels )
