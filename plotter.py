from crawler import crawlRead
from crawler import splitData
import numpy as np
import math
import argparse
import sys

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--unbound', action='store_true')
parser.add_argument('--orbel', action='store_true')
parser.add_argument('--mass', action='store_true')
parser.add_argument('--energy', action='store_true')
parser.add_argument('--no_latex', action='store_true')
parser.add_argument('--nplots', nargs=1, type=int)
parser.add_argument('--py2', action='store_true')
args = parser.parse_args()

movingBC = True
G = 6.674e-8
Rsun = 7.0e10
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9476bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

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
	DMKEtot = []
	velCMDMnorm = []
	fracunbound_noIe = []
	ejeceff_noIe = []
	gasKEunbound = []
	gasKEbound = []
	gasIEunbound = []
	gasIEbound = []
	PECoreGasUnboundPrim = []
	PECoreGasBoundPrim = []
	PECoreGasUnboundComp = []
	PECoreGasBoundComp = []
	PECoreCore = []
	PEGasGasUnbound = []
	PEGasGasBound = []
	Emech = []
	gasPbound = []
	gasPunbound = []
	gasPxbound = []
	gasPybound = []
	gasPzbound = []
	gasPxunbound = []
	gasPyunbound = []
	gasPzunbound = []
	gasLbound = []
	gasLunbound = []
	gasLxbound = []
	gasLybound = []
	gasLzbound = []
	gasLxunbound = []
	gasLyunbound = []
	gasLzunbound = []
	corePx = []
	corePy = []
	corePz = []
	compPx = []
	compPy = []
	compPz = []
	coreLx = []
	coreLy = []
	coreLz = []
	compLx = []
	compLy = []
	compLz = []
	reflectiveCount = []
	edgeCount = []
	mirrorLeft = []
	mirrorRight = []
	mirrorRadius = []
	mirrorCenterX = []
	mirrorCenterY = []
	mirrorCenterZ = []
	mirrorMass = []
	mirrorVelX = []
	mirrorVelY = []
	mirrorVelZ = []
	mirrorForceX = []
	mirrorForceY = []
	mirrorForceZ = []
	mirrorGravX = []
	mirrorGravY = []
	mirrorGravZ = []
	mirrorGravCorrX = []
	mirrorGravCorrY = []
	mirrorGravCorrZ = []
	dynFric = []
	dynFricV = []
	dynFricNoCorr = []
	mPrim = []
	mComp = []
	gravPrimGasX = []
	gravPrimGasY = []
	gravPrimGasZ = []

	for i in range(0,nplots) :
		numsetsN, dataN = crawlRead(paths[i])
		setnumsN, timeN, posCMxN, posCMyN, posCMzN, vCMxN, vCMyN, vCMzN, fracunboundN, fracunbound_iN, sepN, \
		velCMnormN, posPrimxN, posPrimyN, posPrimzN, posCompxN, posCompyN, posCompzN, \
		massGasTotN, ejeceffN, ejeceff_iN, ietotN, ie_idealtotN, gasKEtotN, DMKEtotN, velCMDMnormN, fracunbound_noIeN, ejeceff_noIeN, \
		gasKEunboundN, gasKEboundN, gasIEunboundN, gasIEboundN, PECoreGasUnboundPrimN, PECoreGasBoundPrimN, PECoreGasUnboundCompN, PECoreGasBoundCompN, PECoreCoreN, PEGasGasUnboundN, PEGasGasBoundN, \
		EmechN, gasPboundN, gasPunboundN, gasPxboundN, gasPyboundN, gasPzboundN, gasPxunboundN, gasPyunboundN, gasPzunboundN, gasLboundN, gasLunboundN, gasLxboundN, gasLyboundN, gasLzboundN, gasLxunboundN, gasLyunboundN, gasLzunboundN, \
		corePxN, corePyN, corePzN, compPxN, compPyN, compPzN, coreLxN, coreLyN, coreLzN, compLxN, compLyN, compLzN, \
		reflectiveCountN, edgeCountN, mirrorLeftN, mirrorRightN, mirrorRadiusN, mirrorCenterXN, mirrorCenterYN, mirrorCenterZN, \
        mirrorMassN, mirrorVelXN, mirrorVelYN, mirrorVelZN, mirrorForceXN, mirrorForceYN, mirrorForceZN, mirrorGravXN, mirrorGravYN, mirrorGravZN, \
		mirrorGravCorrXN, mirrorGravCorrYN, mirrorGravCorrZN, dynFricN, dynFricVN, dynFricNoCorrN, \
		mPrimN, mCompN, gravPrimGasXN, gravPrimGasYN, gravPrimGasZN = splitData(dataN)

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
		DMKEtot.append(DMKEtotN)
		velCMDMnorm.append(velCMDMnormN)
		fracunbound_noIe.append(fracunbound_noIeN)
		ejeceff_noIe.append(ejeceff_noIeN)
		gasKEunbound.append(gasKEunboundN)
		gasKEbound.append(gasKEboundN)
		gasIEunbound.append(gasIEunboundN)
		gasIEbound.append(gasIEboundN)
		PECoreGasUnboundPrim.append(PECoreGasUnboundPrimN)
		PECoreGasBoundPrim.append(PECoreGasBoundPrimN)
		PECoreGasUnboundComp.append(PECoreGasUnboundCompN)
		PECoreGasBoundComp.append(PECoreGasBoundCompN)
		PECoreCore.append(PECoreCoreN)
		PEGasGasUnbound.append(PEGasGasUnboundN)
		PEGasGasBound.append(PEGasGasBoundN)
		Emech.append(EmechN)
		gasPbound.append(gasPboundN)
		gasPunbound.append(gasPunboundN)
		gasPxbound.append(gasPxboundN)
		gasPybound.append(gasPyboundN)
		gasPzbound.append(gasPzboundN)
		gasPxunbound.append(gasPxunboundN)
		gasPyunbound.append(gasPyunboundN)
		gasPzunbound.append(gasPzunboundN)
		gasLbound.append(gasLboundN)
		gasLunbound.append(gasLunboundN)
		gasLxbound.append(gasLxboundN)
		gasLybound.append(gasLyboundN)
		gasLzbound.append(gasLzboundN)
		gasLxunbound.append(gasLxunboundN)
		gasLyunbound.append(gasLyunboundN)
		gasLzunbound.append(gasLzunboundN)
		corePx.append(corePxN)
		corePy.append(corePyN)
		corePz.append(corePzN)
		compPx.append(compPxN)
		compPy.append(compPyN)
		compPz.append(compPzN)
		coreLx.append(coreLxN)
		coreLy.append(coreLyN)
		coreLz.append(coreLzN)
		compLx.append(compLxN)
		compLy.append(compLyN)
		compLz.append(compLzN)
		reflectiveCount.append(reflectiveCountN)
		edgeCount.append(edgeCountN)
		mirrorLeft.append(mirrorLeftN)
		mirrorRight.append(mirrorRightN)
		mirrorRadius.append(mirrorRadiusN)
		mirrorCenterX.append(mirrorCenterXN)
		mirrorCenterY.append(mirrorCenterYN)
		mirrorCenterZ.append(mirrorCenterZN)
		mirrorMass.append(mirrorMassN)
		mirrorVelX.append(mirrorVelXN)
		mirrorVelY.append(mirrorVelYN)
		mirrorVelZ.append(mirrorVelZN)
		mirrorForceX.append(mirrorForceXN)
		mirrorForceY.append(mirrorForceYN)
		mirrorForceZ.append(mirrorForceZN)
		mirrorGravX.append(mirrorGravXN)
		mirrorGravY.append(mirrorGravYN)
		mirrorGravZ.append(mirrorGravZN)
		mirrorGravCorrX.append(mirrorGravCorrXN)
		mirrorGravCorrY.append(mirrorGravCorrYN)
		mirrorGravCorrZ.append(mirrorGravCorrZN)
		dynFric.append(dynFricN)
		dynFricV.append(dynFricVN)
		dynFricNoCorr.append(dynFricNoCorrN)
		mPrim.append(mPrimN)
		mComp.append(mCompN)
		gravPrimGasX.append(gravPrimGasXN)
		gravPrimGasY.append(gravPrimGasYN)
		gravPrimGasZ.append(gravPrimGasZN)

	return setnums, time, posCMx, posCMy, posCMz, vCMx, vCMy, vCMz, fracunbound, fracunbound_i, \
	sep, velCMnorm, posPrimx, posPrimy, posPrimz, posCompx, posCompy, \
	posCompz, massGasTot, ejeceff, ejeceff_i, ietot, ie_idealtot, gasKEtot, DMKEtot, velCMDMnorm, fracunbound_noIe, ejeceff_noIe, \
	gasKEunbound, gasKEbound, gasIEunbound, gasIEbound, PECoreGasUnboundPrim, PECoreGasBoundPrim, PECoreGasUnboundComp, PECoreGasBoundComp, PECoreCore, PEGasGasUnbound, PEGasGasBound, \
	Emech, gasPbound, gasPunbound, gasPxbound, gasPybound, gasPzbound, gasPxunbound, gasPyunbound, gasPzunbound, gasLbound, gasLunbound, gasLxbound, gasLybound, gasLzbound, gasLxunbound, gasLyunbound, gasLzunbound, \
	corePx, corePy, corePz, compPx, compPy, compPz, coreLx, coreLy, coreLz, compLx, compLy, compLz, \
	reflectiveCount, edgeCount, mirrorLeft, mirrorRight, mirrorRadius, mirrorCenterX, mirrorCenterY, mirrorCenterZ, \
	mirrorMass, mirrorVelX, mirrorVelY, mirrorVelZ, mirrorForceX, mirrorForceY, mirrorForceZ, mirrorGravX, mirrorGravY, mirrorGravZ, \
	mirrorGravCorrX, mirrorGravCorrY, mirrorGravCorrZ, dynFric, dynFricV, dynFricNoCorr, mPrim, mComp, gravPrimGasX, gravPrimGasY, gravPrimGasZ

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
	savePlot(fig,'masstot.pdf')
	plt.clf()

def plotEnergy( time, ietot, ie_idealtot, Emech, nplots, labels ):
	fig = plt.figure()

	ietot = np.multiply(ietot,G)
	ie_idealtot = np.multiply(ie_idealtot,G)
	Emech = np.multiply(Emech,G)
	# gasKEtot = np.multiply(gasKEtot,G)
	# gasPEtot = np.multiply(gasPEtot,G)
	# DMKEtot = np.multiply(DMKEtot,G)
	# DMPEtot = np.multiply(DMPEtot,G)

	for i in range(0,nplots):

		# KEtot = gasKEtot[i] + DMKEtot[i]
		# PEtot = gasPEtot[i] + DMPEtot[i]
		# Etot = ietot[i] + KEtot[i] + PEtot[i]
		# Etot_ideal = ie_idealtot[i] + KEtot + PEtot
		# gasEtot = ietot[i] + gasKEtot[i] + gasPEtot[i]
		# gasEtot_ideal = ie_idealtot[i] + gasKEtot[i] + gasPEtot[i]
		# DMEtot = DMKEtot[i] + DMPEtot[i]
		Etot = Emech[i] + ietot[i]

		# plt.plot(time[i], ietot[i], c='r', linestyle='-', label='ie', lw=2)
		# plt.plot(time[i], ie_idealtot[i], c='r', linestyle='--', label='ie ideal', lw=2)
		# plt.plot(time[i], KEtot, c='b', label='KE tot', lw=2)
		# plt.plot(time[i], DMKEtot[i], c='b', linestyle=':', label='KE DM', lw=2)
		# plt.plot(time[i], gasKEtot[i], c='b', linestyle='--', label='KE Gas', lw=2)
		# plt.plot(time[i], PEtot, c='g', label='PE tot', lw=2)
		# plt.plot(time[i], gasPEtot[i], c='g', linestyle='--', label='PE Gas', lw=2)
		# plt.plot(time[i], DMPEtot[i], c='g', linestyle=':', label='PE DM', lw=2)
		plt.plot(time[i], Etot, linestyle='-', label=labels[i], lw=2)
		# plt.plot(time[i], Etot_ideal, c='k', linestyle='--', label='E tot ideal', lw=2)
		# plt.plot(time[i], gasEtot, c='y', linestyle='-', label = 'Gas tot', lw=2)
		# plt.plot(time[i], gasEtot_ideal, c='y', linestyle='--', label = 'Gas tot ideal', lw=2)
		# plt.plot(time[i], DMEtot, c='m', linestyle='-', label = 'DM tot', lw=2)

		# plt.plot(time[i], gasPEtot[i], c='r', linestyle='-', label='PE Gas', lw=2)
		# plt.plot(time[i], gasKEtot[i], c='r', linestyle=':', label='KE Gas', lw=2)
		# plt.plot(time[i], DMPEtot[i], c='b', linestyle='-', label='PE DM', lw=2)
		# plt.plot(time[i], DMKEtot[i], c='b', linestyle=':', label='KE DM', lw=2)
		# plt.hlines( 0., 0., 300. )

	if nplots > 1 :
		plt.legend()

	# plt.axis([0.,240.,-1.2e48,0.3e48])

	# plt.legend()
	plt.yscale('linear')
	plt.xlabel('Time (days)', fontsize=25 )
	plt.ylabel('Energy (ergs)', fontsize=25 )
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.tight_layout()

	savePlot(fig,'energies.pdf')
	plt.clf()

def plotMomentum( time, Emech, gasPbound, gasPunbound, gasPxbound, gasPybound, gasPzbound, gasPxunbound, gasPyunbound, gasPzunbound, gasLbound, gasLunbound, gasLxbound, gasLybound, gasLzbound, gasLxunbound, gasLyunbound, gasLzunbound, corePx, corePy, corePz, compPx, compPy, compPz, coreLx, coreLy, coreLz, compLx, compLy, compLz, nplots, labels ):
	colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9476bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], Emech[i], c=colors[i], lw=2, linestyle='-', label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~({\rm d})$', fontsize=25 )
	plt.ylabel('Mechnical Energy', fontsize=25 )
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,2.,0.,0.002])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'Emech.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		coreP = np.sqrt( corePx[i]*corePx[i] + corePy[i]*corePy[i] + corePz[i]*corePz[i] )
		compP = np.sqrt( compPx[i]*compPx[i] + compPy[i]*compPy[i] + compPz[i]*compPz[i] )
		plt.plot( time[i], gasPbound[i]+gasPunbound[i]+coreP+compP, c=colors[i], lw=2, linestyle='-', label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~({\rm d})$', fontsize=25 )
	plt.ylabel('Linear Momentum', fontsize=25 )
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,2.,0.,0.002])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'momentum.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		coreL = np.sqrt( coreLx[i]*coreLx[i] + coreLy[i]*coreLy[i] + coreLz[i]*coreLz[i] )
		compL = np.sqrt( compLx[i]*compLx[i] + compLy[i]*compLy[i] + compLz[i]*compLz[i] )
		plt.plot( time[i], gasLbound[i]+gasLunbound[i]+coreL+compL, c=colors[i], lw=2, linestyle='-', label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~({\rm d})$', fontsize=25 )
	plt.ylabel('Angular Momentum', fontsize=25 )
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,2.,0.,0.002])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'angmomentum.pdf')
	plt.clf()

def plotL( time, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz, mPrim, mComp, corePx, corePy, corePz, compPx, compPy, compPz, nplots, labels ) :

	fig = plt.figure()
	for i in range(0,nplots):
		posPrim = np.array([posPrimx[i], posPrimy[i], posPrimz[i]])
		posComp = np.array([posCompx[i], posCompy[i], posCompz[i]])
		posCMDM = (mPrim[i]*posPrim+mComp[i]*posComp)/(mPrim[i]+mComp[i])
		vPrim = np.array([corePx[i]/mPrim[i],corePy[i]/mPrim[i],corePz[i]/mPrim[i]])
		vComp = np.array([compPx[i]/mComp[i],compPy[i]/mComp[i],compPz[i]/mComp[i]])
		velCMDM = (mPrim[i]*vPrim+mComp[i]*vComp)/(mPrim[i]+mComp[i])

		Lprim = np.zeros(np.shape(posPrim))
		Lcomp = np.zeros(np.shape(posPrim))
		LprimPlane = np.zeros(len(posPrimx[i]))
		LcompPlane = np.zeros(len(posPrimx[i]))
		LtotPlane = np.zeros(len(posPrimx[i]))

		for j in range(0, len(posPrimx[i]) ) :
			Lprim[:,j] = mPrim[i][j]*np.cross( posPrim[:,j]-posCMDM[:,j], vPrim[:,j]-velCMDM[:,j] )
			Lcomp[:,j] = mComp[i][j]*np.cross( posComp[:,j]-posCMDM[:,j], vComp[:,j]-velCMDM[:,j] )

			vRel = vComp[:,j] - vPrim[:,j]
			vRelUnit = vRel / np.linalg.norm(vRel)
			rRelUnit = (posComp[:,j]-posPrim[:,j])/np.linalg.norm(posComp[:,j]-posPrim[:,j])
			nplane = np.cross(rRelUnit,vRelUnit)
			nplane = nplane / np.linalg.norm(nplane)

			LprimPlane[j] = np.dot(Lprim[:,j],nplane)
			LcompPlane[j] = np.dot(Lcomp[:,j],nplane)
			LtotPlane[j] = LprimPlane[j] + LcompPlane[j]

		plt.plot( time[i], LtotPlane*np.sqrt(G), c=colors[i], lw=2, linestyle='-', label=labels[i] )
	if nplots > 1 :
		plt.legend(prop={'size': 15})
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel(r'$L~/~{\rm g~cm^{2}~s^{-1}}$', fontsize=25 )
	plt.axis([10.,40.,0.5e52,1.5e52])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	# saveas = writepath + 'unbound_' + simname + '.pdf'
	savePlot(fig,'L.pdf')
	plt.clf()

def plotTorques( time, sep, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz, mPrim, mComp, mirrorForceX, mirrorForceY, mirrorForceZ, mirrorGravX, mirrorGravY, mirrorGravZ, mirrorGravCorrX, mirrorGravCorrY, mirrorGravCorrZ, gravPrimGasX, gravPrimGasY, gravPrimGasZ, dynFric, dynFricV, dynFricNoCorr, corePx, corePy, corePz, compPx, compPy, compPz, nplots, labels ) :

	fig = plt.figure()
	i=0
	posPrim = np.array([posPrimx[i], posPrimy[i], posPrimz[i]])
	posComp = np.array([posCompx[i], posCompy[i], posCompz[i]])
	posCMDM = (mPrim[i]*posPrim+mComp[i]*posComp)/(mPrim[i]+mComp[i])
	vPrim = np.array([corePx[i]/mPrim[i],corePy[i]/mPrim[i],corePz[i]/mPrim[i]])
	vComp = np.array([compPx[i]/mComp[i],compPy[i]/mComp[i],compPz[i]/mComp[i]])
	velCMDM = (mPrim[i]*vPrim+mComp[i]*vComp)/(mPrim[i]+mComp[i])
	forceCoreCore = mPrim[i]*mComp[i]/sep[i]/sep[i]/sep[i]/Rsun/Rsun/Rsun*(posPrim-posComp)
	hydroForce = np.array([mirrorForceX[i],mirrorForceY[i],mirrorForceZ[i]])
	gravForceComp = np.array([mirrorGravX[i],mirrorGravY[i],mirrorGravZ[i]]) * mComp[i]
	gravForcePrim = np.array([gravPrimGasX[i],gravPrimGasY[i],gravPrimGasZ[i]]) - forceCoreCore

	accelCMDM = (hydroForce+gravForceComp+gravForcePrim)/(mPrim[i]+mComp[i])

	gravTorquePrim = np.zeros(np.shape(posPrim))
	gravTorqueComp = np.zeros(np.shape(posPrim))
	hydroTorque = np.zeros(np.shape(posPrim))
	totalTorque = np.zeros(np.shape(posPrim))
	gravTorquePrimCMDM = np.zeros(np.shape(posPrim))
	gravTorqueCompCMDM = np.zeros(np.shape(posPrim))
	hydroTorqueCMDM = np.zeros(np.shape(posPrim))
	totalTorqueCMDM = np.zeros(np.shape(posPrim))
	gravTorquePrimCMDMaccel = np.zeros(np.shape(posPrim))
	gravTorqueCompCMDMaccel = np.zeros(np.shape(posPrim))
	hydroTorqueCMDMaccel = np.zeros(np.shape(posPrim))
	totalTorqueCMDMaccel = np.zeros(np.shape(posPrim))
	gravTorqueRel = np.zeros(np.shape(posPrim))
	hydroTorqueRel = np.zeros(np.shape(posPrim))
	totalTorqueRel = np.zeros(np.shape(posPrim))

	gravTorquePrimPlane = np.zeros(len(posPrimx[i]))
	gravTorqueCompPlane = np.zeros(len(posPrimx[i]))
	hydroTorquePlane = np.zeros(len(posPrimx[i]))
	totalTorquePlane = np.zeros(len(posPrimx[i]))
	gravTorquePrimCMDMPlane = np.zeros(len(posPrimx[i]))
	gravTorqueCompCMDMPlane = np.zeros(len(posPrimx[i]))
	hydroTorqueCMDMPlane = np.zeros(len(posPrimx[i]))
	totalTorqueCMDMPlane = np.zeros(len(posPrimx[i]))
	gravTorquePrimCMDMaccelPlane = np.zeros(len(posPrimx[i]))
	gravTorqueCompCMDMaccelPlane = np.zeros(len(posPrimx[i]))
	hydroTorqueCMDMaccelPlane = np.zeros(len(posPrimx[i]))
	totalTorqueCMDMaccelPlane = np.zeros(len(posPrimx[i]))
	gravTorqueRelPlane = np.zeros(len(posPrimx[i]))
	hydroTorqueRelPlane = np.zeros(len(posPrimx[i]))
	totalTorqueRelPlane = np.zeros(len(posPrimx[i]))

	for j in range(0, len(posPrimx[i]) ) :
		gravTorquePrim[:,j] = np.cross(posPrim[:,j],gravForcePrim[:,j])
		gravTorqueComp[:,j] = np.cross(posComp[:,j],gravForceComp[:,j])
		hydroTorque[:,j] = np.cross(posComp[:,j],hydroForce[:,j])
		totalTorque[:,j] = gravTorquePrim[:,j] + gravTorqueComp[:,j] + hydroTorque[:,j]

		gravTorquePrimCMDM[:,j] = np.cross(posPrim[:,j]-posCMDM[:,j],gravForcePrim[:,j])
		gravTorqueCompCMDM[:,j] = np.cross(posComp[:,j]-posCMDM[:,j],gravForceComp[:,j])
		hydroTorqueCMDM[:,j] = np.cross(posComp[:,j]-posCMDM[:,j],hydroForce[:,j])
		totalTorqueCMDM[:,j] = gravTorquePrimCMDM[:,j] + gravTorqueCompCMDM[:,j] + hydroTorqueCMDM[:,j]

		gravTorquePrimCMDMaccel[:,j] = np.cross(posPrim[:,j]-posCMDM[:,j],gravForcePrim[:,j]-mPrim[i][j]*accelCMDM[:,j])
		gravTorqueCompCMDMaccel[:,j] = np.cross(posComp[:,j]-posCMDM[:,j],gravForceComp[:,j]-mComp[i][j]*accelCMDM[:,j])
		hydroTorqueCMDMaccel[:,j] = np.cross(posComp[:,j]-posCMDM[:,j],hydroForce[:,j])
		totalTorqueCMDMaccel[:,j] = gravTorquePrimCMDMaccel[:,j] + gravTorqueCompCMDMaccel[:,j] + hydroTorqueCMDMaccel[:,j]

		gravTorqueRel[:,j] = np.cross(posComp[:,j]-posPrim[:,j],gravForceComp[:,j])
		hydroTorqueRel[:,j] = np.cross(posComp[:,j]-posPrim[:,j],hydroForce[:,j])
		totalTorqueRel[:,j] = gravTorqueRel[:,j] + hydroTorqueRel[:,j]

		vRel = vComp[:,j] - vPrim[:,j]
		vRelUnit = vRel / np.linalg.norm(vRel)
		rRelUnit = (posComp[:,j]-posPrim[:,j])/np.linalg.norm(posComp[:,j]-posPrim[:,j])
		nplane = np.cross(rRelUnit,vRelUnit)
		nplane = nplane / np.linalg.norm(nplane)

		gravTorquePrimPlane[j] = -np.dot(gravTorquePrim[:,j],nplane)
		gravTorqueCompPlane[j] = -np.dot(gravTorqueComp[:,j],nplane)
		hydroTorquePlane[j] = -np.dot(hydroTorque[:,j],nplane)
		totalTorquePlane[j] = -np.dot(totalTorque[:,j],nplane)
		gravTorquePrimCMDMPlane[j] = -np.dot(gravTorquePrimCMDM[:,j],nplane)
		gravTorqueCompCMDMPlane[j] = -np.dot(gravTorqueCompCMDM[:,j],nplane)
		hydroTorqueCMDMPlane[j] = -np.dot(hydroTorqueCMDM[:,j],nplane)
		totalTorqueCMDMPlane[j] = -np.dot(totalTorqueCMDM[:,j],nplane)
		gravTorquePrimCMDMaccelPlane[j] = -np.dot(gravTorquePrimCMDMaccel[:,j],nplane)
		gravTorqueCompCMDMaccelPlane[j] = -np.dot(gravTorqueCompCMDMaccel[:,j],nplane)
		hydroTorqueCMDMaccelPlane[j] = -np.dot(hydroTorqueCMDMaccel[:,j],nplane)
		totalTorqueCMDMaccelPlane[j] = -np.dot(totalTorqueCMDMaccel[:,j],nplane)
		gravTorqueRelPlane[j] = -np.dot(gravTorqueRel[:,j],nplane)
		hydroTorqueRelPlane[j] = -np.dot(hydroTorqueRel[:,j],nplane)
		totalTorqueRelPlane[j] = -np.dot(totalTorqueRel[:,j],nplane)

	plt.plot( time[i], hydroTorqueCMDMPlane*G, c=colors[0], lw=2, linestyle='-', label='Hydrodynamic' )
	plt.plot( time[i], (gravTorquePrimCMDMPlane+gravTorqueCompCMDMPlane)*G, c=colors[1], lw=2, linestyle='-', label='Gravitational')
	plt.plot( time[i], totalTorqueCMDMPlane*G, c=colors[2], lw=2, linestyle='-', label='Total (Inertial)' )
	#plt.plot( time[i], totalTorqueCMDMaccelPlane*G, c=colors[3], lw=2, linestyle=':', label='Total (Non-Inertial)' )
	plt.legend(prop={'size': 12})
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel(r'${\rm Torque~/~dyn~cm}$', fontsize=25 )
	plt.axis([0.,40.,-2.4e46,2.0e46])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'torque.pdf')
	plt.clf()

	fig = plt.figure()
	plt.plot( time[i], hydroTorqueRelPlane*G, c=colors[0], lw=2, linestyle='-', label='Hydro' )
	plt.plot( time[i], gravTorqueRelPlane*G, c=colors[1], lw=2, linestyle='-', label='Grav')
	plt.plot( time[i], totalTorqueRelPlane*G, c=colors[2], lw=2, linestyle='-', label='Total' )
	plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel(r'${\rm Torque~/~dyn~cm}$', fontsize=25 )
	plt.axis([0.,40.,-6.0e46,6.0e46])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'torquerel.pdf')
	plt.clf()


def plotMirrorForces( time, mirrorMass, mirrorRadius, mirrorForceX, mirrorForceY, mirrorForceZ, mirrorGravX, mirrorGravY, mirrorGravZ, mirrorGravCorrX, mirrorGravCorrY, mirrorGravCorrZ, dynFric, dynFricV, dynFricNoCorr, sep, nplots, labels ):
	colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9476bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

	fig = plt.figure()
	for i in range(0,nplots):
		mirrorTotalForceX = mirrorForceX[i] + mirrorGravX[i] * mirrorMass[i]
		mirrorTotalForceY = mirrorForceY[i] + mirrorGravY[i] * mirrorMass[i]
		mirrorTotalForceZ = mirrorForceZ[i] + mirrorGravZ[i] * mirrorMass[i]
		mirrorTotalForce = np.sqrt(mirrorTotalForceX*mirrorTotalForceX+mirrorTotalForceY*mirrorTotalForceY+mirrorTotalForceZ*mirrorTotalForceZ)
		plt.plot( time[i], mirrorTotalForce, c=colors[i], lw=2, linestyle='-', label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel('Mirror Total Force', fontsize=25 )
	# plt.axis([0.,240.,0.,0.4])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	# saveas = writepath + 'unbound_' + simname + '.pdf'
	savePlot(fig,'mirrorforce.pdf')
	plt.clf()

	fig, ax1 = plt.subplots()
	ax1.set_xlabel(r'$t~/~{\rm d}$', fontsize=25)
	ax1.set_ylabel(r'$F_{\rm g,drag}~/~{\rm dynes}$', fontsize=25)
	if nplots == 1 :
		ax1.plot( time[i], dynFric[i]*G, c=colors[1], lw=2, linestyle='-', label='Dynamical Friction' )
		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)
		plt.grid(True)
		ax2 = ax1.twinx()
		ax2.set_ylabel(r'$r~/~{\rm R_{\odot}}$', fontsize=25)
		ax2.plot( time[i], sep[i], c=colors[0], lw=2, label='Separation' )
		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)
		#plt.legend(prop={'size':15})
		ax2.axis([0.,40.,0.,60.])
		ax1.axis([0.,40.,-6.0e34,5.0e34])
	plt.tight_layout()
	savePlot(fig,'dynfric.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		mirrorGravForceX = mirrorGravX[i] * mirrorMass[i]
		mirrorGravForceY = mirrorGravY[i] * mirrorMass[i]
		mirrorGravForceZ = mirrorGravZ[i] * mirrorMass[i]
		mirrorGravForce = np.sqrt(mirrorGravForceX*mirrorGravForceX+mirrorGravForceY*mirrorGravForceY+mirrorGravForceZ*mirrorGravForceZ)
		plt.plot( time[i], mirrorGravForce, c=colors[i], lw=2, linestyle='-', label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel('Mirror Grav Force', fontsize=25 )
	# plt.axis([0.,240.,0.,0.4])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	# saveas = writepath + 'unbound_' + simname + '.pdf'
	savePlot(fig,'gravforce.pdf')
	plt.clf()

	fig = plt.figure()
	avgForce = np.zeros(nplots)
	radius = np.zeros(nplots)
	for i in range(0,nplots):
		mirrorForce = np.sqrt(mirrorForceX[i]*mirrorForceX[i]+mirrorForceY[i]*mirrorForceY[i]+mirrorForceZ[i]*mirrorForceZ[i])
		plt.plot( time[i], mirrorForce*G, c=colors[i], lw=2, linestyle='-', label=labels[i] )

		startTime = 0.
		endTime = 60.
		beforeArray = time[i] < endTime
		afterArray = time[i] > startTime
		boolArray = np.logical_and(beforeArray,afterArray)
		mirrorForceCut = mirrorForce[boolArray]
		dynFricCut = dynFric[i][boolArray]
		avgForce[i] = mirrorForceCut.mean()
		dynFricCutAbs = np.absolute(dynFricCut)
		avgDynFric = dynFricCutAbs.mean()
		radius[i] = mirrorRadius[i][0]/7.0e10
		print('Radius:',radius[i],'average force:',avgForce[i],'dyn fric:',avgDynFric)

	if nplots > 1 :
		plt.legend(prop={'size': 15})
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel(r'$F_{\rm hy}~/~{\rm dynes}$', fontsize=25 )
	plt.axis([0.,40.,0.,1.2e34])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	# saveas = writepath + 'unbound_' + simname + '.pdf'
	savePlot(fig,'gasforce.pdf')
	plt.clf()

	if nplots == 1 :
		fig = plt.figure()
		plt.plot( time[i], dynFric[i]*G, c=colors[0], lw=2, linestyle='-', label='Dynamical' )
		plt.plot( time[i], mirrorForce*G, c=colors[1], lw=2, linestyle='-', label='Hydrodynamic' )
		plt.plot( time[i], dynFricV[i]*G, c=colors[2], lw=2, linestyle='-', label='Along v' )
		plt.plot( time[i], dynFricNoCorr[i]*G, c=colors[3], lw=2, linestyle='-', label='Lab Frame' )
		plt.legend(fontsize=15)
		plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
		plt.ylabel(r'$F_{\rm drag}~/~{\rm dynes}$', fontsize=25 )
		plt.axis([0.,40.,-6.0e34,4.6e34])
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.grid(True)
		plt.tight_layout()
		savePlot(fig,'forcecomp.pdf')
		plt.clf()

	fig = plt.figure()
	plt.scatter( radius, avgForce*G, lw=2, linestyle='-', c=colors[0], label='Simulation' )
	myRadius = 4.0
	myMass = 0.99 * 2.0e33
	myRho = 1.52e-5
	theoryForce = 2.0 * math.pi * myRadius*7.0e10 * myMass * myRho * G
	plt.plot( [0.,myRadius], [0.,theoryForce], c=colors[1], label='Theory' )
	plt.xlabel(r'${\rm Boundary~Radius}~/~{\rm R_{\odot}}$', fontsize=25 )
	plt.ylabel(r'$F_{\rm hy}~/~{\rm dynes}$', fontsize=25 )
	plt.legend(fontsize=15)
	plt.axis([0.,4.,0.,3.5e33])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'forcevsradius.pdf')
	plt.clf()

def plotUnbound( time, fracunbound, ejeceff, fracunbound_noIe, ejeceff_noIe, nplots, labels ):
	colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9476bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], fracunbound[i], c=colors[i], lw=2, linestyle='-', label=labels[i] )
	# for i in range(0,nplots):
	# 	plt.plot( time[i], fracunbound_noIe[i], c=colors[i], lw=2, linestyle='--', label=labels[i] + ' (No Internal Energy)' )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel('Unbound Mass Fraction', fontsize=25 )
	# plt.axis([0.,240.,0.,0.4])
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	# saveas = writepath + 'unbound_' + simname + '.pdf'
	savePlot(fig,'unbound.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], ejeceff[i], c=colors[i], lw=2, linestyle='-', label=labels[i] )
	# for i in range(0,nplots):
	# 	plt.plot( time[i], ejeceff_noIe[i], c=colors[i], lw=2, linestyle='--', label=labels[i] + ' (No Internal Energy)' )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~({\rm d})$', fontsize=25 )
	plt.ylabel(r'$f_{\rm unb}$', fontsize=25 )
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,25.,0.,0.01])
	plt.grid(True)
	plt.tight_layout()
	# saveas = writepath + 'unbound_' + simname + '.pdf'
	savePlot(fig,'ejeceff.pdf')
	plt.clf()

def plotUnbound_i( time, fracunbound_i, ejeceff_i, nplots, labels ):
    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], fracunbound_i[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
    plt.ylabel('Unbound Mass Fraction', fontsize=25 )
    # plt.axis([0.,240.,0.,0.5])
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    savePlot(fig,'unbound_i.pdf')
    plt.clf()

    fig = plt.figure()
    for i in range(0,nplots):
    	plt.plot( time[i], ejeceff_i[i], lw=2, label=labels[i] )
    if nplots > 1 :
        plt.legend()
    plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
    plt.ylabel('Ejection Efficiency', fontsize=25 )
    plt.xticks( fontsize=20)
    plt.yticks( fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # saveas = writepath + 'unbound_' + simname + '.pdf'
    savePlot(fig,'ejeceff_i.pdf')
    plt.clf()

def plotSmoothSep( nplots, labels, time, sep ):
	fig = plt.figure()
	for i in range(0,nplots):
		smoothsep, smoothtime = smoothData(sep[i],7,time[i])
		smoothsep, smoothtime = smoothData(smoothsep,7,smoothtime)
		# print(smoothsep[len(smoothsep)-1])
		plt.plot( smoothtime, smoothsep, lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend(prop={'size':15})
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel(r'$r_{\rm smoothed}~/~{\rm R_{\odot}}$', fontsize=25 )
	plt.axis([15.,60.,5.,30.])
	# plt.hlines( 1.9935 + 0.99, 0., 1000. ) # paper
	plt.yscale('log')
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'smoothsep.pdf')
	plt.clf()

def plotOrbEl( nplots, labels, time, sep, a, ecc, boolArray, velCMnorm, posCMx, \
posCMy, posCMz, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz, velCMDMnorm, numorbits ):

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], sep[i], lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend(prop={'size':15})
	plt.xlabel(r'$t~/~{\rm d}$', fontsize=25 )
	plt.ylabel(r'$r~/~{\rm R_{\odot}}$', fontsize=25 )
	#plt.axis([15.,60.,0.,30.])
	# plt.hlines( 3.75 + 11.981 , 0., 1000. ) # m70soft4 initial
	# plt.hlines( 3.75 + 6.253 , 0., 1000. ) # massive before change?
	# plt.hlines( 2.0 + 1.0 , 0., 1000. ) # massive after change
	# plt.hlines( 3.1265 + 0.36, 0., 1000. ) # ccaceeccp
	# plt.hlines( 1.9935 + 0.99, 0., 1000. ) # paper
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'sep.pdf')

	# plt.axis([0.,240.,2.,53.])
	plt.yscale('log')
	savePlot(fig,'seplog.pdf')
	plt.clf()

	if (nplots==1):
		fig = plt.figure()
		plt.plot( posPrimx[0], posPrimy[0], c='g', lw=2, label='Primary' )
		plt.plot( posCompx[0], posCompy[0], c='b', lw=2, label='Companion' )
		plt.plot( posCMx[0], posCMy[0], c='r', lw=2, label='CM' )
		plt.legend()
		plt.xlabel('x (cm)', fontsize=20)
		plt.ylabel('y (cm)', fontsize=20)
		plt.xticks( fontsize=20)
		plt.yticks( fontsize=20)
		plt.grid(True)
		plt.tight_layout()
		savePlot(fig,'CMpos.pdf')
		plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], velCMnorm[i], lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$',fontsize=20 )
	plt.ylabel(r'$v_{\rm CM}~/~{\rm km~s^{-1}}$',fontsize=20)
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,240.,0.,8.])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'CMvel.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], velCMDMnorm[i], lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$',fontsize=20 )
	plt.ylabel('DM CM Velocity (km/s)',fontsize=20)
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,240.,0.,16.])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'CMDMvel.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		plt.plot( time[i], numorbits[i], lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend()
	plt.xlabel(r'$t~/~{\rm d}$',fontsize=20 )
	plt.ylabel('Number of Completed Orbits',fontsize=20)
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	# plt.axis([0.,240.,0.,16.])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'numorbits.pdf')
	plt.clf()

	fig = plt.figure()
	for i in range(0,nplots):
		Time = time[i]
		BoolArray = boolArray[i]
		Ecc = ecc[i]
		plt.plot( Time[BoolArray], Ecc[BoolArray], lw=2, label=labels[i] )
	if nplots > 1 :
		plt.legend(prop={'size':15})
	plt.xlabel(r'$t~/~{\rm d}$',fontsize=20 )
	plt.ylabel(r'${\rm Eccentricity}$',fontsize=20)
	plt.xticks( fontsize=20)
	plt.yticks( fontsize=20)
	#plt.axis([15.,60.,0.23,0.52])
	plt.grid(True)
	plt.tight_layout()
	savePlot(fig,'ecc.pdf')
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
    numorbits = np.zeros(nframes)

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

    numorbits[0] = 0
    for j in range(1,nframes):
        if (is_peri[j] == True):
            numorbits[j] = numorbits[j-1] + 1
        else :
            numorbits[j] = numorbits[j-1]

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
    return a, ecc, boolArray, numorbits

paths, labels = getPaths(nplots,args.py2)
setnums, time, posCMx, posCMy, posCMz, vCMx, vCMy, vCMz, fracunbound, fracunbound_i, \
sep, velCMnorm, posPrimx, posPrimy, posPrimz, posCompx, posCompy, \
posCompz, massGasTot, ejeceff, ejeceff_i, ietot, ie_idealtot, gasKEtot, DMKEtot, velCMDMnorm, fracunbound_noIe, ejeceff_noIe, \
gasKEunbound, gasKEbound, gasIEunbound, gasIEbound, PECoreGasUnboundPrim, PECoreGasBoundPrim, PECoreGasUnboundComp, PECoreGasBoundComp, PECoreCore, PEGasGasUnbound, PEGasGasBound, \
Emech, gasPbound, gasPunbound, gasPxbound, gasPybound, gasPzbound, gasPxunbound, gasPyunbound, gasPzunbound, gasLbound, gasLunbound, gasLxbound, gasLybound, gasLzbound, gasLxunbound, gasLyunbound, gasLzunbound, \
corePx, corePy, corePz, compPx, compPy, compPz, coreLx, coreLy, coreLz, compLx, compLy, compLz, \
reflectiveCount, edgeCount, mirrorLeft, mirrorRight, mirrorRadius, mirrorCenterX, mirrorCenterY, mirrorCenterZ, \
mirrorMass, mirrorVelX, mirrorVelY, mirrorVelZ, mirrorForceX, mirrorForceY, mirrorForceZ, mirrorGravX, mirrorGravY, mirrorGravZ, \
mirrorGravCorrX, mirrorGravCorrY, mirrorGravCorrZ, dynFric, dynFricV, dynFricNoCorr, \
mPrim, mComp, gravPrimGasX, gravPrimGasY, gravPrimGasZ = collectData(nplots,paths)

if movingBC :
	plotMirrorForces( time, mirrorMass, mirrorRadius, mirrorForceX, mirrorForceY, mirrorForceZ, mirrorGravX, mirrorGravY, mirrorGravZ, mirrorGravCorrX, mirrorGravCorrY, mirrorGravCorrZ, dynFric, dynFricV, dynFricNoCorr, sep, nplots, labels )
	plotL( time, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz, mPrim, mComp, corePx, corePy, corePz, compPx, compPy, compPz, nplots, labels )
	if nplots == 1 :
		plotTorques( time, sep, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz, mPrim, mComp, mirrorForceX, mirrorForceY, mirrorForceZ, mirrorGravX, mirrorGravY, mirrorGravZ, mirrorGravCorrX, mirrorGravCorrY, mirrorGravCorrZ, gravPrimGasX, gravPrimGasY, gravPrimGasZ, dynFric, dynFricV, dynFricNoCorr, corePx, corePy, corePz, compPx, compPy, compPz, nplots, labels )
if args.unbound :
	plotUnbound( time, fracunbound, ejeceff, fracunbound_noIe, ejeceff_noIe, nplots, labels )
	plotUnbound_i( time, fracunbound_i, ejeceff_i, nplots, labels )
	plotMomentum( time, Emech, gasPbound, gasPunbound, gasPxbound, gasPybound, gasPzbound, gasPxunbound, gasPyunbound, gasPzunbound, gasLbound, gasLunbound, gasLxbound, gasLybound, gasLzbound, gasLxunbound, gasLyunbound, gasLzunbound, corePx, corePy, corePz, compPx, compPy, compPz, coreLx, coreLy, coreLz, compLx, compLy, compLz, nplots, labels )
	plotEnergy( time, ietot, ie_idealtot, Emech, nplots, labels )
if args.mass :
    plotMass( time, massGasTot, nplots, labels )
# if args.energy :
# 	plotEnergy( time, ietot, ie_idealtot, gasKEtot, gasPEtot, DMKEtot, DMPEtot, nplots, labels )
if args.orbel :
	a = []
	ecc = []
	boolArray = []
	numorbits = []
	for i in range(0,nplots):
		aN, eccN, boolArrayN, numorbitsN = findAE( sep[i] )
		a.append(aN)
		ecc.append(eccN)
		boolArray.append(boolArrayN)
		numorbits.append(numorbitsN)

	plotOrbEl( nplots, labels, time, sep, a, ecc, boolArray, velCMnorm, posCMx, \
	posCMy, posCMz, posPrimx, posPrimy, posPrimz, posCompx, posCompy, posCompz, velCMDMnorm, numorbits )
	plotSmoothSep( nplots, labels, time, sep )
