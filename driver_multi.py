import numpy as np
from multiprocessing import Process

from config.rg_run1_config import *

def rp_mult():
	import radprof_mult
def tp_mult():
	import tempprof_mult
def ct():
	import coretemp
def rp():
	import radprof
def tp():
	import tempprof
def dens():
	import densanim
def densx():
	densanim_direction = 'x'
	import densanim
def densy():
	densanim_direction = 'y'
	import densanim
def densz():
	densanim_direction = 'z'
	import densanim
def ps():
	import partslice

if do_comparison:

	nproc = do_radprof + do_tempprof + do_densanim * 3
	if (nproc > maxproc):
		print '\n Terminating: too many processes \n'
		import sys
		sys.exit(0)

	if do_radprof:
		print '\nStarting Radial Density Profile '+str(nrows)+' x '+str(ncolumns)+' ( '+comparison_name+' )\n'
		p_rpm = Process(target = rp_mult)
		p_rpm.start()
		
	if do_tempprof:
		print '\nStarting Radial Temperature Profile '+str(nrows)+' x '+str(ncolumns)+' ( '+comparison_name+' )\n'
		p_tpm = Process(target = tp_mult)
		p_tpm.start()

	if do_densanim:
		print '\nStarting Density Projection (All Directions) ( ' + simname + ' )\n'

		densanim_direction = 'x'
		p_dx = Process(target = densx)
		p_dx.start()

		densanim_direction = 'y'
		p_dy = Process(target = densy)
		p_dy.start()

		densanim_direction = 'z'
		p_dz = Process(target = densz)
		p_dz.start()
		
	if do_radprof:
		p_rpm.join()
	if do_tempprof:
		p_tpm.join()
	if do_densanim:
		p_dx.join()
		p_dy.join()
		p_dz.join()
	
else:
	
	nproc = do_coretemp + do_radprof + do_tempprof + do_densanim + do_partslice
	if (nproc > maxproc):
		print '\n Terminating: too many processes \n'
		import sys
		sys.exit(0)
	
	if do_coretemp:
		print '\nStarting Core Temperature ( ' + simname + ' )\n'
		p_ct = Process(target = ct)
		p_ct.start()

	if do_radprof:
		print '\nStarting Radial Density Profile ( ' + simname + ' )\n'
		p_rp = Process(target = rp)
		p_rp.start()
	
	if do_tempprof:
		print '\nStarting Radial Temperature Profile ( ' + simname + ' )\n'
		p_tp = Process(target = tp)
		p_tp.start()
	
	if do_densanim:
		print '\nStarting ' + densanim_direction + ' Density Projection ( ' + simname + ' )\n'
		p_d = Process(target = dens)
		p_d.start()
	
	if do_partslice:
		print '\nStarting '+partslice_direction+' '+partslice_parttype+' Particle Slice ( '+simname+' )\n'
		p_ps = Process(target = ps)
		p_ps.start()
		
	if do_coretemp:
		p_ct.join()
	if do_radprof:
		p_rp.join()
	if do_tempprof:
		p_tp.join()
	if do_densanim:
		p_d.join()
	if do_partslice:
		p_ps.join()