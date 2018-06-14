from __main__ import *
import yt
import matplotlib.pyplot as pl
import matplotlib.animation as animation

if densanim_fixlimits:
	sizingappend = ''
else:
	sizingappend = '_sizing'

# create figure
ts = yt.load( readpath + 'star.out.00000' + str(startingset) )
plot = yt.ProjectionPlot(ts, densanim_direction, ('gas', 'density'), width = densanim_plotwidth )

if (densanim_plotwidth > 1.0e12):
	timelabel = 'day'
else:
	timelabel = 'hr'

plot.annotate_timestamp(time_unit = timelabel)
# plot.set_buff_size([2000,2000])

if densanim_fixlimits:
	plot.set_zlim('all',densanim_lowlim,densanim_highlim)
	
# plot.set_axes_unit('unitary')
fig = plot.plots['density'].figure

# create frames
def animate(i):
	num = i*frameskip + 1000000 + startingset
	numstr = str(num)
	cut = numstr[1:7]
	print 'densanim: ' + simname + ' Frame ' + str(i) + ' Data Set ' + cut
	
	ds = yt.load(readpath + 'star.out.' + cut)
	
	plot._switch_ds(ds)
	
# create animation object
anim = animation.FuncAnimation(fig, animate, frames = nframes, interval = period, repeat = False)
densanim_saveas = writepath + densanim_direction + '_dens_' + simname + sizingappend + '.mp4'
anim.save(densanim_saveas)
print 'densanim: Saved animation ' + writepath + densanim_direction + '_dens_' + simname + sizingappend + '.mp4'
