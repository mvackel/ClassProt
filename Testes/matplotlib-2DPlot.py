from matplotlib import pyplot as plt
from matplotlib import cm as CM
from matplotlib import mlab as ml
import numpy as np

def example1():
	a = np.random.random((16, 16))
	plt.imshow(a, cmap='hot', interpolation='nearest')
	
def example2():
	a = np.random.normal(0.0, 0.5, size=(5000, 10)) ** 2
	a = a / np.sum(a, axis=1)[:, None]  # Normalize
	plt.pcolor(a)

	
def example3():
	a = np.random.normal(0.0, 0.5, size=(5000, 10)) ** 2
	maxvi = np.argsort(a, axis=1)
	ii = np.argsort(maxvi[:, -1])
	plt.pcolor(a[ii, :])


def example4():
	x = np.random.random(50)
	y = np.random.random(50)
	z = np.random.random(50)
	gridsize = 30
	plt.hexbin(x, y, C=z)  # <- You need to do the hexbin plot
	cb = plt.colorbar()
	cb.set_label('density')
	
import matplotlib.mlab as mlab
def example5():
	# Fixing random state for reproducibility
	np.random.seed(19680801)
	
	n = 100000
	x = np.random.standard_normal(n)
	y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
	xmin = x.min()
	xmax = x.max()
	ymin = y.min()
	ymax = y.max()
	
	fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
	fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
	ax = axs[0]
	hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')
	ax.axis([xmin, xmax, ymin, ymax])
	ax.set_title("Hexagon binning")
	cb = fig.colorbar(hb, ax=ax)
	cb.set_label('counts')
	
	ax = axs[1]
	hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
	ax.axis([xmin, xmax, ymin, ymax])
	ax.set_title("With a log color scale")
	cb = fig.colorbar(hb, ax=ax)
	cb.set_label('log10(N)')
	
def example6():
	# Comparing pcolor with similar functions
	# Demonstrates similarities between pcolor, pcolormesh, imshow and pcolorfast
	#   for drawing quadrilateral grids.
	# make these smaller to increase the resolution
	dx, dy = 0.15, 0.05
	
	# generate 2 2d grids for the x & y bounds
	y, x = np.mgrid[slice(-3, 3 + dy, dy),
					slice(-3, 3 + dx, dx)]
	z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
	# x and y are bounds, so z should be the value *inside* those bounds.
	# Therefore, remove the last value from the z array.
	z = z[:-1, :-1]
	z_min, z_max = -np.abs(z).max(), np.abs(z).max()
	
	fig, axs = plt.subplots(2, 2)
	
	ax = axs[0, 0]
	c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
	ax.set_title('pcolor')
	# pcolor can be very slow for large arrays; consider using the similar but much faster pcolormesh() instead.
	# set the limits of the plot to the limits of the data
	ax.axis([x.min(), x.max(), y.min(), y.max()])
	fig.colorbar(c, ax=ax)
	
	ax = axs[0, 1]
	'''
	pcolormesh is similar to pcolor(), but uses a different mechanism and returns
	a different object; pcolor returns a PolyCollection but pcolormesh returns a QuadMesh.
	It is much faster, so it is almost always preferred for large arrays.
	'''
	c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
	ax.set_title('pcolormesh')
	# set the limits of the plot to the limits of the data
	ax.axis([x.min(), x.max(), y.min(), y.max()])
	fig.colorbar(c, ax=ax)
	
	
	ax = axs[1, 0]
	c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
				  extent=[x.min(), x.max(), y.min(), y.max()],
				  interpolation='nearest', origin='lower')
	ax.set_title('image (nearest)')
	fig.colorbar(c, ax=ax)
	
	
	ax = axs[1, 1]
	'''
	pseudocolor plot of a 2-D array

	Experimental; this is a pcolor-type method that provides the fastest possible
	rendering with the Agg backend, and that can handle any quadrilateral grid.
	It supports only flat shading (no outlines), it lacks support for log scaling of
	the axes, and it does not have a pyplot wrapper.
	'''
	c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
	ax.set_title('pcolorfast')
	fig.colorbar(c, ax=ax)
	
	fig.tight_layout()

if __name__ == "__main__":

	# plt.figure(1)
	# example1()
	# plt.figure(2)
	# example2()
	# plt.figure(3)
	# example3()
	# plt.figure(4)
	# example4()
	#example5()
	example6()
	
	plt.show()

	
	