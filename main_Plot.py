import sys
import requests
import urllib
from urllib import parse
import os
from PdbToGraphDB import *

from PDBActors import *
from typing import List

from MemGraph import *
from math import sqrt, log
from joblib import Parallel, delayed
import itertools

dissTest = np.ndarray((5,5))
def setDissTest(x,y):
	dissTest[x][y] = dissTest[y][x] = x + y


class PlotHeatMap:
	def __init__(self):
		self._lstSym:List[str]=[]
		self._lstProtObj:List[MemGraph] = []
		self._dissimMatrix = np.ndarray((1,1))
	
	def calcXYDiss(self, ):
		pass
	
	def setDissValues(self, x:int, y:int):
		diss_G1G2 = self._lstProtObj[x].RavettiDissimilarity(self._lstProtObj[y])
		self._dissimMatrix[x][y] = self._dissimMatrix[y][x] = diss_G1G2
	
	def run(self, dbGraph:Graph, lstPdbEntries: List[str], standardScale=True) -> int:
		print(f'Number of PDB entries to plot: {len(lstPdbEntries)}')
		if len(lstPdbEntries) == 0:
			print('Exiting...')
			return 3
		start0 = time.time()
		for sym in lstPdbEntries:
			prot = MemGraph()
			if prot.fromGraphDB(dbGraph, sym):
				self._lstSym.append(sym)
				self._lstProtObj.append(prot)
		dim = len(self._lstSym)    # final dissimilarity matrix will be dim X dim
		
		self._dissimMatrix = np.zeros((dim, dim))
		print(f'Duration list of protein graphs: {time.time()-start0}')
		start1 = time.time()
		# log: list(itertools.combinations([x for x in range(4)],2))
		# calc all Dissimilarities in parallel:
		# they will be stored inside of the object properties _JSD and _NND
		#posX, posY, D_G1G2 = \
		Parallel(n_jobs=8) (#, backend="threading")(
			    				delayed(self.setDissValues)(x,y)
				          		for x,y in list(itertools.combinations([i for i in range(dim)],2))
							)
		# for posX in range(0, dim-1):   #does not include the last one in the list
		# 	prot = self._lstProtObj[posX]
		# 	jsd1, nnd1 = prot.JensenShannonDiverg()
		#
			
			# for posY in range(posX+1, dim):
			# 	if posX != posY:
			# 		other = lstProts[posY]
			# 		# jsd2, nnd2 = other.JensenShannonDiverg()
			# 		# jsdG1G2 = prot.JensenShannon2Graphs(other)
			# 		# D_G1G2 = 0.5 * np.math.sqrt(jsdG1G2/np.math.log(2)) + \
			# 		# 		 0.5 * np.math.fabs(np.math.sqrt(nnd1)-np.math.sqrt(nnd2))
			# 		D_G1G2 = prot.RavettiDissimilarity(other)
			# 		dissimMatrix[posX, posY] = dissimMatrix[posY, posX] = D_G1G2
		end = time.time()
		print(f'Dissimilarity Matrix: {end-start1} seconds')
		
		
		# PLOT DISSIMILARITY MATRIX:
		fig = plt.figure(1)
		ax = plt.gca()
		#c = ax.pcolorfast(dissimMatrix, cmap='seismic') #, vmin=z_min, vmax=z_max)
		if standardScale:
			c = ax.pcolorfast(self._dissimMatrix, cmap='Reds', vmin=0.0, vmax=1.0) #, vmin=z_min, vmax=z_max)
		else:
			c = ax.pcolorfast(self._dissimMatrix, cmap='Reds') #, vmin=z_min, vmax=z_max)

		ax.set_title('Dissimilarity Color Map')
		cb = fig.colorbar(c, ax=ax)
		cb.set_label('Dissimilarity')
		
		fig.tight_layout()
		plt.show()


if __name__ == "__main__":

	pn.authenticate("localhost:7474","neo4j","pxyon123")
	dbGraph = pn.Graph()

	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU']
	#lstPdbEntries = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']
	lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6']

	PlotHeatMap().run(dbGraph, lstPdbEntries, False)
