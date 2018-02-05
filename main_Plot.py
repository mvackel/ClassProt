import sys
import requests
import urllib
from urllib import parse
import os
from PdbToGraphDB import *
import itertools as it

from PDBActors import *
from typing import List

from ClassProtGraph import *
from math import sqrt, log
from joblib import Parallel, delayed, parallel_backend
import itertools
import multiprocessing as mp

dissTest = np.ndarray((5,5))
def setDissTest(x,y):
	dissTest[x][y] = dissTest[y][x] = x + y


class PlotHeatMap:
	def __init__(self):
		self._lstSym:List[str]=[]
		self._lstProtObj:List[ClassProtGraph] = []
		self._lstLabels = []
		self._dissimMatrix = np.ndarray((1,1))
	
	def calcXYDiss(self, ):
		pass
	
	def calcDissValues(self, x:int, y:int) -> (int, int, float):
		p1Id = self._lstProtObj[x].idPdb
		p2Id = self._lstProtObj[y].idPdb
		diss_G1G2 = self._lstProtObj[x].RavettiDissimilarity(self._lstProtObj[y])
		print(f'{p1Id}--{p2Id}   Diss: {diss_G1G2:.8f}')
		#self._dissimMatrix[x][y] = self._dissimMatrix[y][x] = diss_G1G2
		#print(self._dissimMatrix)
		return (x, y, diss_G1G2)
	
	def calcJS(self, numItem:int, prot:ClassProtGraph, k=1) -> ClassProtGraph:     #(int, ClassProtGraph):
		# Helper function for parallel tasks
		prot.JensenShannonDiverg(k)
		# Debug:
		print(f'{numItem}. {prot.idPdb}, ', end='', flush=True)
		#return (numItem, prot)
		# Need to return the protein object because of the serializaxion/deserialization process
		return prot

	def run(self, dbGraph:Graph, lstPdbEntries: List[str], k:int=1, standardScale=True ) -> int:
		print(f'Number of PDB entries to plot: {len(lstPdbEntries)}')
		if len(lstPdbEntries) == 0:
			print('Exiting...')
			return 3
		
		lProts = []
		start0 = time.time()
		for sym in lstPdbEntries:
			prot = ClassProtGraph()
			if prot.fromGraphDB(dbGraph, sym):
				if prot.number_of_nodes() > 0:
					lProts.append(prot)
				else:
					print(f'{sym}: 0 -- Not Found.')
		
		# Sort the protein list by EC (first assumption):
		lProts.sort(key=lambda p: p.ec)
				
		for cnt, prot in enumerate(lProts):
			sym = prot.idPdb
			numCas = prot.number_of_nodes()
			ec = prot.ec
			prot.userData = cnt		# to be used for ordering (when parallel processing)
			print(f'{cnt: 5d}. {sym}: {numCas: 4d}  {ec}')
			#self._lstSym.append(sym)
			self._lstProtObj.append(prot)
			#self._lstLabels.append(f'{numCas}={ec}={sym}')
					
		dim = len(self._lstProtObj)    # final dissimilarity matrix will be dim X dim
		
		self._dissimMatrix = np.zeros((dim, dim))
		print(f'Duration list of protein graphs: {time.time()-start0:.3f}')
		
		numOfProcessors = mp.cpu_count()
		print(f'CPUs: {numOfProcessors}')
		print('Calculated:')
		
		start1 = time.time()
		# calc all Dissimilarities in parallel:
		# they will be stored inside of the object properties _JSD and _NND
	
		
		newlProts = []
		# -- impl 1:
		# with Parallel(n_jobs=-1, backend='multiprocessing', verbose=40) as parallel:
		# 	for idx, prot in enumerate(lProts):
		# 		newlProts = parallel(delayed(self.calcJS)(idx, prot))
		# newlProts = Parallel(n_jobs=-1, backend='multiprocessing', verbose=0)(
		# 	delayed(self.calcJS)(idx, prot) for idx, prot in enumerate(lProts)
		# )
		# -- impl 2:
		# newlProts = []
		# numProts = len(lProts)
		#
		# with Parallel(n_jobs=-1, backend='multiprocessing', verbose=0) as parallel:
		# 	idx  = 0
		# 	while idx < numProts: #, prot in enumerate(lProts):
		# 		stop = idx + numOfProcessors
		# 		stop = stop if stop <= numProts else numProts
		# 		prots = parallel(delayed(self.calcJS)(i, self._lstProtObj[i], k)
		# 						 for i in range(idx, stop))
		# 		idx += stop
		# 		newlProts.extend(prots)
		# 		print('' if idx < numProts else '.', flush=True)
		#
		# newlProts.sort(key=lambda prot: prot.userData )
		# self._lstProtObj = newlProts
		#
		# # Parallel(n_jobs=-1, backend='multiprocessing', verbose=40)(
		# # 	delayed(self.calcJS)(prot) for prot in lProts
		# # )
		# #Parallel(n_jobs=8) (#, backend="threading")(
		# lResult = Parallel(n_jobs=-1) (#, backend=‘multiprocessing’)(
		# 	    				delayed(self.calcDissValues)(x, y)
		# 		          		for x,y in list(itertools.combinations([i for i in range(dim)],2))
		# 					)
		# # for posX in range(0, dim-1):   #does not include the last one in the list
		# # 	prot = self._lstProtObj[posX]
		# # 	jsd1, nnd1 = prot.JensenShannonDiverg()
		# #
		#
		# 	# for posY in range(posX+1, dim):
		# 	# 	if posX != posY:
		# 	# 		other = lstProts[posY]
		# 	# 		# jsd2, nnd2 = other.JensenShannonDiverg()
		# 	# 		# jsdG1G2 = prot.JensenShannon2Graphs(other)
		# 	# 		# D_G1G2 = 0.5 * np.math.sqrt(jsdG1G2/np.math.log(2)) + \
		# 	# 		# 		 0.5 * np.math.fabs(np.math.sqrt(nnd1)-np.math.sqrt(nnd2))
		# 	# 		D_G1G2 = prot.RavettiDissimilarity(other)
		# 	# 		dissimMatrix[posX, posY] = dissimMatrix[posY, posX] = D_G1G2

		# -- impl 3 (serial - NGraph implements parallelization:
		lResult = []
		for x, y in list(itertools.combinations([i for i in range(dim)], 2)):
			lResult.append(self.calcDissValues(x,y))
			
		end = time.time()
		
		# Set the dissimilarity matrix values:
		for res in lResult:
			x = res[0]
			y = res[1]
			diss_G1G2 = res[2]
			self._dissimMatrix[x][y] = self._dissimMatrix[y][x] = diss_G1G2
		#print(self._dissimMatrix)
		print(f'Time to Dissimilarity Matrix: {end-start1:.3f} seconds')

		
		self.plot('Dissimilarity Heat Map - by EC', standardScale, 'Reds')
		
		
	def plot(self, title:str, standardScale:bool, colorMap:str='Reds'):
		import matplotlib.ticker as ticker
		# PLOT DISSIMILARITY MATRIX:
		fig = plt.figure()
		ax = plt.gca()
		#c = ax.pcolorfast(dissimMatrix, cmap='seismic') #, vmin=z_min, vmax=z_max)
		if standardScale:
			c = ax.pcolorfast(self._dissimMatrix, cmap=colorMap, vmin=0.0, vmax=1.0) #, vmin=z_min, vmax=z_max)
		else:
			c = ax.pcolorfast(self._dissimMatrix, cmap=colorMap) #, vmin=z_min, vmax=z_max)

		numProt = len(self._lstProtObj)
		plt.locator_params(nbins=2*numProt)
		labels = [ f'{p.number_of_nodes()}={p.ec}={p.idPdb}' for p in self._lstProtObj]
		# labels = [''] * numProt
		# labels = list(zip(labels, self._lstLabels))
		# labels = [i for sl in labels for i in sl]
		ax.xaxis.set_major_locator(ticker.IndexLocator(base=1.0, offset = 0.5))
		ax.yaxis.set_major_locator(ticker.IndexLocator(base=1.0, offset = 0.5))

		ax.set_xticklabels(labels, rotation='vertical')
		ax.set_yticklabels(labels)
		ax.set_title(title)
		cb = fig.colorbar(c, ax=ax)
		cb.set_label('Dissimilarity')
		
		fig.tight_layout()
		plt.show()


if __name__ == "__main__":
	
	# Combinations: nCr = n! / ( r! * (n-r)! )

	pn.authenticate("localhost:7474","neo4j","pxyon123")
	dbGraph = pn.Graph()

	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU']
	#lstPdbEntries = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6', '1Z28']
	#lstPdbEntries = ['5O75', '2JKU', '1LS6']
	
	# fewer CAs:
	lstPdbEntries = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW', '3PC7', '1BDO', '2BDO', '3BDO', '2KCC',
					 '1A6X', '1IN1', '1IMO', '5B0B', '5B09', '4HR9', '5B0A', '2EJM', '2DN8', '5B0D',
					 '4RT5', '5B0G', '5B0C', '5B0F', '5B08', '5B0E', '4MZC', '4HJM', '4MZB', '4N10',
					 '4N0Z', '3C1S', '3CTF', '2CQY', '4N11', '3CTG', '3C1R']

	lstPdbEntries = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW', '3PC7', '1BDO', '2BDO', '3BDO', '2KCC',
					 '1A6X', '1IN1', '1IMO', '5B0B', '5B09', '4HR9', '5B0A', '2EJM', '2DN8', '5B0D',
					 '4RT5', '5B0G', '5B0C', '5B0F', '5B08', '5B0E', '4MZC', '4HJM', '4MZB', '4N10']

	lstPdbEntries = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW', '3PC7', '1BDO', '2BDO', '3BDO', '2KCC',
					 '1A6X', '1IN1', '1IMO', '5B0B', '5B09', '4HR9', '5B0A', '2EJM', '2DN8', '5B0D']   # 35 to 100 CAs
	
	lstPdbEntries = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW', '3PC7', '1BDO', '2BDO', '3BDO', '2KCC']   # 35 to 84 CAs
	#lstPdbEntries = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW', '1BDO', '2BDO']
	#lstPdbEntries = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW']
	#lstPdbEntries = ['2JKU', '1HF9', '1GMJ']
	#lstPdbEntries = ['2JKU', '1HF9']


	PlotHeatMap().run(dbGraph, lstPdbEntries, k=1, standardScale=False)
