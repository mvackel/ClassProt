import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import py2neo as pn
from py2neo import *
import ClassProtGraph as mg
import time

if __name__ == "__main__":

	# ========== dbGraph:
	pn.authenticate("localhost:7474", "neo4j", "pxyon123")
	dbGraph = pn.Graph()
	
	pdbSym1 = '2JKU'    # 35 CAs
	#pdbSym1 = '1HF9'    # 41 CAs
	#pdbSym1 = '1GMJ'    # 65 CAs
	#pdbSym1 = '3PC7'  # 80 CAs     - gn: 7.4 seg
	pdbSym1 = '4RT5'    # 100 CAs    - gn: 14 seg
	#pdbSym1 = '1FM8'    # 212 CAs   - gn: 106 seg
	
	gProt1 = mg.ClassProtGraph(dbGraph, pdbSym1)
	
	numNodes = gProt1.number_of_nodes()
	print('Number of Nodes: ', gProt1.number_of_nodes())
	
	# nx.draw(gProt1, with_labels=True, node_color='c')
	# plt.show()
	
	k=2
	
	if numNodes < 81:
		print('Calculating - serial...')
		startTime = time.time()
		smats, sweights, sdiameters = gProt1.buildNodeDistMatricesToK(k=k)
		print(f'Elapsed (ser): {time.time() - startTime}')
		
	
		print('Calculating - parallel...')
		startTime = time.time()
		pmats, pweights, pdiameters = gProt1.parBuildNodeDistMatricesToK(k=k)
		print(f'Elapsed (par): {time.time() - startTime}')
	
		print('')
		print('Serial weights  :', sweights)
		print('Parallel weights:', pweights)
		print('')
		print('Serial diameters  :', sdiameters)
		print('Parallel diameters:', pdiameters)
		print('')
		# print('Serial mat[k=1:')
		# print(smats[0])
		# print('')
		# print('Parallel mat[k=1]:')
		# print(pmats[0])
		# print('')
		# print('Serial mat[k=2:')
		# print(smats[1])
		# print('')
		# print('Parallel mat[k=2]:')
		# print(pmats[1])

	print('Calculating Graph Number...')
	startTime = time.time()
	gn = gProt1.graphNumber(k=k)
	print(f'Elapsed: {time.time() - startTime}')
	print(f'Graph Number (k={k}): ', gn)



