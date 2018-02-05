
import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from ClassProtGraph import *
import py2neo as pn
from py2neo import *
import networkx as nx
import numpy as np
import time

if __name__ == "__main__":
	
	bGNumberPdb = True
	if bGNumberPdb:
		pn.authenticate("localhost:7474", "neo4j", "pxyon123")
		dbGraph = pn.Graph()
		
		# lstSym = ['5O75', '2JKU', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA', '1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K', '3U3M', '3U3O', '3U3R', '4GRA']
		# lstSym = ['2D06', '3QVU', '3QVV', '3U3J', '3U3K']
		lstSym = ['5O75', '3U3K']    # 5O75: 71 CAs,    3U3K:  289 CAs
		lstSym = ['3U3K']    # 289 CAs - 161.7 segs (cutoff 6.0)
		lstSym = ['5O75']    # 5O75: 71 CAs   1 seg (cutoff 6.0)
		lstSym = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW', '3PC7', '1BDO', '2BDO', '3BDO', '2KCC']  # 35 to 84 CAs
		#lstSym = ['2JKU', '1HF9', '1GMJ', '5O75', '4FIW']  # 35 to 76 CAs
		lstSym = ['2JKU']    #, '1HF9'] #, '1GMJ', '5O75', '4FIW']  # 35 to 76 CAs

		print(f'Num Prots: {len(lstSym)}')
		
		start = time.time()
		lstProts = [ClassProtGraph(dbGraph, sym) for sym in lstSym]
		for i in range(len(lstSym)):
			print(f'{lstProts[i].idPdb}: {lstProts[i].number_of_nodes()}', end='  ')
		print(' ')
		
		lstProts = [_graph.graphCuttoff(6.0) for _graph in lstProts]
		
		print(f'Mount duration (serial): {time.time() - start}')
		start = time.time()
		
		k = 3
		for mp in lstProts:
			print(f'IdPdb: {mp.idPdb}   graphNumber = {mp.graphNumber(k=k)}')
		dur = time.time() - start
		print(f'Matrix duration (serial): {dur} -- Average: {dur/len(lstSym)} / protein')

	stream_test = False
	if stream_test:
		lstProts[0].streamToGephi(port=9009)
	
	_3dPlot_test = True
	if _3dPlot_test:
		
		len = len(lstProts)
		xs = np.zeros(len)
		ys = np.zeros(len)
		zs = np.zeros(len)
		for i, prot in enumerate(lstProts):
			xs[i] = prot.graphNumber(k=3)[0]
			ys[i] = prot.graphNumber(k=3)[1]
			zs[i] = prot.graphNumber(k=3)[2]

		def onpick3d(event):
			ind = event.ind
			idx = ind[0]
			print('Protein: ', lstProts[idx].idPdb, lstProts[idx].number_of_nodes(),
				               lstProts[idx].ec, lstProts[idx].taxid )

		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		ax.scatter(xs, ys, zs, marker='^', picker=True)
		ax.set_xlabel('GNumber[0]')
		ax.set_ylabel('GNumber[1]')
		ax.set_zlabel('GNumber[2]')
		fig.canvas.mpl_connect('pick_event', onpick3d)
		plt.show()

	
		
	
		
# print('normal:')
# start = time.time()
# for i in range(10):
# 	m = gDodec.buildNodeDistMatrixByLine()
# print(f'Dur: {time.time()-start}')
# print(m[:2])
# print('-- k:')
# start = time.time()
# for i in range(10):
# 	m = gDodec.buildNodeDistMatrixKByLine()
# print(f'Dur: {time.time()-start}')
# print(m[:2])
