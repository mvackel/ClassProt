
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import py2neo as pn
from py2neo import *
import NGraph as ng
import time


if __name__ == "__main__":

	lDodec = [(1,2), (1,10), (1,11), (2,3),(2,12), (3,4), (3,13), (4,5), (4,14),
			  (5,6), (5,15),(6,7), (6,16),(7,8), (7,17),(8,9),(8,18),(9,10),(9,19),
			  (10,20),(11,13),(11,19),(12,14),(12,20),(13,15),(14,16),(15,17),(16,18),
			  (17,19),(18,20)]
	ngDodec = ng.NGraph(weight=None)
	ngDodec.add_edges_from(lDodec)
	# nx.draw(mgDodec, with_labels=True, node_color='y')
	# plt.show()
	
	
	lDesargues = [(1,2),(1,10),(1,11),(2,3),(2,12),(3,4),(3,13),(4,5),(4,14),(5,6),
				  (5,15),(6,7),(6,16),(7,8),(7,17),(8,9),(8,18),(9,10),(9,19),
				  (10,20),(11,14),(11,18),(12,15),(12,19),(13,16),(13,20),(14,17),
				  (15,18),(16,19),(17,20)]
	
	ngDesarg = ng.NGraph(weight=None)
	ngDesarg.add_edges_from(lDesargues)
	# nx.draw(mgDesarg, with_labels=True, node_color='c')
	# plt.show()
	
	k=2
	gn = ngDodec.graphNumber(k=k)
	print(f'Dodec  Graph Number (k={k}): ', gn)
	print(f'Desarg Graph Number (k={k}): ', ngDesarg.graphNumber(k=k))

	print(f'Dodec  Homogeneous (k={k}): ', ngDodec.JensenShannonHomogeneous(k=2))
	print(f'Desarg Homogeneous (k={k}): ', ngDesarg.JensenShannonHomogeneous(k=2))
	
	

	

	
	