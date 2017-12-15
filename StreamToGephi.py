import json
import requests

import networkx

class JsonClient(object):
	def __init__(self, url='http://127.0.0.1:8090/workspace1'):
		self.url = url
		
	def __send(self, data):
		#conn = urllib3.urlopen(self.url+ '?operation=updateGraph', data)
		#return conn.read()
		headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
		#r = requests.post(self.url + '?operation=updateGraph', data=data, headers=headers)
		r = requests.post(self.url + '?operation=updateGraph', data=data)
		return r.text
		# url = "http://localhost:8080"
		# data = {'sender': 'Alice', 'receiver': 'Bob', 'message': 'We did it!'}
		# headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
		# r = requests.post(url, data=json.dumps(data), headers=headers)
	
	
	
	def add_node(self, id, **attributes):
		self.__send( json.dumps( {"dn":{id:{}}} ) )
		
	def add_edge(self, id, source, target, directed=False, **attributes):
		attributes['source'] = source
		attributes['target'] = target
		attributes['directed'] = directed
		self.__send(data = json.dumps( {"ae":{id:attributes}} ) )
		
	def clean(self):
		self.__send( json.dumps( {"dn":{"filter":"ALL"}} ) )
		
	def delete_edge(self, id):
		self.__send( json.dumps( {"de":{id:{}}} ) )
		

import time

if __name__ == "__main__":
	g = JsonClient('http://localhost:8090/workspace1')
	#g.clear()
	n = 1000
	node_att = {'size':10, 'r':1.0, 'g':0.0, 'b':0.0, 'x':1 }
	
	for i in range(0,n):
		gg = networkx.barabasi_albert_graph(100,20, seed=i)
		for n in gg:
			g.add_node(str(n), **node_att)
			g.add_node(str(n))

		for n,nbrsdict in    gg.adjacency():
			for nbr,eattr in nbrsdict.items():
				g.add_edge(str(n), str(n), str(nbr))
		time.sleep(10)
		g.clean()
		