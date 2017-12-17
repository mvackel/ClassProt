import requests
import numpy as np
from py2neo import Graph, authenticate, Node, Relationship
from typing import List


# def calcDistancesArr( aAllCoords, idxCAlpha) :
#     posCAlpha = aAllCoords[idxCAlpha]
#     dists = []
#     for aCoord in aAllCoords:
#         dists.append( np.linalg.norm( aCoord - posCAlpha ) )
#     return dists

def calcDistances(arrCAlphas, idxCAlphaFrom):
	posCAlphaFrom = arrCAlphas[idxCAlphaFrom].coords
	dists = []
	for oCA in arrCAlphas:
		dists.append(np.linalg.norm(np.array(oCA.coords) - np.array(posCAlphaFrom)))
	return dists


class PDBEntry(Node):
	def __init__(self, pdbId: str, title: str, ptype: str = '', organism: str = '', taxid: str = '', classif: str = '',
	             ec: str = ''):
		self.pdbId = pdbId.upper()
		self.title = title
		self.organism = organism
		self.taxid = taxid
		self.ptype = ptype  # Ex.: ENZIME
		self.classif = classif  # Ex.: LIGASE
		self.EC = ec
		super(PDBEntry, self).__init__('PDB', IdPDB=self.pdbId, Title=title, Type=ptype, Organism=organism, Taxid=taxid,
		                               Classif=classif, EC=ec)


class CAlpha(Node):
	def __init__(self, pdbId: str, resSeq: int, resName: str, atomSerial: int, coords: List[float]):
		self.pdbId = pdbId.upper()
		self.resSeq = resSeq
		self.resName = resName
		self.atomSerial = atomSerial
		self.coords = coords
		super(CAlpha, self).__init__('CAlpha', IdPDB=self.pdbId, ResSeq=resSeq, ResName=resName, AtomSerial=atomSerial,
		                             Coords=coords)
	
	def calcDist(self, newCAlpha):
		p1 = np.array(self.coords)
		p2 = np.array(newCAlpha.coords)
		return np.linalg.norm(p1 - p2)


class Near_10A(Relationship):
	def __init__(self, oCAlpha1, oCAlpha2):
		self.ca1 = oCAlpha1
		self.ca2 = oCAlpha2
		self.dist = oCAlpha1.calcDist(oCAlpha2)
		super(Near_10A, self).__init__(oCAlpha1, 'NEAR_10A', oCAlpha2, Dist=self.dist)


def getPDB(idPdb) -> (PDBEntry, List[CAlpha]):
	URL = f"https://files.rcsb.org/view/{idPdb}.pdb"
	
	r = requests.get(url=URL)  # , params=PARAMS)
	
	# extracting data in json format
	data = r.text
	# data = r.json()
	# print( data )
	arrCAlfa = []
	pdbEntryData = []
	title = ptype = organism = taxid = classif = ec = ''
	visited = {}
	linesList = data.split("\n")
	organismOk = False
	taxidOk = False
	ecOk = False
	for line in linesList:
		id = line[0:6]
		id = id.strip()
		if id == 'ATOM':
			name = line[12:16].strip()
			if name == 'CA' or name == "C1" or name == 'C1\'':
				resSeq = int(line[22:26].strip())
				if resSeq >= 0 and resSeq not in visited:
					visited[resSeq] = 1
					serial = int(line[6:11].strip())
					resName = line[17:20].strip()
					# type_of_chain = list[4]
					x = float(line[30:38].strip())
					y = float(line[38:46].strip())
					z = float(line[46:54].strip())
					aCoo = [x, y, z]
					calfa = CAlpha(idPdb, resSeq, resName, serial, aCoo)
					# print ( serial, name, resName, resSeq, x, y, z )
					arrCAlfa.append(calfa)
		elif id == 'SOURCE':
			name = line[11:].strip()
			tokens = name.split(':')
			if not organismOk and tokens[0] == 'ORGANISM_SCIENTIFIC':
				organism = tokens[1].strip(' ;')
				organismOk = True
			elif not taxidOk and tokens[0] == 'ORGANISM_TAXID':
				taxid = tokens[1].strip(' ;')
				taxidOk = True
		elif id == 'COMPND':
			if not ecOk and line[11:14] == 'EC:':
				ec = line[14:].strip(' ;')
				ecOk = True
		elif id == 'TITLE':
			title = line[10:].strip()
		elif id == 'HEADER':
			classif = line[10:50].strip()
	
	ptype = 'PROTEIN'
	if (len(ec) > 0):
		ptype = "ENZIME"
	
	pdbE = PDBEntry(idPdb, title, ptype, organism, taxid, classif, ec)
	
	return (pdbE, arrCAlfa)


def getIdNodePDB(graph:Graph, pdbId:str) -> int:
	query = f'MATCH (p: PDB) WHERE p.IdPDB = "{pdbId}" RETURN id(p) AS id'
	curs = graph.run(query)
	return curs.evaluate(field=0)


def existsNodePDB(graph:Graph, pdbId:str) -> bool:
	id = getIdNodePDB(graph, pdbId)
	return (id is not None)


def getIdNodeCA(graph:Graph, pdbId:str, resSeq:int) -> int:
	query = f'MATCH (n: CAlpha) WHERE n.IdPDB = "{pdbId}" AND n.ResSeq = {resSeq} RETURN id(n) AS id'
	curs = graph.run(query)
	return curs.evaluate(field=0)


def existsNodeCA(graph:Graph, pdbId:str, resSeq:int) -> bool:
	id = getIdNodeCA(graph, pdbId, resSeq)
	return (id is not None)

import time, sys
S_IndexCreated = False

def connectDB(dbUser:str, dbPwd:str, port=7474, host='localhost') -> Graph:
	authenticate("localhost:7474", dbUser, dbPwd)
	numTries = 10		# this is only necessary when there are many simultaneous connections
	dbGraph = None
	while numTries > 0:
		try:
			dbGraph = Graph()
			break
		except:
			print(f'-----Err: ', sys.exc_info()[0])
			time.sleep(1)
			numTries -= 1
			
	# This is to assure that there is an index on :CAlpha(IdPDB)
	# TODO: Refatorar todas as funcoes de DBGraph para uma classe
	global S_IndexCreated
	if not S_IndexCreated:
		query = f'CREATE INDEX ON :CAlpha(IdPDB)'
		dbGraph.run(query)
		S_IndexCreated = True
	
	return dbGraph