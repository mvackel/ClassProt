# importing the requests library
import requests
import numpy
from py2neo import Node
from py2neo import Relationship
from typing import List


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
		super(PDBEntry, self).__init__('PDB', IdPDB=self.pdbId, Title=title, Type=ptype, Organism=organism, Taxid=taxid, Classif=classif, EC=ec)


class CAlpha(Node):
	def __init__(self, pdbId: str, resSeq: int, resName: str, atomSerial: int, coords: List[float]):
		self.pdbId = pdbId.upper()
		self.resSeq = resSeq
		self.resName = resName
		self.atomSerial = atomSerial
		self.coords = coords
		super(CAlpha, self).__init__('CAlpha', IdPDB=self.pdbId, ResSeq=resSeq, ResName=resName,
		                             AtomSerial=atomSerial, Coords=coords)


def getPDB3(idPdb) -> (PDBEntry, List[CAlpha]):
	# =========== GET
	# api-endpoint
	URL = f"https://files.rcsb.org/view/{idPdb}.pdb"
	
	# location given here
	# location = "delhi technological university"
	# location = "universidade federal de minas gerais"
	
	# defining a params dict for the parameters to be sent to the API
	# PARAMS = {'address': location}
	
	# sending get request and saving the response as response object
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
	if (len(ec)>0):
		ptype = "ENZIME"
		
	pdbE = PDBEntry(idPdb, title, ptype, organism, taxid, classif, ec)
	
	return (pdbE, arrCAlfa)


if __name__ == "__main__":
	# lstPdbEntries = ['5O75', '1LS6', '1Z28']
	idPdb = '5NH5'
	# idPdb = '5O75'
	# idPdb = '1LS6'
	# idPdb = '1Z28'
	idPdb = '5WEA'
	(entry, arrCa) = getPDB3(idPdb)
	print(f'== idPdb: {idPdb}  #CAlfa: {len(arrCa)}')
	print(f'== entry: {entry}')
	print(arrCa)
