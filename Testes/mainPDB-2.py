# importing the requests library
import requests
import numpy


def getPDB(idPdb) -> list():
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
	visited = {}
	linesList = data.split("\n")
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
					aCoo = numpy.array((x, y, z))
					calfa = [serial, name, resName, resSeq, aCoo]
					# print ( serial, name, resName, resSeq, x, y, z )
					arrCAlfa.append(calfa)
		elif id == 'SOURCE':
			# TODO: IMPLEMENTAR - ORGANISMO, EC, NOME, ETC DA ENTRADA DO PDB
			name = line[11:].strip()
			tokens = name.split(':')
			if tokens[0] == 'ORGANIS_SCIENTIFIC':
				organism = tokens[1].strip(' ;')
				#pdbEntryData. =
			elif tokens[0] == 'ORGANISM_TAXID':
				taxid = tokens[1].strip(' ;')
		elif id == 'COMPND':
			#TODO: IMPLEMENTAR EC
			if line[11:14] == 'EC:':
				ec = line[14:].strip(' ;')
		elif id == 'TITLE':
			title = line[10:].strip()
	
	#dictRet = {'ENTRY': {}}
				
	return arrCAlfa


if __name__ == "__main__":
	# lstPdbEntries = ['5O75', '1LS6', '1Z28']
	idPdb = '5NH5'
	#idPdb = '5O75'
	#idPdb = '1LS6'
	#idPdb = '1Z28'
	idPdb = '5WEA'
	res = getPDB(idPdb)
	print(f'== idPdb: {idPdb}  #CAlfa: {len(res)}')
	print(res)

# main()



# extracting latitude, longitude and formatted address
# of the first matching location
# latitude = data['results'][0]['geometry']['location']['lat']
# longitude = data['results'][0]['geometry']['location']['lng']
# formatted_address = data['results'][0]['formatted_address']

# printing the output
# print("Latitude:%s\nLongitude:%s\nFormatted Address:%s"
#      % (latitude, longitude, formatted_address))
