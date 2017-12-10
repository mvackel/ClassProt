
import mainPDB
from mainPDB import *
from mainDist import *
import py2neo
from py2neo import Graph
from py2neo import Node
from py2neo import Relationship

class PDBEntry(Node) :
    def __init__(self, pdbId, ptype='', organism='', taxid='', classif='', ec='' ):
        self.pdbId = pdbId.upper()
        self.organism = organism
        self.taxid = taxid
        self.ptype = ptype       # Ex.: ENZIME
        self.classif = classif   # Ex.: LIGASE
        self.EC = ec
        super(PDBEntry, self).__init__('PDB', IdPDB = self.pdbId, Type = ptype, Classif = classif, EC = ec)

class CAlpha(Node) :
    def __init__(self, pdbId, resSeq, resName, atomSerial, coords):
        self.pdbId = pdbId.upper()
        self.resSeq = resSeq
        self.resName = resName
        self.atomSerial = atomSerial
        self.coords = coords
        super(CAlpha, self).__init__('CAlpha', IdPDB = self.pdbId, ResSeq = resSeq, ResName = resName,
                                     AtomSerial = atomSerial, Coords = coords)


def getIdNodePDB( graph, pdbId ) :
    query = f'MATCH (p: PDB) WHERE p.IdPDB = "{pdbId}" RETURN id(p) AS id'
    curs = graph.run(query)
    return curs.evaluate(field=0)

def existsNodePDB( graph, pdbId ) :
    id = getIdNodePDB( graph, pdbId )
    return (id is not None)

def getIdNodeCA( graph, pdbId, resSeq ) :
    query = f'MATCH (n: CAlpha) WHERE n.IdPDB = "{pdbId}" AND n.ResSeq = {resSeq} RETURN id(n) AS id'
    curs = graph.run(query)
    return curs.evaluate(field=0)

def existsNodeCA( graph, pdbId, resSeq ) :
    id = getIdNodeCA( graph, pdbId, resSeq )
    return (id is not None)


# def createNodePDB( graph, pdbEntry ) :
#     if not existsNodePDB( graph, pdbEntry.pdbId ) :
#         # cuidado: label que comeca com numero ou tem espaco tem que ter a aspas!! "`"
#         if len(pdbEntry.EC) == 0 :
#             #query = f'CREATE ( p:PDB :`{pdbEntry.pdbId}` {{ Type: "{pdbEntry.ptype}" }} )'
#             query = f'CREATE ( p:PDB {{ IdPDB: "{pdbEntry.pdbId}", Type: "{pdbEntry.ptype}" }} )'
#         else :
#             query = f'CREATE ( p:PDB {{ IdPDB: "{pdbEntry.pdbId}", Type: "{pdbEntry.ptype}", '
#             'Classif: "{pdbEntry.classif}", EC: "{pdbEntry.EC}" }} )'
#
#         graph.run(query)
#         return True
#     else :
#         return False
#
# def creatNodeCA( graph, pdbId, oCA ) :
#     if not existsNodePDB(graph, pdbId):
#         # cuidado: label que comeca com numero ou tem espaco tem que ter a aspas!! "`"
#         query = f'CREATE ( p:CAlpha {{ IdPDB: "{pdbId}", ResSeq: "{oCA.resSeq}",  '
#         'ResName: "{oCA.resName}",  AtomName: "{oCA.atomName}", AtomSeq:"{oCA.atomSeq}", '
#         'X: "{oCA.x}", Y: "{oCA.y}", Z: "{oCA.z}" }} )'
#
#         ret = graph.run(query)
#         return True
#     else:
#         return False


#def createRelCA( graph, ) :

# def createRelNear() :


#py2neo.authenticate("localhost:7474/db/data","neo4j","pxyon123")
#aCAs = getPDB("1YN2")
def main() :

    pdbId = "5O75"

    py2neo.authenticate("localhost:7474","neo4j","pxyon123")
    graph = Graph()

    id = getIdNodePDB(graph, "1XX1")
    print(id)
    id2 = getIdNodePDB(graph, "ABCD")
    print(id2)
    id3 = getIdNodeCA(graph, "1XX1", 10)
    print( id3 )
    id3 = getIdNodeCA(graph, "1XX1", 555)
    print( id3 )
    print('-------')

    ez1 = PDBEntry("2XX2", 'HUMAN', '9686', 'ENZIME', 'LIGASE', '2.2.2.2')
    ez2 = PDBEntry("3XX3", 'HUMAN', '9686', 'ENZIME', 'HIDROLASE', '3.3.3.3')

    ca1 = CAlpha("2XX2", 21, "GLU", 111, [2,3,4])
    ca2 = CAlpha("2XX2", 22, "LYS", 222, [2, 4, 4])
    ca3 = CAlpha("2XX2", 33, "ALA", 333, [2, 4, 5])

    graph.merge(ez1)
    graph.merge(ez2)
    graph.merge(ca1)
    graph.merge(ca2)
    graph.merge(ca3)

    id1 = getIdNodePDB(graph, "2XX2")
    print( id1 )
    id2 = getIdNodePDB(graph, "3XX3")
    print( id2 )

    id3 = getIdNodeCA(graph, "2XX2", 22)
    print (id3)

    r1 = Relationship(ca1, "NEAR_10A", ca2, distance=2.1)
    r2 = Relationship(ca1, "NEAR_10A", ca3, distance=3.2)

    graph.merge(r1)
    graph.merge(r2)
    # if not existsNodePDB( graph, pdbId ):
    #     aCAs = getPDB(pdbId)
    #
    #     print(f"Numero de carbonos alfa: {len(aCAs)}")
    #     print(f"Primeiro C Alfa: {aCAs[0]}")
    #
    #     entry = PDBEntry( pdbId, "ENZIME", "LIGASE", "2.3.2.27" )
    #     createNodePDB( graph, entry )
    #
    #     coords = [x[4] for x in aCAs ]
    #     for nca in range(len(coords)):
    #         dists = calcDistances( coords, nca )
    #         print( dists )


main()

# node = """
# CREATE ( p:PDB :Enzime {name: 'Marcos', code: '13.1.1.1'})
# """

#data = graph.run(query)



