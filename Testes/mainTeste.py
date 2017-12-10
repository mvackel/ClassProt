
import mainPDB
from mainPDB import *
from mainDist import *
import py2neo
from py2neo import Graph

class PDBEntry() :
    def __init__(self, pdbId, ptype, classif="", ec="" ):
        self.pdbId = pdbId
        self.ptype = ptype
        self.classif = classif
        if len(ec) > 0 and len(classif) == 0 :
            self.classif = "Enzime"
        self.EC = ec

# class PDBEnzime(PDBEntry) :
#     def __init__(self, pdbId, classif, ec) :
#         super(PDBEnzime, self).__init__(pdbId, "ENZIME")
#         self.classif = classif
#         self.EC = ec

class AlphaCarbon() :
    def __init__(self, resSeq, resName, atomSerial, atomName, x, y, z):
        self.resSeq = resSeq
        self.resName = resName
        self.atomSerial = atomSerial
        self.atomName = atomName
        self.x = x
        self.y = y
        self.z = z


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


def createNodePDB( graph, pdbEntry ) :
    if not existsNodePDB( graph, pdbEntry.pdbId ) :
        # cuidado: label que comeca com numero ou tem espaco tem que ter a aspas!! "`"
        if len(pdbEntry.EC) == 0 :
            #query = f'CREATE ( p:PDB :`{pdbEntry.pdbId}` {{ Type: "{pdbEntry.ptype}" }} )'
            query = f'CREATE ( p:PDB {{ IdPDB: "{pdbEntry.pdbId}", Type: "{pdbEntry.ptype}" }} )'
        else :
            query = f'CREATE ( p:PDB {{ IdPDB: "{pdbEntry.pdbId}", Type: "{pdbEntry.ptype}", '
            'Classif: "{pdbEntry.classif}", EC: "{pdbEntry.EC}" }} )'

        graph.run(query)
        return True
    else :
        return False

def creatNodeCA( graph, pdbId, oCA ) :
    if not existsNodePDB(graph, pdbId):
        # cuidado: label que comeca com numero ou tem espaco tem que ter a aspas!! "`"
        query = f'CREATE ( p:CAlpha {{ IdPDB: "{pdbId}", ResSeq: "{oCA.resSeq}",  '
        'ResName: "{oCA.resName}",  AtomName: "{oCA.atomName}", AtomSeq:"{oCA.atomSeq}", '
        'X: "{oCA.x}", Y: "{oCA.y}", Z: "{oCA.z}" }} )'

        ret = graph.run(query)
        return True
    else:
        return False


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



