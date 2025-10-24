
import os
import sys
import pathlib
import numpy as np
from splib3.loaders import loadPointListFromFile

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from Constants import *

path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
MOEMeshesPath = os.path.dirname(os.path.abspath(__file__))+'/moe_mesh/'

# Updated mesh paths for the new finger model
VolumetricMeshPath = MOEMeshesPath + 'updatedFingerNoHoles.vtk'
SurfaceMeshPath = MOEMeshesPath + 'updatedFingerNoHoles.stl'

# Cable JSON files
Cable1Path = MOEMeshesPath + 'cable1copy.json'
Cable2Path = MOEMeshesPath + 'cable2copy.json'
Cable3Path = MOEMeshesPath + 'cable3copy.json'
Cable4Path = MOEMeshesPath + 'cable4copy.json'


def Finger(rootNode, fixingBox, visu, simu, pullPointLocation, control1='1', control2='2', name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0]):

    model = rootNode.addChild(name)
    if simu:
        model.addObject('EulerImplicitSolver', name='odesolver')
        model.addObject('EigenSimplicialLDLT',template='CompressedRowSparseMatrixd', name='linearSolver')

    model.addObject('MeshVTKLoader', name='loader', filename=VolumetricMeshPath, scale3d=[1, 1, 1],
                    translation=translation, rotation=rotation)
    model.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra")
    model.addObject('TetrahedronSetTopologyModifier')
    model.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')

    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale='4e-5')
    model.addObject('UniformMass', totalMass='0.1')
    model.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                    poissonRatio=PoissonRation,  youngModulus=YoungsModulus)

    c = model.addChild("FixedBox")
    c.addObject('BoxROI', name='BoxROI', box=fixingBox, drawBoxes=True, doUpdate=False)
    c.addObject('RestShapeSpringsForceField', points='@BoxROI.indices', stiffness='1e12')

    if simu:
        model.addObject('LinearSolverConstraintCorrection', name='GCS', solverName='@precond')

        collisionmodel = model.addChild("CollisionMesh")
        collisionmodel.addObject("MeshSTLLoader", name="loader", filename=SurfaceMeshPath,
                                 rotation=rotation, translation=translation)
        collisionmodel.addObject('MeshTopology', src="@loader")
        collisionmodel.addObject('MechanicalObject')

        collisionmodel.addObject('PointCollisionModel')
        collisionmodel.addObject('LineCollisionModel')
        collisionmodel.addObject('TriangleCollisionModel')

        collisionmodel.addObject('BarycentricMapping')

    ##########################################
    # Effector                               #
    ##########################################

    for i in range(1, 5):
        CavitySurfaceMeshPath = MeshesPath+'Cavity0' + str(i) + '.stl'
        CurrentCavity = model.addChild('Cavity0'+str(i))
        CurrentCavity.addObject('MeshSTLLoader', name='MeshLoader', filename=CavitySurfaceMeshPath, rotation=rotation,
                                translation=translation)
        CurrentCavity.addObject('MeshTopology', name='topology', src='@MeshLoader')
        CurrentCavity.addObject('MechanicalObject', src="@topology")
        CurrentCavity.addObject('BarycentricMapping', name="Mapping", mapForces="false", mapMasses="false")

    ##########################################
    # Visualization                          #
    ##########################################
    if visu:
        modelVisu = model.addChild('visu')
        modelVisu.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader")
        modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1], translation=translation, rotation=rotation)
        modelVisu.addObject('BarycentricMapping')

    ##########################################
    # Actuation - Updated for 4 cables      #
    ##########################################

    cables = model.addChild('cables')
    
    # Cable 1
    cable1 = cables.addChild('cable1')
    cable1Geometry = loadPointListFromFile(Cable1Path)
    cable1.addObject('MechanicalObject', position=cable1Geometry,
                     translation=translation, rotation=rotation)
    cable1.addObject('CableConstraint', name="aCableActuator", 
                     indices=" ".join([str(i) for i in range(len(cable1Geometry))]), 
                     pullPoint=pullPointLocation)
    cable1.addObject('BarycentricMapping')

    # Cable 2
    cable2 = cables.addChild('cable2')
    cable2Geometry = loadPointListFromFile(Cable2Path)
    cable2.addObject('MechanicalObject', position=cable2Geometry,
                     translation=translation, rotation=rotation)
    cable2.addObject('CableConstraint', name="aCableActuator", 
                     indices=" ".join([str(i) for i in range(len(cable2Geometry))]), 
                     pullPoint=pullPointLocation)
    cable2.addObject('BarycentricMapping')

    # Cable 3
    cable3 = cables.addChild('cable3')
    cable3Geometry = loadPointListFromFile(Cable3Path)
    cable3.addObject('MechanicalObject', position=cable3Geometry,
                     translation=translation, rotation=rotation)
    cable3.addObject('CableConstraint', name="aCableActuator", 
                     indices=" ".join([str(i) for i in range(len(cable3Geometry))]), 
                     pullPoint=pullPointLocation)
    cable3.addObject('BarycentricMapping')

    # Cable 4
    cable4 = cables.addChild('cable4')
    cable4Geometry = loadPointListFromFile(Cable4Path)
    cable4.addObject('MechanicalObject', position=cable4Geometry,
                     translation=translation, rotation=rotation)
    cable4.addObject('CableConstraint', name="aCableActuator", 
                     indices=" ".join([str(i) for i in range(len(cable4Geometry))]), 
                     pullPoint=pullPointLocation)
    cable4.addObject('BarycentricMapping')

    return model
