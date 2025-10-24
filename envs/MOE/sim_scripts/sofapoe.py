# -*- coding: utf-8 -*-
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from stlib3.physics.collision import CollisionMesh
from softrobots.actuators import PullingCable
from splib3.loaders import loadPointListFromFile
from stlib3.physics.mixedmaterial import Rigidify
import math
import Sofa.Core
import Sofa.constants.Key as Key
from stlib3.components import addOrientedBoxRoi
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
def getTranslated(points, vec):
    return [[v[0] + vec[0], v[1] + vec[1], v[2] + vec[2]] for v in points]

def transform(point, T1):
    T0 = np.eye(4)
    T0[0,3], T0[1,3], T0[2,3] = point
    T = np.matmul(T1, T0)
    return T[:3,3]


class FingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.cable1, self.cable2, self.cable3, self.cable4 = args[0:4]
        self.elasticobject = args[4]
        self.T = args[5]
        self.name = "FingerController"
        self.count = 0
        


    def moveDown(self, e):
        displacement1 = self.cable1.CableConstraint.value[0]
        displacement2 = self.cable2.CableConstraint.value[0]
        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        #print("Key pressed:", ord(e["key"]))

        
        displacement1 += 10000.
        displacement2 -= 10000.
        displacement2 = max(displacement2, 0)
       
        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)
     

        self.cable1.CableConstraint.value = [displacement1]
        self.cable2.CableConstraint.value = [displacement2]
      

    def moveUp(self):
        displacement1 = self.cable1.CableConstraint.value[0]
        displacement2 = self.cable2.CableConstraint.value[0]

        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        
        displacement1 -= 10000.
        displacement2 += 10000.
        displacement1 = max(displacement1, 0)
        
        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)
        #mecaobject.position.value = getTranslated(mecaobject.position.value, direction)
        #print("Cable displacements:", displacement1, displacement2, displacement3, displacement4)


        self.cable1.CableConstraint.value = [displacement1]
        self.cable2.CableConstraint.value = [displacement2]
    
    def moveLeft(self):
        displacement3 = self.cable3.CableConstraint.value[0]
        displacement4 = self.cable4.CableConstraint.value[0]
        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        
        displacement3 -= 10000.
        displacement4 += 10000.
        displacement3 = max(displacement3, 0)
        
       
        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)
        #mecaobject.position.value = getTranslated(mecaobject.position.value, direction)
        #print("Cable displacements:", displacement1, displacement2, displacement3, displacement4)
        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]

    def moveRight(self):
        displacement3 = self.cable3.CableConstraint.value[0]
        displacement4 = self.cable4.CableConstraint.value[0]
        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        print(direction)
        #Just right arrow
        displacement3 += 10000.
        displacement4 -= 10000.
        displacement4 = max(displacement4, 0)
        
        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)
        #mecaobject.position.value = getTranslated(mecaobject.position.value, direction)
        #print("Cable displacements:", displacement1, displacement2, displacement3, displacement4)

        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]
        #print(displacement3)
        #print(displacement4)
    

    def onAnimateBeginEvent(self, event):
        displacement1 = self.cable1.CableConstraint.value[0]
        displacement2 = self.cable2.CableConstraint.value[0]
        displacement3 = self.cable3.CableConstraint.value[0]
        displacement4 = self.cable4.CableConstraint.value[0]

        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]

        self.count += 0.05  
        amp = 5000.0      

        dx = amp * math.cos(self.count)
        dy = amp * math.sin(self.count)

        displacement1 = max(0.0,  displacement1 + dx)
        displacement2 = max(0.0, displacement2 - dx)
        

        displacement3 = max(0.0, displacement3 + dy)
        displacement4 = max(0.0, displacement4 - dy)

        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)

        self.cable1.CableConstraint.value = [displacement1]
        self.cable2.CableConstraint.value = [displacement2]
        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]

        
        print("Force - " + str(self.cable3.CableConstraint.force.value))
        print("L1 - " + str(self.cable1.CableConstraint.cableLength.value))
        print("L2 - " + str(self.cable2.CableConstraint.cableLength.value))
        print("L3 - " + str(self.cable3.CableConstraint.cableLength.value))
        print("L4 - " + str(self.cable4.CableConstraint.cableLength.value))
        
    def onKeypressedEvent(self, e):
        displacement1 = self.cable1.CableConstraint.value[0]
        displacement2 = self.cable2.CableConstraint.value[0]
        displacement3 = self.cable3.CableConstraint.value[0]
        displacement4 = self.cable4.CableConstraint.value[0]
        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        print(direction)
        #print("Key pressed:", ord(e["key"]))
        if e["key"] == Key.uparrow:
            print("ENTER")
            displacement1 -= 10000.
            displacement2 += 10000.
            displacement1 = max(displacement1, 0)
        elif e["key"] == Key.downarrow:
            displacement1 += 10000.
            displacement2 -= 10000.
            displacement2 = max(displacement2, 0)
        elif e["key"] == Key.leftarrow:
            displacement3 -= 10000.
            displacement4 += 10000.
            displacement3 = max(displacement3, 0)
        elif e["key"] == Key.rightarrow:
            displacement3 += 10000.
            displacement4 -= 10000.
            displacement4 = max(displacement4, 0)

        elif e["key"] == Key.plus:
            direction = [0.0, 0.0, 1.0]
       
        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)
        #mecaobject.position.value = getTranslated(mecaobject.position.value, direction)
        #print("Cable displacements:", displacement1, displacement2, displacement3, displacement4)


        self.cable1.CableConstraint.value = [displacement1]
        self.cable2.CableConstraint.value = [displacement2]
        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]

def Finger(parentNode=None, name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
           fixingBox=[0.0, 0.0, 0.0], pullPointLocation=[0.0, 0.0, 0.0],
           dir=None, T=None, T_dir=None): 
    finger = parentNode.addChild(name)
    parentNode.gravity.value = [-0, 0, 0]
    eobject = ElasticMaterialObject(
        attachedTo=parentNode,
        volumeMeshFileName='SIM_SCENE_MESH/MOE_MESH/Finger_with_Tip.vtk',
        name='finger',
        rotation=rotation,
        translation=translation,
        surfaceMeshFileName='SIM_SCENE_MESH/MOE_MESH/Finger_with_Tip.stl',
        collisionMesh='SIM_SCENE_MESH/MOE_MESH/Finger_with_Tip.stl',
        withConstrain=True,
        surfaceColor=[1.0, 1.5, 1.5, 1.6],
        poissonRatio=0.1,
        youngModulus=2000000,
        totalMass=400.0)

    finger.addChild(eobject)
    eobject.integration.rayleighStiffness = 0.1
    eobject.integration.rayleighMass = 0.1

    T_box = T.copy()
    T_box[0:3,0:3] = np.eye(3)
    top = transform([-150, -10, -150], T_box)
    bottom = transform([150, 10, 150], T_box)
    FixedBox(eobject, doVisualization=False, atPositions=[*top, *bottom])

    cable1 = PullingCable(eobject, cableGeometry=loadPointListFromFile("SIM_SCENE_MESH/OBJECTS/cable1copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable2 = PullingCable(eobject, cableGeometry=loadPointListFromFile("SIM_SCENE_MESH/OBJECTS/cable2copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable3 = PullingCable(eobject, cableGeometry=loadPointListFromFile("SIM_SCENE_MESH/OBJECTS/cable3copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable4 = PullingCable(eobject, cableGeometry=loadPointListFromFile("SIM_SCENE_MESH/OBJECTS/cable4copy.json"), valueType="force", translation=translation, rotation=rotation)

    finger.addObject(FingerController(cable1, cable2, cable3, cable4, eobject, T_dir))

    top = transform([-50, 92, -50], T)
    bottom = transform([50, 97, 50], T)

    finger.addObject('VisualModelOBJExporter',
                     exportEveryNumberOfSteps='5',
                     exportAtBegin='0',
                     name='exporter'+name,
                     listening='0',
                     exportAtEnd='1',
                     filename=dir + '/mesh')

    eobject.CollisionModel.addObject('VTKExporter',
                     name='exporter'+name,
                     pointsDataFields='collision.constraint',
                     XMLformat='0',
                     edges='0',
                     triangles='0',
                     exportEveryNumberOfSteps='5',
                     exportAtBegin='0',
                     exportAtEnd='1',
                     listening='1',
                     filename=dir + '/contact/'+name)

def createScene(rootNode):
    from stlib3.scene import MainHeader, ContactHeader

    # Plugin requirements
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Solver')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Setting')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')

    MainHeader(rootNode, plugins=['SofaPython3', 'SoftRobots', 'SofaOpenglVisual'])
    rootNode.VisualStyle.displayFlags = "showBehavior showCollisionModels"
    rootNode.addObject('BackgroundSetting', color='0 0.0 0.0')
    ContactHeader(rootNode, alarmDistance=2, contactDistance=0.1, frictionCoef=1.0)
    rootNode.getObject("GenericConstraintSolver").computeConstraintForces = True

    xi, yi, zi, ytheta, xtheta = 0, 0, 0, 0.0, 0.0


    exp_name = 'experiment_demo'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., '20250612_113045'
    root_dir = f'/media/moe/data/{object}_{xi}_{yi}_{zi}_{ytheta}_{xtheta}_{timestamp}'.replace(".", "-")
    os.makedirs(root_dir + '/mesh')
    os.makedirs(root_dir + '/contact')

    dx, dy, dz = 30.0*xi, 30.0*yi, 30.0*zi
    Ry = R.from_euler('y', 30.0*ytheta, degrees=True).as_matrix()
    Rx = R.from_euler('x', 30.0*xtheta, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3,:3] = Rx @ Ry
    T[0,3], T[1,3], T[2,3] = dx, dy, dz
    T_dir = T.copy()
    T_dir[:3,:3] = Ry

    T0 = np.eye(4)
    R0 = R.from_euler('y', 0, degrees=True).as_matrix()
    T0[:3,:3], T0[0,3] = R0, -33.0
    T1 = np.matmul(T, T0)
    rotation = R.from_matrix(T1[:3,:3]).as_euler('xyz', degrees=True)

    Finger(rootNode, 'finger1', rotation, [T1[0,3], T1[1,3], T1[2,3]], dir=root_dir, T=T, T_dir=T_dir)
    np.save(root_dir+'/T.npz', T)

    T0[:3,:3] = R.from_euler('y', 180, degrees=True).as_matrix()
    T0[0,3] = 33.0
    T1 = np.matmul(T, T0)
    rotation = R.from_matrix(T1[:3,:3]).as_euler('xyz', degrees=True)
    Finger(rootNode, 'finger2', rotation, [T1[0,3], T1[1,3], T1[2,3]], dir=root_dir, T=T, T_dir=T_dir)

    return rootNode




