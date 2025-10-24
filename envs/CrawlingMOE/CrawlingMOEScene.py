import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from CrawlingMOEToolbox import ApplyAction, RewardShaper, StateInitializer
from sofagym.header import addVisu
from splib3.animation import AnimationManagerController
from stlib3.scene import ContactHeader, MainHeader

from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from splib3.loaders import loadPointListFromFile
import numpy as np
from scipy.spatial.transform import Rotation as R
from stlib3.components import addOrientedBoxRoi

global_gravity = [0.0, 0.0, 0.0]

def transform(point, T1):
    T0 = np.eye(4)
    T0[0, 3], T0[1, 3], T0[2, 3] = point
    T = np.matmul(T1, T0)
    return T[:3, 3]

def addMOE(parentNode, name, rotation, translation, T, T_dir, dir):
    finger = parentNode.addChild(name)
    parentNode.gravity.value = global_gravity
    
    mesh_dir = str(pathlib.Path(__file__).parent.absolute())

    eobject = ElasticMaterialObject(
        attachedTo=parentNode,
        volumeMeshFileName=f'{mesh_dir}/SIM_SCENE_MESH/MOE_MESH/Finger_with_Tip.vtk',
        name='finger',
        rotation=rotation,
        translation=translation,
        surfaceMeshFileName=f'{mesh_dir}/SIM_SCENE_MESH/MOE_MESH/Finger_with_Tip.stl',
        collisionMesh=f'{mesh_dir}/SIM_SCENE_MESH/MOE_MESH/Finger_with_Tip.stl',
        withConstrain=True,
        surfaceColor=[1.0, 1.5, 1.5, 1.6],
        poissonRatio=0.1,
        youngModulus=2000000,
        totalMass=400.0
    )

    finger.addChild(eobject)
    eobject.integration.rayleighStiffness = 0.1
    eobject.integration.rayleighMass = 0.1

    T_box = T.copy()
    T_box[0:3, 0:3] = np.eye(3)
    top = transform([-150, -10, -150], T_box)
    bottom = transform([150, 10, 150], T_box)
    FixedBox(eobject, doVisualization=False, atPositions=[*top, *bottom])

    cable1 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable1copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable2 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable2copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable3 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable3copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable4 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable4copy.json"), valueType="force", translation=translation, rotation=rotation)

    # Exporters (Currently Unnecessary)
    # finger.addObject('VisualModelOBJExporter',
    #                  exportEveryNumberOfSteps='5',
    #                  exportAtBegin='0',
    #                  name='exporter'+name,
    #                  listening='0',
    #                  exportAtEnd='1',
    #                  filename=dir + '/mesh')

    # eobject.CollisionModel.addObject('VTKExporter',
    #                  name='exporter'+name,
    #                  pointsDataFields='collision.constraint',
    #                  XMLformat='0',
    #                  edges='0',
    #                  triangles='0',
    #                  exportEveryNumberOfSteps='5',
    #                  exportAtBegin='0',
    #                  exportAtEnd='1',
    #                  listening='1',
    #                  filename=dir + '/contact/'+name)

    return finger, [cable1, cable2, cable3, cable4]


def createScene(rootNode,
                config={"seed": None,
                        "zFar": 400,
                        "dt": 0.01,
                        "init_states": [0.0] * 4},
                mode='simu'):

    visu, simu = 'visu' in mode, 'simu' in mode

    # === HEADER ===
    rootNode.name = "MOEEnv"
    rootNode.dt = config['dt']

    plugin_list = [
        "Sofa.Component.Visual",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.Engine.Generate",
        "Sofa.Component.Mass",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Model",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Collision.Geometry",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
        "SoftRobots"
    ]

    MainHeader(rootNode, gravity=global_gravity, dt=config['dt'], plugins=plugin_list)
    ContactHeader(rootNode, alarmDistance=0.2, contactDistance=0.1, frictionCoef=1.0)

    rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))
    if visu:
        cam_pos = [[0, 0, 300]]
        cam_dir = [[0, 0, -1]]
        addVisu(rootNode, config, cam_pos, cam_dir, cutoff=250)

    # === MODELING ===
    modeling = rootNode.addChild("Modeling")

    # Setup transform matrix T and T_dir
    xi = config.get("xi", 0)
    yi = config.get("yi", 0)
    zi = config.get("zi", 0)
    ytheta = config.get("ytheta", 0.0)
    xtheta = config.get("xtheta", 0.0)


    dx, dy, dz = 30.0 * xi, 30.0 * yi, 30.0 * zi
    Ry = R.from_euler('y', 30.0 * ytheta, degrees=True).as_matrix()
    Rx = R.from_euler('x', 30.0 * xtheta, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rx @ Ry
    T[0, 3], T[1, 3], T[2, 3] = dx, dy, dz
    T_dir = T.copy()
    T_dir[:3, :3] = Ry

    T0 = np.eye(4)
    T0[:3, :3] = R.from_euler('y', 0, degrees=True).as_matrix()
    T0[0, 3] = 0.0 #-33.0
    T1 = T @ T0
    rotation = R.from_matrix(T1[:3, :3]).as_euler('xyz', degrees=True)
    

    finger, cables = addMOE(modeling, name="finger", rotation=rotation,
                            translation=[T1[0,3], T1[1,3], T1[2,3]],
                            T=T, T_dir=T_dir, dir="")
    
    rootNode.addObject(StateInitializer(name="StateInitializer",
                                    rootNode=rootNode,
                                    cables=cables))

    target = config.get("target_pos", [35.0, 35.0, 35.0])


    rootNode.addObject(RewardShaper(name="Reward",
                                    rootNode=rootNode,
                                    target_pos=target))

    rootNode.addObject(ApplyAction(name="ApplyAction",
                                root=rootNode,
                                cables=cables))
    
    # === GOAL VISUALIZATION ===
    # target = config.get("target_pos", [0.0, 0.0, 50.0])
    # goalVisu = rootNode.addChild("TargetVisualization")
    # goalVisu.addObject("MechanicalObject", name="goalMO", position=[target])
    # goalVisu.addObject("Sphere", radius=5.0)
    # goalVisu.addObject("OglModel", color=[1.0, 0.0, 0.0, 1.0])  # red
    # goalVisu.addObject("IdentityMapping", input="@goalMO", output="@goalMO")
    # === GOAL VISUALIZATION (no deprecated Sphere) ===

    if visu:
        import os
        from pathlib import Path

        # pick any small mesh you already have in the repo; there are multiple cube.obj files
        cube_path = str(Path(__file__).parent.parent / "CartStem" / "mesh" / "cube.obj")
        # or, for another copy:
        # cube_path = str(Path(__file__).parent.parent / "MOEGripper" / "mesh" / "cube.obj")

        goal = rootNode.addChild("Target")
        visu = goal.addChild("Visu")
        visu.addObject("MeshOBJLoader", name="loader", filename=cube_path, triangulate=True)
        visu.addObject(
            "OglModel",
            src="@loader",
            color=[1.0, 0.0, 0.0, 1.0],     # red
            scale3d=[3.0, 3.0, 3.0],        # tweak size to taste
            translation=target              # place it at the goal
        )
    # No mapping needed since we just place it via 'translation'


    
    return rootNode

