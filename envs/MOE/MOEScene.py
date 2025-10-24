
import os
import SofaRuntime
import numpy as np

from splib3.animation import AnimationManagerController


import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from MOEToolbox import rewardShaper, goalSetter
from Gripper import Gripper

path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

SofaRuntime.importPlugin("Sofa.Component")


def add_plugins(root):
    root.addObject('RequiredPlugin', pluginName='BeamAdapter')
    root.addObject('RequiredPlugin', name='SofaOpenglVisual')
    root.addObject('RequiredPlugin', name="SofaMiscCollision")
    root.addObject('RequiredPlugin', name="SofaPython3")
    root.addObject('RequiredPlugin', name="SofaPreconditioner")
    root.addObject('RequiredPlugin', name="SoftRobots")
    root.addObject('RequiredPlugin', name="SofaConstraint")
    root.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    root.addObject('RequiredPlugin', name="SofaLoader")
    root.addObject('RequiredPlugin', name="SofaSparseSolver")

    root.addObject('RequiredPlugin', name="SofaDeformable")
    root.addObject('RequiredPlugin', name="SofaEngine")
    root.addObject('RequiredPlugin', name="SofaMeshCollision")
    root.addObject('RequiredPlugin', name="SofaMiscFem")
    root.addObject('RequiredPlugin', name="SofaRigid")
    root.addObject('RequiredPlugin', name="SofaSimpleFem")
    return root


def add_visuals_and_solvers(root, config, visu, simu):
    if visu:

        source = config["source"]
        target = config["target"]
        root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideMappings hideForceFields showWireframe')
        root.addObject("LightManager")

        spotLoc = [0, 0, source[2]]
        root.addObject("SpotLight", position=spotLoc, direction=[0.0, 0.0, -np.sign(source[2])])
        root.addObject('InteractiveCamera', name='camera', position=source, lookAt=target, zFar=500)
        root.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    if simu:
        root.addObject('FreeMotionAnimationLoop')
        root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
        root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
        root.addObject('BruteForceBroadPhase')
        root.addObject('BVHNarrowPhase')
    
        root.addObject('LocalMinDistance', contactDistance=5.0, alarmDistance=10.0, name='localmindistance',
                       angleCone=0.2)
        root.addObject('DefaultContactManager', name='Response', response='FrictionContactConstraint')

        root.addObject(AnimationManagerController(root))

    return root


def CreateObject(node, name, surfaceMeshFileName, visu, simu, translation=[0., 0., 0.], rotation=[0., 0., 0.],
                 uniformScale=1., totalMass=1., volume=1., inertiaMatrix=[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 color=[1., 1., 0.], isAStaticObject=False):

    object = node.addChild(name)

    object.addObject('MechanicalObject', name="mstate", template="Rigid3", translation2=translation,
                     rotation2=rotation, showObjectScale=uniformScale)

    object.addObject('UniformMass', name="mass", vertexMass=[totalMass, volume, inertiaMatrix[:]])

    if not isAStaticObject:
        object.addObject('UncoupledConstraintCorrection')
        object.addObject('EulerImplicitSolver', name='odesolver')
        object.addObject('CGLinearSolver', name='Solver')

    # collision
    if simu:
        objectCollis = object.addChild('collision')
        objectCollis.addObject('MeshObjLoader', name="loader", filename=surfaceMeshFileName, triangulate="true",
                               scale=uniformScale)

        objectCollis.addObject('MeshTopology', src="@loader")
        objectCollis.addObject('MechanicalObject')

        movement = not isAStaticObject
        objectCollis.addObject('TriangleCollisionModel', moving=movement, simulated=movement)

    # visualization
    if visu:
        objectVisu = object.addChild('visu')
        objectVisu.addObject('MeshObjLoader', name="loader", filename=surfaceMeshFileName, triangulate="true",
                             scale=uniformScale)
        objectVisu.addObject('OglModel', src="@loader", color=color)

    return object


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal_mo = goal.addObject('MechanicalObject', name='goalMO', position=[0., 0., 0.])
    return goal_mo


def create_scene(root,  config, visu, simu):
    """Create the scene with the gripper and the object to catch.

    Args:
        root: the root of the scene
        config: the configuration of the scene
        visu: if we want to add visualization
        simu: if we want to add simulation

    Returns:
        the scene
    """
    root.addObject('RequiredPlugin', name='SofaPython3')
    root.addObject('RequiredPlugin', name='SoftRobots')
    root.addObject('RequiredPlugin', name='BeamAdapter')

    root.addObject('RequiredPlugin', name='SofaOpenglVisual')
    root.addObject('RequiredPlugin', name='SofaMiscCollision')
    root.addObject('RequiredPlugin', name='SofaConstraint')
    root.addObject('RequiredPlugin', name='SofaImplicitOdeSolver')
    root.addObject('RequiredPlugin', name='SofaLoader')
    root.addObject('RequiredPlugin', name='SofaSparseSolver')
    root.addObject('RequiredPlugin', name='SofaDeformable')
    root.addObject('RequiredPlugin', name='SofaEngine')
    root.addObject('RequiredPlugin', name='SofaMeshCollision')
    root.addObject('RequiredPlugin', name='SofaMiscFem')
    root.addObject('RequiredPlugin', name='SofaRigid')
    root.addObject('RequiredPlugin', name='SofaSimpleFem')

    root.addObject('RequiredPlugin', name='SofaPreconditioner')

    add_visuals_and_solvers(root, config, visu, simu)

    if simu:
        root.gravity.value = [0.0, -9.81, 0.0]

    root.dt.value = config["dt"]

    gripper = Gripper(root, visu, simu)
    
    # Add cube object
    cubeNode = root.addChild("Cube")

    if simu:
        cubeNode.addObject('EulerImplicitSolver')
        cubeNode.addObject('SparseLDLSolver', name='solver')
        cubeNode.addObject('GenericConstraintCorrection', solverName='solver')

    cubeNode.addObject('MeshObjLoader', filename="mesh/cube.obj", flipNormals=False, triangulate=True,
                       name='meshLoader', scale=10.0)
    cubeNode.addObject('MeshTopology', position='@meshLoader.position', tetrahedra='@meshLoader.tetrahedra',
                       triangles='@meshLoader.triangles', drawTriangles='0')
    cubeMO = cubeNode.addObject('MechanicalObject', name="mstate", template="Vec3", scale=2.0,
                                translation=[0.0, -130.0, 0.0])
    cubeNode.addObject('UniformMass', totalMass=1.0)
    cubeNode.addObject('TriangleFEMForceField', youngModulus='4e2')

    cubeNode.addObject('TriangleCollisionModel')
    cubeNode.addObject('LineCollisionModel')
    cubeNode.addObject('PointCollisionModel')

    if visu:
        Cube_deform_Visu = cubeNode.addChild("VisualModel")
        Cube_deform_Visu.addObject('OglModel', name="model", src="@../meshLoader", color=[1., 1., 0.],
                                   updateNormals=True)
        Cube_deform_Visu.addObject('IdentityMapping')

    if config["goal"]:
        goal_mo = add_goal_node(root)
        root.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos=config['goalPos']))
        root.addObject(rewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos'], effMO=cubeMO))

    return root


def createScene(root, config={"source": [-600.0, -25, 100],
                              "target": [30, -25, 100],
                              "goalPos": [0, 0, 0],
                              "dt": 0.01}, mode='simu_and_visu'):
    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    create_scene(root, config, visu, simu)

    return root


def main():
    import SofaRuntime
    import Sofa.Gui

    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("SofaImplicitOdeSolver")

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    # Run the simulation
    if not root.configurationFileName.value:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(root, __file__)
        Sofa.Gui.GUIManager.closeGUI()

    print("Simulation is finished.")


if __name__ == '__main__':
    main()
