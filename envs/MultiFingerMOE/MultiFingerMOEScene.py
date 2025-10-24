import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from MultiFingerMOEToolbox import ApplyAction, RewardShaper, StateInitializer
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

global_gravity = [0.0, 9810, 0.0]

def transform(point, T1):
    T0 = np.eye(4)
    T0[0, 3], T0[1, 3], T0[2, 3] = point
    T = np.matmul(T1, T0)
    return T[:3, 3]

def place_fingers_on_arc(
    n_fingers, arc_degrees=360.0, radius=50.0, tilt_deg=0.0, roll_deg=0.0,
    phase_deg=00.0,   # <<< NEW
):
    if n_fingers <= 0:
        return []

    if n_fingers == 1:
        base_R = R.from_euler('xyz', [tilt_deg, 0.0, roll_deg], degrees=True)
        return [(base_R.as_euler('xyz', degrees=True), [0.0, 0.0, 0.0])]

    phase_deg = -180.0/n_fingers
    out = []
    is_circle = abs((arc_degrees % 360.0)) < 1e-6 or abs(arc_degrees - 360.0) < 1e-6

    if is_circle:
        # closed circle: even spacing (prevents overlap)
        step = 360.0 / n_fingers
        for i in range(n_fingers):
            yaw = phase_deg + i * step
            x = radius * np.sin(np.deg2rad(yaw))
            z = radius * np.cos(np.deg2rad(yaw))
            y = 0.0

            R_yaw = R.from_euler('y', yaw, degrees=True)
            R_tilt_roll = R.from_euler('xz', [tilt_deg, roll_deg], degrees=True)
            R_flip = R.from_euler('z', 0, degrees=True)   # 180° roll
            euler = (R_flip * R_tilt_roll * R_yaw).as_euler('xyz', degrees=True)
            out.append((euler, [x, y+0, z]))
    else:
        # open arc centered on +Z
        start = -arc_degrees / 2.0 + phase_deg
        step = arc_degrees / (n_fingers - 1)
        for i in range(n_fingers):
            yaw = start + i * step
            x = radius * np.sin(np.deg2rad(yaw))
            z = radius * np.cos(np.deg2rad(yaw))
            y = 0.0

            R_yaw = R.from_euler('y', yaw, degrees=True)
            R_tilt_roll = R.from_euler('xz', [tilt_deg, roll_deg], degrees=True)
            R_flip = R.from_euler('z', 0, degrees=True)   # 180° roll
            euler = (R_flip * R_tilt_roll * R_yaw).as_euler('xyz', degrees=True)
            out.append((euler, [x, y+0, z]))

    return out

# === RENAMED: addMOE -> add_finger (kept alias below for backward compatibility) ===
def add_finger(parentNode, name, rotation, translation, T, T_dir, dir):
    finger = parentNode.addChild(name)
    
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
        youngModulus=50000,
        totalMass=400.0
    )

    finger.addChild(eobject)
    eobject.integration.rayleighStiffness = 0.1
    eobject.integration.rayleighMass = 0.1

    # Keep your existing fixed box, but decouple rotation (clamp to axis-aligned box)
    T_box = T.copy()
    T_box[0:3, 0:3] = np.eye(3)
    top = transform([-150, -10, -150], T_box)
    bottom = transform([150, 10, 150], T_box)
    FixedBox(eobject, doVisualization=True, atPositions=[*top, *bottom])

    cable1 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable1copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable2 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable2copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable3 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable3copy.json"), valueType="force", translation=translation, rotation=rotation)
    cable4 = PullingCable(eobject, cableGeometry=loadPointListFromFile(f"{mesh_dir}/SIM_SCENE_MESH/OBJECTS/cable4copy.json"), valueType="force", translation=translation, rotation=rotation)

    return finger, [cable1, cable2, cable3, cable4]

# Backward-compat alias so other files can still call addMOE
addMOE = add_finger

def createScene(rootNode,
                config={"seed": None,
                        "zFar": 400,
                        "dt": 0.01,
                        "init_states": [0.0] * 4},
                mode='simu and visu'):

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
    # ContactHeader(rootNode, alarmDistance=0.2, contactDistance=0.1, frictionCoef=1.0)
    ContactHeader(rootNode, alarmDistance=10.0, contactDistance=5.0, frictionCoef=1.0)


    rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))
    if visu:
        cam_pos = [[0, 0, 300]]
        cam_dir = [[0, 0, -1]]
        addVisu(rootNode, config, cam_pos, cam_dir, cutoff=250)

    # === MODELING ===
    modeling = rootNode.addChild("Modeling")


    # === CHANGED: read multi-finger placement params (all optional) ===
    n_fingers = int(config.get("n_fingers", 3))           # NEW
    arc_degrees = float(config.get("arc_degrees", 360.0)) # NEW
    palm_radius = float(config.get("palm_radius", 50.0))  # NEW
    tilt_deg = float(config.get("tilt_deg", 0.0))         # NEW (pitch)
    roll_deg = float(config.get("roll_deg", 0.0))         # NEW (roll)

    # Keep your legacy single-finger offsets/angles to preserve behavior
    xi = config.get("xi", 0)
    yi = config.get("yi", 0)
    zi = config.get("zi", 0)
    ytheta = config.get("ytheta", 0.0)
    xtheta = config.get("xtheta", 0.0)

    # Base transform for legacy single-finger; we’ll reuse its style per finger
    dx, dy, dz = 30.0 * xi, 30.0 * yi, 30.0 * zi
    Ry = R.from_euler('y', 30.0 * ytheta, degrees=True).as_matrix()
    Rx = R.from_euler('x', 30.0 * xtheta, degrees=True).as_matrix()
    base_T = np.eye(4)
    base_T[:3, :3] = Rx @ Ry
    base_T[0, 3], base_T[1, 3], base_T[2, 3] = dx, dy, dz
    base_T_dir = base_T.copy()
    base_T_dir[:3, :3] = Ry

    # === NEW: compute per-finger (rotation, translation) around a palm
    placements = place_fingers_on_arc(
        n_fingers=n_fingers,
        arc_degrees=arc_degrees,
        radius=palm_radius,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg
    )

    # Collect handles
    all_cables = []                # flattened for compatibility
    finger_nodes = []
    finger_cable_spans = []        # (start_idx, end_idx) per finger for your RL/action indexing

    for idx, (rot_euler, trans_xyz) in enumerate(placements):
        # Compose with your base_T so legacy per-scene orientation still applies
        T0 = np.eye(4)
        T0[:3, :3] = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        T0[0, 3], T0[1, 3], T0[2, 3] = trans_xyz

        T1 = base_T @ T0
        rotation = R.from_matrix(T1[:3, :3]).as_euler('xyz', degrees=True)

        finger_name = f"finger_{idx+1}" if n_fingers > 1 else "finger"

        # === CHANGED: use add_finger (renamed) ===
        finger_node, cables = add_finger(
            modeling,
            name=finger_name,
            rotation=rotation,
            translation=[T1[0, 3], T1[1, 3], T1[2, 3]],
            T=base_T,
            T_dir=base_T_dir,
            dir=""
        )

        finger_nodes.append(finger_node)

        # record span before extending
        start = len(all_cables)
        all_cables.extend(cables)    # flatten
        end = len(all_cables)
        finger_cable_spans.append((start, end))  # 4 per finger with current design

    # === CHANGED: feed flattened cables so existing components still work ===
    rootNode.addObject(StateInitializer(
        name="StateInitializer",
        rootNode=rootNode,
        cables=all_cables  # flattened list across fingers
    ))

    target = config.get("target_pos", [35.0, 35.0, 35.0])

    rootNode.addObject(RewardShaper(
        name="Reward",
        rootNode=rootNode,
        target_pos=target
    ))

    rootNode.addObject(ApplyAction(
        name="ApplyAction",
        root=rootNode,
        cables=all_cables  # flattened; index mapping in `finger_cable_spans` if you need it
    ))

    # Optionally expose mapping for downstream controllers/agents
    # (SOFA objects can carry data fields; here we just attach Python attributes)
    rootNode.finger_cable_spans = finger_cable_spans  # NEW (helper for action slicing)

    # === GOAL VISUALIZATION (unchanged except variable reuse) ===
    # if visu:
    #     from pathlib import Path
    #     cube_path = str(Path(__file__).parent.parent / "CartStem" / "mesh" / "cube.obj")
    #     goal = rootNode.addChild("Target")
    #     v = goal.addChild("Visu")
    #     v.addObject("MeshOBJLoader", name="loader", filename=cube_path, triangulate=True)
    #     v.addObject(
    #         "OglModel",
    #         src="@loader",
    #         color=[1.0, 0.0, 0.0, 1.0],
    #         scale3d=[10.0, 10.0, 10.0],
    #         translation=target
    #   )  


    floor = rootNode.addChild("Floor")
    # Load any flat mesh (a big quad or plane). If you already have one, use that path.
    floor.addObject('MeshOBJLoader', name='loader', filename='mesh/floor.obj', triangulate=True, scale=1.0)

    floor.addObject('MeshTopology', src='@loader')
    floor.addObject('MechanicalObject', name='mstate', template='Vec3', translation=[0.0, 160.0, 0.0])

    # Collision models (STATIC → moving=False, simulated=False)
    floor.addObject('TriangleCollisionModel', moving=False, simulated=False)
    floor.addObject('LineCollisionModel',     moving=False, simulated=False)
    floor.addObject('PointCollisionModel',    moving=False, simulated=False)

    # Visual (optional)
    vis = floor.addChild('Visu')
    vis.addObject('OglModel', src='@../loader', color=[0.6, 0.6, 0.6, 1.0])
    vis.addObject('IdentityMapping')
    

    cubeNode = rootNode.addChild("Cube")
    if simu:
        cubeNode.addObject('EulerImplicitSolver')
        cubeNode.addObject('SparseLDLSolver', name='solver', template="CompressedRowSparseMatrixMat3x3d")
        cubeNode.addObject('GenericConstraintCorrection', solverName='solver')

    cubeNode.addObject('MeshObjLoader', filename="mesh/cube.obj", flipNormals=False, triangulate=True,
                       name='meshLoader', scale=10.0)
    cubeNode.addObject('MeshTopology', position='@meshLoader.position', tetrahedra='@meshLoader.tetrahedra',
                       triangles='@meshLoader.triangles', drawTriangles='0')
    cubeMO = cubeNode.addObject('MechanicalObject', name="mstate", template="Vec3", scale=1.0,
                                translation=[0.0, 130.0, 0.0])
    cubeNode.addObject('UniformMass', totalMass=1.0)
    cubeNode.addObject('TriangleFEMForceField', youngModulus='4e3')

    cubeNode.addObject('TriangleCollisionModel')
    cubeNode.addObject('LineCollisionModel')
    cubeNode.addObject('PointCollisionModel')

    if visu:
        Cube_deform_Visu = cubeNode.addChild("VisualModel")

        Cube_deform_Visu.addObject('OglModel', name="model", src="@../meshLoader", color=[1., 1., 0.],
                                   updateNormals=True)

        Cube_deform_Visu.addObject('IdentityMapping')

    return rootNode
