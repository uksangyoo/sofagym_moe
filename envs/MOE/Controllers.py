# -*- coding: utf-8 -*-

import Sofa.Core
import Sofa.Simulation
from math import cos, sin, pi
import numpy as np
from scipy.spatial.transform import Rotation as R

from MOEToolbox import translateFingers, getRotationCenter


def rotate_x(point, angle, rotationCenter):
    translated = [point[0]-rotationCenter[0], point[1]-rotationCenter[1], point[2]-rotationCenter[2]]
    rotated = [translated[0],
               translated[1]*cos(angle)-translated[2]*sin(angle),
               translated[1]*sin(angle)+translated[2]*cos(angle)]
    return [rotated[0]+rotationCenter[0], rotated[1]+rotationCenter[1], rotated[2]+rotationCenter[2]]


def rotate_y(point, angle, rotationCenter):
    translated = [point[0]-rotationCenter[0], point[1]-rotationCenter[1], point[2]-rotationCenter[2]]
    rotated = [translated[0]*cos(angle)+translated[2]*sin(angle),
               translated[1],
               -translated[0]*sin(angle)+translated[2]*cos(angle)]
    return [rotated[0]+rotationCenter[0], rotated[1]+rotationCenter[1], rotated[2]+rotationCenter[2]]


def rotate_z(point, angle, rotationCenter):
    translated = [point[0]-rotationCenter[0], point[1]-rotationCenter[1], point[2]-rotationCenter[2]]
    rotated = [translated[0]*cos(angle)-translated[1]*sin(angle),
               translated[0]*sin(angle)+translated[1]*cos(angle),
               translated[2]]
    return [rotated[0]+rotationCenter[0], rotated[1]+rotationCenter[1], rotated[2]+rotationCenter[2]]


def rotateFingers(fingers, rotate, rot):
    rotationCenter = getRotationCenter(fingers)
    for finger in fingers:
        mecaobject = finger.tetras
        mecaobject.getData('rest_position').value = getRotated(rotate, mecaobject.getData('rest_position').value, rot,
                                                               rotationCenter)

        cable = finger.cables.cable1.aCableActuator
        p = cable.pullPoint
        cable.getData("pullPoint").value = rotate(p, rot, rotationCenter)


def getRotated(rotate, points, angle, rotationCenter):
    r = []
    for v in points:
        r.append(rotate(v, angle, rotationCenter))
    return r


def getTranslated(points, vec):
    return [[v[0] + vec[0], v[1] + vec[1], v[2] + vec[2]] for v in points]

def transform(point, T1):
    T0 = np.eye(4)
    T0[0,3], T0[1,3], T0[2,3] = point
    T = np.matmul(T1, T0)
    return T[:3,3]

class MOEFingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.cable1, self.cable2, self.cable3, self.cable4 = args[0:4]
        self.elasticobject = args[4]
        self.T = args[5]
        self.name = "MOEFingerController"
        self.count = 0

    def moveDown(self, e):
        displacement1 = self.cable1.CableConstraint.value[0]
        displacement2 = self.cable2.CableConstraint.value[0]
        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        
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
        
        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]

    def moveRight(self):
        displacement3 = self.cable3.CableConstraint.value[0]
        displacement4 = self.cable4.CableConstraint.value[0]
        directionT = np.eye(4)
        directionT[2,3] = 0.5 
        self.T[0:3,3] = 0.0
        direction = np.matmul(self.T, directionT)[0:3,3]
        
        displacement3 += 10000.
        displacement4 -= 10000.
        displacement4 = max(displacement4, 0)
        
        mecaobject = self.elasticobject.getObject('dofs')
        mecaobject.rest_position.value = getTranslated(mecaobject.rest_position.value, direction)
        
        self.cable3.CableConstraint.value = [displacement3]
        self.cable4.CableConstraint.value = [displacement4]

    def onAnimateBeginEvent(self, event):
        self.count += 1

    def onKeypressedEvent(self, e):
        key = e['key']
        if key == ord('W') or key == ord('w'):
            self.moveUp()
        elif key == ord('S') or key == ord('s'):
            self.moveDown(e)
        elif key == ord('A') or key == ord('a'):
            self.moveLeft()
        elif key == ord('D') or key == ord('d'):
            self.moveRight()

class MOEGripperController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.finger1_controller = args[0]
        self.finger2_controller = args[1]
        self.name = "MOEGripperController"

    def onKeypressedEvent(self, e):
        key = e['key']
        if key == ord('W') or key == ord('w'):
            self.finger1_controller.moveUp()
            self.finger2_controller.moveUp()
        elif key == ord('S') or key == ord('s'):
            self.finger1_controller.moveDown(e)
            self.finger2_controller.moveDown(e)
        elif key == ord('A') or key == ord('a'):
            self.finger1_controller.moveLeft()
            self.finger2_controller.moveRight()
        elif key == ord('D') or key == ord('d'):
            self.finger1_controller.moveRight()
            self.finger2_controller.moveLeft()


class GripperController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self)
        self.fingers = kwargs['fingers']
        self.name = "GripperController"
        self.rootNode = kwargs['rootNode']
        self.N = 0
        self.i = 0
        self.errorPlot = []
        self.flag = True

    def onBeginAnimationStep(self, deltaTime):
        pass

    def onKeypressedEvent(self, k):
        c = k['key']

        rot = None
        rotate = None
        direction = None

        if c == 'C':
            rot = 1/(2*pi)
            rotate = rotate_y
        elif c == 'A':
            rot = -1/(2*pi)
            rotate = rotate_y
        elif c == '5':
            rot = 1/(2*pi)
            rotate = rotate_x
        elif c == '6':
            rot = -1/(2*pi)
            rotate = rotate_x
        elif c == '7':
            rot = 1/(2*pi)
            rotate = rotate_z
        elif c == '8':
            rot = -1/(2*pi)
            rotate = rotate_z
        elif c == 'U':
            direction = [0.0, 1.0, 0.0]
        elif c == 'D':
            direction = [0, -1, 0]
        elif ord(c) == 18:
            direction = [1.0, 0.0, 0.0]
        elif ord(c) == 20:
            direction = [-1.0, 0.0, 0.0]
        elif ord(c) == 19:
            direction = [0.0, 0.0, 1.0]
        elif ord(c) == 21:
            direction = [0.0, 0.0, -1.0]

        if rot is not None:
            rotateFingers(self.fingers, rotate, rot)

        if direction is not None:
            translateFingers(self.fingers, direction)


class FingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self)
        self.name = 'Controller_' + kwargs['name']
        self.fingerName = kwargs['name']
        self.control1 = kwargs['control1']
        self.control2 = kwargs['control2']
        self.node = kwargs['node']
        self.cable = self.node.cables.cable1.aCableActuator.getData('value')

    def onKeypressedEvent(self, k):
        c = k['key']

        if c == self.control1:
            self.cable.value = [self.cable.value[0] + 1.]

        elif c == self.control2:
            displacement = self.cable.value[0] - 1.
            if displacement < 0:
                displacement = 0.
            self.cable.value = [displacement]
