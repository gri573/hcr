#!/usr/bin/python3

import taichi as ti
import numpy as np
from PIL import Image as IM

ti.init(arch=ti.gpu)

@ti.func
def smoothNoiseTi(x : float) -> float:
    h = ti.sin(0.3 * x) + ti.cos(0.5 * x) + 3 * ti.sin(0.1 * x) + 5 * ti.sin(0.03 * x) + 0.2 * ti.sin(0.9 * x)
    return 0.005 * x * h

def smoothNoisePy(x : float) -> float:
    h = ti.sin(0.3 * x) + ti.cos(0.5 * x) + 3 * ti.sin(0.1 * x) + 5 * ti.sin(0.03 * x) + 0.2 * ti.sin(0.9 * x)
    return 0.005 * x * h

def dSmoothNoisedx(x : float, dx : float = 0.05) -> float:
    f = (smoothNoisePy(x + dx * 0.5) - smoothNoisePy(x - dx * 0.5)) / dx
    if (f == 0):
        return 0.00000000001
    return f
mat2 = ti.types.matrix(2, 2, float)
mat3 = ti.types.matrix(3, 3, float)
vec2 = ti.types.vector(2, float)
vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)
ivec2 = ti.types.vector(2, int)
@ti.func
def genRotMatTi(phi : float) -> mat2:
    return mat2(ti.cos(phi), ti.sin(phi), -ti.sin(phi), ti.cos(phi))

def genRotMatPy(phi : float) -> mat2:
    return mat2(ti.cos(phi), ti.sin(phi), -ti.sin(phi), ti.cos(phi))

class car_t:
    def __init__(
        self,
        m = 5.0,
        lower = vec2(-1.5, 0.2),
        upper = vec2(1.5, 1.2),
        driver = vec2(0, 1.7),
        k = 4.0,
        gamma = 30.0,
        traction = 20.0,
        wrad = 0.4,
        wloc = mat2(0.87, 0, -0.85, 0)
    ):
        self.x = vec2(0, 2)
        self.prevx = self.x
        self.rpm = 0.0
        self.hp = 3.0
        self.phi = 0.0
        self.prevphi = 0.0
        self.wlocdiff = vec2(0, 0)
        self.m = m
        self.J = m / 12 * ((upper[0] - lower[0]) ** 2 + (upper[1] - lower[1]) ** 2)
        self.lower = lower
        self.upper = upper
        self.driver = driver
        self.k = k
        self.gamma = gamma
        self.traction = traction
        self.wrad = wrad
        self.wloc = wloc

car = car_t()
flips = 0
backflips = 0
rotloc = 0
gravity = vec2(0, -9.81)

canvassize = (800, 500)
pixels = vec3.field(shape=canvassize)
with IM.open("chassis.png") as chassis0:
    chassis1 = np.array(chassis0)
    chassis = vec4.field(shape=chassis1.shape[:2])
    for i in range(chassis1.shape[0]):
        for j in range(chassis1.shape[1]):
            chassis[i, j] = vec4(chassis1[i, j, :]) / 255.0
with IM.open("tire.png") as tire0:
    tire1 = np.array(tire0)
    tire = vec4.field(shape=tire1.shape[:2])
    for i in range(tire1.shape[0]):
        for j in range(tire1.shape[1]):
            tire[i, j] = vec4(tire1[i, j, :]) / 255.0

pixelsize = 0.03
screenOffset = vec2(-200, -200)

@ti.func
def col(r : float, g : float, b : float):
    return ti.Vector([r, g, b])

@ti.kernel
def drawBackground(car_x : vec2):
    for i, j in pixels:
        loc = pixelsize * (vec2(i, j) + screenOffset) + car_x * vec2(1, 0.5)
        color = vec3(-0.06 * loc[1] + 0.3, -0.04 * loc[1] + 0.5, 0.01 * loc[1] + 1.0)
        height = loc[1] - smoothNoiseTi(loc[0])
        if (height < 0):
            color = vec3(0.3, 0.4, 0.1)
        pixels[i, j] = color

@ti.kernel
def drawCar(car_x : vec2, car_phi : float, car_wlocdiff : vec2):
    rotMat = genRotMatTi(car_phi)
    tireRotMat = genRotMatTi(-car_x[0] / car.wrad)
    for i, j in pixels:
        loc = pixelsize * (vec2(i, j) + screenOffset) + car_x * vec2(1, 0.5)
        relloc = rotMat @ (loc - car_x)
        #body
        brelloc = (relloc - car.lower) / (car.upper - car.lower)
        if (brelloc[0] > 0 and brelloc[1] > 0 and brelloc[0] < 1 and brelloc[1] < 1):
            brelloc.y = 1 - brelloc.y
            bodycoords = ivec2(chassis.shape * brelloc.yx)
            col = chassis[bodycoords]
            if (col.a > 0.5): pixels[i, j] = col.rgb
        #wheels
        for k in ti.static(range(2)):
            wheelloc = (relloc - (vec2(car.wloc[k, 0], car.wloc[k, 1]) + car_wlocdiff[k] * vec2(0, 1))) / (2 * car.wrad)
            wheelloc = tireRotMat @ wheelloc
            wheelloc += 0.5
            if (wheelloc[0] > 0 and wheelloc[0] < 1 and wheelloc[1] > 0 and wheelloc[1] < 1):
                wheelloc.y = 1 - wheelloc.y
                wheelcoords = ivec2(tire.shape * wheelloc.yx)
                col = tire[wheelcoords]
                if (col.a > 0.5): pixels[i, j] = col.rgb

bounds = mat2(car.lower, car.upper)
alive = True

def moveCar(dt = 0.0166):
    if (ti.cos(car.phi) > 0):
        newrotloc = int(car.phi / (2 * np.pi) + 0.5)
        global rotloc
        if (newrotloc > rotloc):
            global backflips
            backflips += 1
        if (newrotloc < rotloc):
            global flips
            flips += 1
        rotloc = newrotloc
    midair = True
    v = (car.x - car.prevx) / dt
    w = (car.phi - car.prevphi) / dt
    oldx = car.prevx
    oldphi = car.prevphi
    car.prevx = car.x
    car.prevphi = car.phi
    v += gravity * dt
    rotMat = genRotMatPy(-car.phi)
    driver = rotMat @ car.driver + car.x
    if (smoothNoisePy(driver.x) > driver.y - 0.3):
        global alive
        alive = False
    prevRotMat = genRotMatPy(-oldphi)
    localUpDir = vec2(rotMat[0, 1], rotMat[1, 1])
    fwdir = vec2(rotMat[0, 0], rotMat[1, 0])

    #wheel collision
    for k in range(2):
        wheelloc = vec2(car.wloc[k, 0], car.wloc[k, 1])
        prevwheelloc = prevRotMat @ wheelloc + oldx
        wheelloc = rotMat @ wheelloc + car.x
        height = smoothNoisePy(wheelloc[0])
        steepness = dSmoothNoisedx(wheelloc[0])
        normal = vec2(1, -1 / steepness)
        normal /= normal.norm()
        if (normal[1] < 0.0):
            normal *= -1
        dist = (wheelloc[1] - height) / (1 + steepness ** 2) ** 0.5 - car.wrad
        dist = min(0, dist)
        cardotsurface = np.dot(normal, localUpDir)
        localv = wheelloc - prevwheelloc
        vdotsurface = -np.dot(normal, localv)
        force = - car.k * dist / cardotsurface
        force = min(force, 5.0)
        if (dist < 0):
            force += car.gamma * (vdotsurface)
            if (force > 0):
                midair = False
                w += 0.5 * force / car.J * car.wloc[k, 0]
                v += force / car.m * normal
                aextra =  min(30 * max(0, abs(car.rpm) - np.dot(fwdir, v) * np.sign(car.rpm)), car.traction * force) * np.sign(car.rpm)
                v += dt * aextra * fwdir
            car.wlocdiff[k] = -dist
        else:
            car.wlocdiff[k] = 0

    # mid-air control
    if (midair): w += dt / car.J * car.rpm
    car.x += v * dt * 0.9 ** dt
    car.phi += w * dt * 0.5 ** dt

    #body collision
    newx = car.x
    newphi = car.phi
    for i in range(2):
        for j in range(2):
            thisCorner0 = vec2(bounds[i, 0], bounds[j, 1])
            thisCorner = rotMat @ thisCorner0 + car.x + dt * v
            height = smoothNoisePy(thisCorner[0])
            steepness = dSmoothNoisedx(thisCorner[0])
            normal = vec2(1, -1 / steepness)
            normal /= normal.norm()
            if (normal[1] < 0.0):
                normal *= -1
            normal = (normal - 0.5 * dt * v)
            normal /= normal.norm()
            dist = (thisCorner[1] - height) / (1 + steepness ** 2) ** 0.5
            if (dist < 0):
                midair = False
                offCenterLength = np.dot(mat2(0, 1, -1, 0) @ rotMat @ thisCorner0, normal)
                xoffset = max(0, np.exp(-abs(offCenterLength)))
                newphi += 0.4 * dist * (1 - xoffset) / offCenterLength
                newx -= dist * normal * xoffset
                
    car.x = newx
    car.phi = newphi
    rotMat = genRotMatPy(-car.phi)
    prevRotMat = genRotMatPy(-car.prevphi)
    localUpDir = vec2(rotMat[0, 1], rotMat[1, 1])
    car.rpm *= 0.01 ** dt

gui = ti.GUI('hcr', canvassize, fast_gui = True)

def handleInputs():
    while (gui.get_event(ti.GUI.PRESS)):
        if (gui.event.key == 'Right'):
            car.rpm += car.hp
        elif (gui.event.key == 'Left'):
            car.rpm -= car.hp
        elif (gui.event.key == 'Escape'):
            gui.close()
            print("")
            exit()
        else:
            print(gui.event.key)

while gui.running and alive:
    drawBackground(car.x)
    drawCar(car.x, car.phi, car.wlocdiff)
    moveCar()
    handleInputs()
    gui.set_image(pixels)
    gui.show()

gui.close()
print(f"\nDistance: {car.x[0]}\nFlips: {flips}\nBackflips: {backflips}")
