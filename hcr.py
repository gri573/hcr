#!/usr/bin/python3

import taichi as ti
import time
import numpy as np
from PIL import Image as IM

ti.init(arch=ti.gpu)

@ti.func
def terrainNoiseTi(x : float) -> float:
    h = ti.sin(0.3 * x) + ti.cos(0.5 * x) + 3 * ti.sin(0.1 * x) + 5 * ti.sin(0.03 * x) + 0.2 * ti.sin(0.9 * x)
    return 0.005 * x * h

def terrainNoisePy(x : float) -> float:
    h = ti.sin(0.3 * x) + ti.cos(0.5 * x) + 3 * ti.sin(0.1 * x) + 5 * ti.sin(0.03 * x) + 0.2 * ti.sin(0.9 * x)
    return 0.005 * x * h

def dTerrainNoisedx(x : float, dx : float = 0.05) -> float:
    f = (terrainNoisePy(x + dx * 0.5) - terrainNoisePy(x - dx * 0.5)) / dx
    if (f == 0):
        return 0.00000000001
    return f
mat2 = ti.types.matrix(2, 2, float)
mat3 = ti.types.matrix(3, 3, float)
vec2 = ti.types.vector(2, float)
vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)
ivec2 = ti.types.vector(2, int)
uvec2 = ti.types.vector(2, ti.u32)
int8_t = ti.types.vector(1, ti.i8)
uint = ti.types.vector(1, ti.u32)
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
        collisionpoints = (vec2(0), vec2(1, 0), vec2(1, 0.5), vec2(0.66, 1), vec2(0.28, 1), vec2(0, 0.5)),
        driver = vec2(0.4, 1.2),
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
        self.phi = 0
        self.prevphi = self.phi
        self.wlocdiff = vec2(0, 0)
        self.m = m
        self.J = m / 12 * ((upper[0] - lower[0]) ** 2 + (upper[1] - lower[1]) ** 2)
        self.lower = lower
        self.upper = upper
        self.collisionpoints = tuple([point * (self.upper - self.lower) + self.lower for point in collisionpoints])
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

endscreentext = int8_t.field(shape=(256, 10))
endscreentext.fill(0)
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

@ti.func
def hash22(x : int, y : int) -> vec2:
    q = uvec2(x, y)
    q *= uvec2(ti.u32(1597334673), ti.u32(3812015801))
    q = (q.x ^ q.y) * uvec2(ti.u32(1597334673), ti.u32(3812015801))
    return vec2(q) * (1.0 / 4294967295.0);

@ti.kernel
def drawBackground(car_x : vec2):
    for i, j in pixels:
        loc = pixelsize * (vec2(i, j) + screenOffset) + car_x * vec2(1, 0.8)
        skyloc = loc - 0.5 * car_x * vec2(1, 0.8)
        color = vec3(-0.06 * skyloc[1] + 0.3, -0.04 * skyloc[1] + 0.5, 0.01 * skyloc[1] + 1.0)
        height = loc[1] - terrainNoiseTi(loc[0])
        if (height < 0):
            color = vec3(0.2, 0.4, 0.1) * 0.2 * (height + 6)
            grassheight = ti.sin(2 * loc[0]) * 0.3 - 0.7
            if (height < grassheight):
                color = vec3(0.5, 0.3, 0.2)
                heightDarkenFactor = 0.05
                #pebbles
                tilecoord = ti.floor(loc * 0.3)
                localcoord = (loc * 0.3 - tilecoord) * 2 - 1
                localcoord += 0.7 * (hash22(int(tilecoord.x + 1000.5), int(tilecoord.y + 1000.5)) * 2 - 1)
                randMat = mat2(hash22(int(tilecoord.x + 1012.5), int(tilecoord.y + 1402.5)) + 0.75, hash22(int(tilecoord.x + 1180.5), int(tilecoord.y + 1292.5)) * 0.5 - 0.25)
                randMat = mat2(randMat[0, 0] * ti.math.sign(randMat[1, 0]), randMat[1, 1], randMat[1, 0], randMat[0, 1])
                localcoord = randMat @ localcoord
                pebbleSize = 0.15 + 0.02 * abs(height)
                localcoord = 0.2 * ti.floor(5 * localcoord / pebbleSize)
                locallen = localcoord.norm()
                if (locallen < 1.0):
                    heightDarkenFactor = 0.04
                    color = vec3(0.2 * localcoord.y + 0.7 - (0.5 * locallen) ** 2)
                color *= ti.exp(heightDarkenFactor * height)
        pixels[i, j] = color

@ti.kernel
def drawCar(car_x : vec2, car_phi : float, car_wlocdiff : vec2):
    rotMat = genRotMatTi(car_phi)
    tireRotMat = genRotMatTi(-car_x[0] / car.wrad)
    for i, j in pixels:
        loc = pixelsize * (vec2(i, j) + screenOffset) + car_x * vec2(1, 0.8)
        relloc = rotMat @ (loc - car_x)
        #body
        brelloc = (relloc - car.lower) / (car.upper - car.lower)
        if (brelloc[0] > 0 and brelloc[1] > 0 and brelloc[0] < 1 and brelloc[1] < 1):
            brelloc.y = 1 - brelloc.y
            bodycoords = ivec2(chassis.shape * brelloc.yx)
            col = chassis[int(bodycoords.x), int(bodycoords.y)]
            if (col.a > 0.5): pixels[i, j] = col.rgb
        #wheels
        for k in ti.static(range(2)):
            wheelloc = (relloc - (vec2(car.wloc[k, 0], car.wloc[k, 1]) + car_wlocdiff[k] * vec2(0, 1))) / (2 * car.wrad)
            wheelloc = tireRotMat @ wheelloc
            wheelloc += 0.5
            if (wheelloc[0] > 0 and wheelloc[0] < 1 and wheelloc[1] > 0 and wheelloc[1] < 1):
                wheelloc.y = 1 - wheelloc.y
                wheelcoords = ivec2(tire.shape * wheelloc.yx)
                col = tire[int(wheelcoords.x), int(wheelcoords.y)]
                if (col.a > 0.5): pixels[i, j] = col.rgb
        #if ((relloc - car.driver).norm() < 0.3): pixels[i, j] = vec3(0, 0, 1)
chars = uint.field(shape=(128,))
chars[65]    = uint(0x747f18c4)
chars[66]    = uint(0xf47d18f8)
chars[67]    = uint(0x746108b8)
chars[68]    = uint(0xf46318f8)
chars[69]    = uint(0xfc39087c)
chars[70]    = uint(0xfc390840)
chars[71]    = uint(0x7c2718b8)
chars[72]    = uint(0x8c7f18c4)
chars[73]    = uint(0x71084238)
chars[74]    = uint(0x084218b8)
chars[75]    = uint(0x8cb928c4)
chars[76]    = uint(0x8421087c)
chars[77]    = uint(0x8eeb18c4)
chars[78]    = uint(0x8e6b38c4)
chars[79]    = uint(0x746318b8)
chars[80]    = uint(0xf47d0840)
chars[81]    = uint(0x74631934)
chars[82]    = uint(0xf47d18c4)
chars[83]    = uint(0x7c1c18b8)
chars[84]    = uint(0xf9084210)
chars[85]    = uint(0x8c6318b8)
chars[86]    = uint(0x8c62a510)
chars[87]    = uint(0x8c635dc4)
chars[88]    = uint(0x8a88a8c4)
chars[89]    = uint(0x8a884210)
chars[90]    = uint(0xf844447c)
chars[97]    = uint(0x0382f8bc)
chars[98]    = uint(0x85b318f8)
chars[99]    = uint(0x03a308b8)
chars[100]   = uint(0x0b6718bc)
chars[101]   = uint(0x03a3f83c)
chars[102]   = uint(0x323c8420)
chars[103]   = uint(0x03e2f0f8)
chars[104]   = uint(0x842d98c4)
chars[105]   = uint(0x40308418)
chars[106]   = uint(0x080218b8)
chars[107]   = uint(0x4254c524)
chars[108]   = uint(0x6108420c)
chars[109]   = uint(0x06ab5ac4)
chars[110]   = uint(0x07a318c4)
chars[111]   = uint(0x03a318b8)
chars[112]   = uint(0x05b31f40)
chars[113]   = uint(0x03671784)
chars[114]   = uint(0x05b30840)
chars[115]   = uint(0x03e0e0f8)
chars[116]   = uint(0x211c420c)
chars[117]   = uint(0x046318bc)
chars[118]   = uint(0x04631510)
chars[119]   = uint(0x04635abc)
chars[120]   = uint(0x04544544)
chars[121]   = uint(0x0462f0f8)
chars[122]   = uint(0x07c4447c)
chars[48]    = uint(0x746b58b8)
chars[49]    = uint(0x23084238)
chars[50]    = uint(0x744c88fc)
chars[51]    = uint(0x744c18b8)
chars[52]    = uint(0x19531f84)
chars[53]    = uint(0xfc3c18b8)
chars[54]    = uint(0x3221e8b8)
chars[55]    = uint(0xfc422210)
chars[56]    = uint(0x745d18b8)
chars[57]    = uint(0x745e1130)
chars[32]    = uint(0x0000000)
chars[46]    = uint(0x000010)
chars[45]    = uint(0x0000e000)
chars[44]    = uint(0x00000220)
chars[58]    = uint(0x02000020)

@ti.kernel
def drawEndScreen():
    for i, j in pixels:
        pixels[i, j] *= 0.5
        tilecoords0 = ivec2(vec2(i, canvassize[1] - j) - 0.25 * vec2(canvassize))
        if (min(tilecoords0.x, tilecoords0.y) >= 0):
            tilecoords0 //= 4
            tilecoords = tilecoords0 // ivec2(6, 10)
            localcoords = tilecoords0 % ivec2(6, 10)
            if (tilecoords.x < endscreentext.shape[0] and tilecoords.y < endscreentext.shape[1] and localcoords.x < 5 and localcoords.y < 6):
                if (endscreentext[int(tilecoords.x), int(tilecoords.y)].x != 0):
                    coordint = int((6 - localcoords.x) + 5 * (5 - localcoords.y))
                    if ((chars[endscreentext[int(tilecoords.x), int(tilecoords.y)].x].x >> coordint) % 2 != 0):
                        pixels[i, j] = vec3(1)

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
    if (terrainNoisePy(driver[0]) > driver[1] - 0.3):
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
        height = terrainNoisePy(wheelloc[0])
        steepness = dTerrainNoisedx(wheelloc[0])
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
                aextra =  min(30 * max(0, abs(car.rpm) - abs(np.dot(fwdir, v))), car.traction * force) * np.sign(car.rpm)
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
    for i in range(len(car.collisionpoints)):
        thisCorner0 = car.collisionpoints[i]
        thisCorner = rotMat @ thisCorner0 + car.x + dt * v
        height = terrainNoisePy(thisCorner[0])
        steepness = dTerrainNoisedx(thisCorner[0])
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
if (gui.running):
    drawBackground(car.x)
    drawCar(car.x, car.phi, car.wlocdiff)
    text = f"Distance: {car.x[0]:5.4g}\nFlips: {flips}\nBackflips: {backflips}\n\nPress escape\nto quit game"
    coords = [0, 0]
    for c in text:
        if (c == '\n'):
            coords[0] = 0
            coords[1] += 1
        else:
            endscreentext[tuple(coords)] = int8_t(ord(c))
            coords[0] += 1
    drawEndScreen()
    gui.set_image(pixels)
    gui.show()
    time.sleep(2)
    while (gui.get_event(ti.GUI.PRESS)): True
    while (not (gui.get_event(ti.GUI.PRESS) and gui.event.key == 'Escape')):
        time.sleep(0.05)
        gui.show()
gui.close()
print(f"\nDistance: {car.x[0]}\nFlips: {flips}\nBackflips: {backflips}")
