#!/usr/bin/env python3
import sys
from itertools import cycle
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
from core import Shader, Viewer, Mesh, load, Node
from texture import Texture, Textured, Skybox, SkyboxMesh
from transform import vec, quaternion, quaternion_from_euler
from animation import KeyFrameControlNode
from transform import translate, identity, rotate, scale
import random



# -------------- Example textured plane class ---------------------------------
class TexturedPlane(Textured):
    """ Simple first textured object """
    def __init__(self, shader,indices, tex_file1):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file1


        # setup plane mesh to be textured
        base_coords = ((-10000, -2, -10000), (10000, -2, -10000), (10000, 1, 10000), (-10000, 1, 10000))
        scaled = np.array(base_coords, np.float32)
        uv_coords = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), 'f')
        mesh = Mesh(shader, attributes=dict(position=scaled,uv=uv_coords), index=indices)


        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        texture1 = Texture(tex_file1, self.wrap, *self.filter)
        
        super().__init__(mesh, diffuse_map = texture1)

    def key_handler(self, key):
        # cycle through texture modes on keypress of F6 (wrap) or F7 (filtering)
        self.wrap = next(self.wraps) if key == glfw.KEY_F6 else self.wrap
        self.filter = next(self.filters) if key == glfw.KEY_F7 else self.filter
        if key in (glfw.KEY_F6, glfw.KEY_F7):
            texture = Texture(self.file, self.wrap, *self.filter)
            self.textures.update(diffuse_map=texture)

class Volcano(Textured):
    """ Volcano object """

    def __init__(self, shader, texture_file1, texture_file2):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = texture_file1

        # setup volcano mesh to be textured
        height = 20
        radius = 10.0
        num_segments = 100
        # generate vertices at different heights
        vertices = []
        indices = []
        uvs = []

        for h in range(int(height)):
            level_vertices = []
            level_uvs = []
            for i in range(num_segments+1):
                angle = (i / num_segments) * 2 * np.pi
                x = (radius-h*0.3) * np.cos(angle) + np.random.normal(0, 0.2)
                y = h - np.random.normal(0,0.1)
                z = (radius-h*0.3) * np.sin(angle) + np.random.normal(0, 0.2)
                level_vertices.append((x, y, z))
                level_uvs.append((i/num_segments,h/height))
            vertices.extend(level_vertices)
            uvs.extend(level_uvs)

        
        for h in range(height-1):
            for i in range(num_segments+1):
                indices.extend([h*num_segments+ i   ,h*num_segments+ 1 + i              ,h*num_segments+ i + num_segments+1])
                indices.extend([h*num_segments+ i+ 1,h*num_segments+ i                  ,h*num_segments+ i + num_segments+1])
                indices.extend([h*num_segments+ i   ,h*num_segments+ i + 1 +num_segments,h*num_segments+ i + num_segments])
                indices.extend([h*num_segments+ i   ,h*num_segments+ i +num_segments    ,h*num_segments+ i + 1 +num_segments])


        normals = np.zeros_like(vertices)

        tangents = np.zeros_like(vertices)
        bitangents = np.zeros_like(vertices)

        for i in range(0, len(indices), 3):

            a, b, c = indices[i:i+3]

            ab = (vertices[b][0] - vertices[a][0],vertices[b][1] - vertices[a][1],vertices[b][2] - vertices[a][2])
            ac = (vertices[c][0] - vertices[a][0],vertices[c][1] - vertices[a][1],vertices[c][2] - vertices[a][2])
            
            face_normal = np.array([ab[1] * ac[2] - ab[2] * ac[1],
                                    ab[2] *  ac[0] - ab[0] * ac[2],
                                    ab[0] * ac[1] - ab[1] * ac[0]])

            normals[a] += face_normal
            normals[b] += face_normal
            normals[c] += face_normal

            delta_uv1 = (uvs[b][0] - uvs[a][0], uvs[b][1] - uvs[a][1])
            delta_uv2 = (uvs[c][0] - uvs[a][0], uvs[c][1] - uvs[a][1])

            f = 1.0/(delta_uv1[0]*delta_uv2[1] - delta_uv2[0]*delta_uv1[1])
            
            tangent = ((delta_uv2[1] * ab[0] - delta_uv1[1] * ac[0])*f , (delta_uv2[1] * ab[1] - delta_uv1[1] * ac[1])*f , (delta_uv2[1] * ab[2] - delta_uv1[1] * ac[2])*f )
            bitangent = ((-delta_uv2[0] * ab[0] + delta_uv1[0] * ac[0])*f, (-delta_uv2[0] * ab[1] + delta_uv1[0] * ac[1])*f, (-delta_uv2[0] * ab[2] + delta_uv1[0] * ac[2])*f)

            tangents[a] += tangent
            tangents[b] += tangent
            tangents[c] += tangent

            bitangents[a] += bitangent
            bitangents[b] += bitangent
            bitangents[c] += bitangent



        mesh = Mesh(shader, attributes=dict(position=vertices,normal=normals,tangent=tangents,bitangent=bitangents,uv_vertex=uvs), index=indices)
        texture1 = Texture(texture_file1, self.wrap, *self.filter)
        texture2 = Texture(texture_file2, self.wrap, *self.filter)
        super().__init__(mesh, diffuse_map = texture1, normal_map = texture2)




class SkyboxCube(SkyboxMesh):
    """ Drawable skybox cube """
    def __init__(self, shader, tex_file):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file

        # setup cube mesh to be textured
        base_coords = ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1),  # left
                       (1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1))     # right
        scaled = 10 * np.array(base_coords, np.float32)
        indices = np.array((1, 0, 2, 3, 2, 0,   # back
                            5, 4, 6, 7, 6, 4,   # front
                            0, 5, 3, 6, 3, 5,   # left
                            4, 1, 7, 2, 7, 1,   # right
                            2, 3, 6, 6, 7, 2,  # bottom
                            1, 4, 0, 5, 0, 4),  # top
                           np.uint32)
        mesh = Mesh(shader, attributes=dict(position=scaled), index=indices)

        # setup & upload texture to GPU, bind it to shader name 'cube_map'
        texture = Skybox(tex_file, *self.filter)
        super().__init__(mesh, skybox=texture)





class Triangle(Mesh):
    """Hello triangle object"""
    def __init__(self, shader):
        position = np.array(((0, .5, 0), (-.5, -.5, 0), (.5, -.5, 0)), 'f')
        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')
        self.color = (1, 1, 0)
        attributes = dict(position=position, color=color)
        super().__init__(shader, attributes=attributes)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def key_handler(self, key):
        if key == glfw.KEY_C:
            self.color = (0, 0, 0)


class Cylinder(Textured):
    """ Simple first textured object """
    def __init__(self, shader, tex_file1):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file1

        point_cercle = [(0, 0, 2)]
        angles = np.arange(0, 2*np.pi, 2*np.pi/100)

        for angle in angles :
            x = np.cos(angle)
            y = np.sin(angle)
            z = 2
            point_cercle.append((x, y, z))
        
        nombre_sommets_base = len(point_cercle)

        point_cercle.append((0, 0, 0))
        
        for angle in angles:
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0
            point_cercle.append((x, y, z))

        position = np.array(point_cercle, 'f')

        indices = []
        for i in range(1, nombre_sommets_base) :
            indices.append(i)
            if i+1 < nombre_sommets_base:
                indices.append(i+1)
            else :
                indices.append(1)
            # indices.append(i+1)
            indices.append(0)
        
        for i in range(1, nombre_sommets_base):
            indices.append(i)
            indices.append(i+nombre_sommets_base)
            if i+nombre_sommets_base+1 < len(position):
                indices.append(i+nombre_sommets_base+1)
            else:
                indices.append(nombre_sommets_base+1)
        for i in range(1, nombre_sommets_base):
            indices.append(i)
            if i+1 < nombre_sommets_base:
                indices.append(i+1+nombre_sommets_base)
                indices.append(i+1)
            else:
                indices.append(1+nombre_sommets_base)
                indices.append(1)
        
        for i in range(nombre_sommets_base-1, 0, -1):
            indices.append(i+nombre_sommets_base)
            if i-1 > 0:
                indices.append(i-1+nombre_sommets_base)
            else :
                indices.append(2*nombre_sommets_base-1)
            # indices.append(i+1)
            indices.append(nombre_sommets_base)

        index = np.array(indices, np.uint32)
        uv_coords = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), 'f')

        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        mesh = Mesh(shader, attributes=dict(position=position,uv=uv_coords), index=index)
        texture1 = Texture(tex_file1, self.wrap, *self.filter)
        super().__init__(mesh, diffuse_map = texture1)



class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0):
        super().__init__(transform=rotate(axis, angle))
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down

    def key_handler(self, key):
        self.angle += 10 * int(key == self.key_up)
        self.angle -= 10 * int(key == self.key_down)
        self.transform = rotate(self.axis, self.angle)
        super().key_handler(key)

# class TransalationControlNode(Node):
#     def __init__(self, key_up, key_down, axis, position=0):
#         super().__init__(transform=translate)
#         self.position, self.axis = position, axis
#         self.key_up, self.key_down = key_up, key_down

#     def key_handler(self, key):
#         self.position[2] += 5 * int(key == self.key_up)
#         self.position[2] -= 5 * int(key == self.key_down)
#         self.transform = rotate(self.position,)
#         super().key_handler(key)

class PointAnimation(Mesh):
    """ Simple animated particle set """
    def __init__(self, shader):
        # render points with wide size to be seen
        GL.glPointSize(3)

        # instantiate and store 100 points to animate
        self.coords = []
        number_of_particles = 2000
        for _ in range (number_of_particles):
            x = random.uniform(-4, 4)
            z = random.uniform(-np.sqrt(16 - x*x), np.sqrt(16 - x*x))
            y = random.uniform(18, 100)
            self.coords.append([x, y, z])

        wind_list = []
        speed = 0.1
        number_of_directions = 129
        for i in range(number_of_particles):
            relative_speed = speed*((i+1)/number_of_particles)
            if i%number_of_directions == 0:
                wind_list.append([0.0, relative_speed, 0.0])
            else:
                wind_list.append([relative_speed*np.cos((i%number_of_directions - 1)*np.pi*2/(number_of_directions-1)), relative_speed, relative_speed*np.sin((i%number_of_directions - 1)*np.pi*2/(number_of_directions-1))])
        
        self.wind = np.array(wind_list)

        self.velocity = np.array([[0.0, 0.0, 0.0] for _ in range(number_of_particles)])

        # send as position attribute to GPU, set uniform variable global_color.
        # GL_STREAM_DRAW tells OpenGL that attributes of this object
        # will change on a per-frame basis (as opposed to GL_STATIC_DRAW)
        super().__init__(shader, attributes=dict(position=self.coords),
                         usage=GL.GL_STREAM_DRAW, global_color=(0.5, 0.5, 0.5))

    def draw(self, primitives=GL.GL_POINTS, attributes=None, **uniforms):
        # compute a sinusoidal x-coord displacement, different for each point.

        random_ratio = 2
        for i in range(len(self.velocity)):
            if i%100 == 0:
                random_ratio = random.choice([2, 5, 6])
            self.velocity[i] = self.velocity[i] + np.array([self.wind[i][0], self.wind[i][1]*random_ratio, self.wind[i][2]])
        dp = []

        for i in range(len(self.coords)) :
            if self.coords[i][1] + self.velocity[i][1] > random.uniform(90, 100):
                self.velocity[i] = [0.0, 0.0, 0.0]
                x = self.coords[i][0]
                z = self.coords[i][2]
                y = 18
                self.coords[i][1] = 18
                dp.append([x, y, z])
            else:
                dp.append([self.coords[i][0] + self.velocity[i][0], self.coords[i][1] + self.velocity[i][1], self.coords[i][2] + self.velocity[i][2]])

        # update position buffer on CPU, send to GPU attribute to draw with it
        coords = np.array(dp, 'f')
        super().draw(primitives, attributes=dict(position=coords), **uniforms)


class LavaAnimation(Mesh):
    """Animated particle set that simulates the movement of lava"""
    def __init__(self, shader):
        # render points with wide size to be seen
        GL.glPointSize(6)

        # instantiate and store 1000 points to animate
        self.coords = []
        number_of_particles = 1000
        for _ in range(number_of_particles):
            x = random.uniform(-4, 4)
            z = random.uniform(-np.sqrt(16 - x * x), np.sqrt(16 - x * x))
            y = random.uniform(18, 22)
            self.coords.append([x, y, z])

        speed = 10
        number_of_directions = 128
        relative_speed = 0
        initial_velocity_list = []
        for i in range(number_of_particles):
            if i%50 == 0:
                relative_speed += speed/20
            initial_velocity_list.append([relative_speed*np.cos((i%number_of_directions)*np.pi*2/number_of_directions), relative_speed, relative_speed*np.sin((i%number_of_directions)*np.pi*2/number_of_directions)])

        self.velocity = np.array(initial_velocity_list)

        # send as position attribute to GPU, set uniform variable global_color.
        # GL_STREAM_DRAW tells OpenGL that attributes of this object
        # will change on a per-frame basis (as opposed to GL_STATIC_DRAW)
        super().__init__(
            shader,
            attributes=dict(position=self.coords),
            usage=GL.GL_STREAM_DRAW,
            global_color=(1.0, 0.5, 0.0) # set the color to orange-red
        )

    def draw(self, primitives=GL.GL_POINTS, attributes=None, **uniforms):

        dp = []
        for i in range(len(self.coords)):
            if (self.coords[i][1] + self.velocity[i][1] * glfw.get_time() - 5*glfw.get_time()*glfw.get_time() > 0):
                dp.append([
                    self.coords[i][0] + self.velocity[i][0] * glfw.get_time(),
                    self.coords[i][1] + self.velocity[i][1] * glfw.get_time() - 5*glfw.get_time()*glfw.get_time(),
                    self.coords[i][2] + self.velocity[i][2] * glfw.get_time()
                ])
            else :
                dp.append([
                    self.coords[i][0] + self.velocity[i][0] * glfw.get_time(),
                    0,
                    self.coords[i][2] + self.velocity[i][2] * glfw.get_time()
                ])
                
        # update position buffer on CPU, send to GPU attribute to draw with it
        coords = np.array(dp, 'f')
        super().draw(primitives, attributes=dict(position=coords), **uniforms)


class Bird(Node):
    """ Very simple cylinder based on provided load function """
    def __init__(self, shader, file):
        super().__init__()
        self.add(*load(file, shader))  # just load cylinder from file





# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    shader = Shader("color.vert","color.frag")
    shader1 = Shader("skybox.vert", "skybox.frag")
    shader2 = Shader("smoke.vert","smoke.frag")
    shader3 = Shader("texture.vert","texture.frag")
    shader4 = Shader("bird3/bird.vert","bird3/bird.frag")
    shader5 = Shader("volcan.vert","volcan.frag")
    shader_lava = Shader("color_lava.vert", "color_lava.frag")



    viewer.add(SkyboxCube(shader1,['leftR.png', 'rightR.png', 'topR.png', 'bottomR.png', 'backR.png', 'frontR.png']))
    indices = np.array((1, 0, 2, 2, 0, 3), np.uint32)
    viewer.add(TexturedPlane(shader3,indices, "bottomR.png"))
    indices = np.array((1, 2, 0, 2, 3, 0), np.uint32)
    viewer.add(TexturedPlane(shader3,indices, "bottomR.png"))
    viewer.add(Volcano(shader5,"volcan.jpg","normal1.jpg"))
    viewer.add(*[mesh for file in sys.argv[1:] for mesh in load(file, shader4)])


    time_frames = np.linspace(0, 3, 1024)
    angles = np.linspace(-180, 1260, 1024)
    translate_keys = dict()
    rotate_keys = dict()
    scale_keys = dict()

    for i in range(1024):
        translate_keys[time_frames[i]] = vec(30*np.cos(angles[i]/180*np.pi), 30, 30*np.sin(angles[i]/180*np.pi))
        rotate_keys[time_frames[i]] = quaternion_from_euler(10, -angles[i], 0)
        scale_keys[time_frames[i]] = 0.5
    
    keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
    keynode.add(Bird(shader4,"bird3/bird.obj"))
    viewer.add(keynode)


    theta = 45.0       # base horizontal rotation angle
    phi1 = 0.0        # arm angle
    phi2 = 90.0 
    nombre_drones = 10

    for i in range(1,nombre_drones+1):
        # La base :
        transform =  scale(0.5) @ translate(0,1,-1) 
        base_shape = Node(children=[Cylinder(shader2,"drone.jpg")], transform=transform)

        # Arm :
        transform =  scale(0.1,1,0.1) @ translate(0,1,0) 
        arm_sahpe = Node(children=[Cylinder(shader2,"drone2.jpg")], transform=transform)

        # Forearm :
        transform =  scale(0.1,0.4,0.1) @ translate(0,1,0) 
        forearm_shape = Node(children=[Cylinder(shader2,"drone2.jpg")], transform=transform)
        
        i = np.random.choice([1, -1])
        j = np.random.choice([1, -1])
        x = np.random.uniform(6,16)*i
        y = np.random.uniform(14,23)
        z = np.random.uniform(6,16)*j

        transform_base = Node(transform=translate(x,y,z) @ rotate((0,1,0),theta))
        transform_arm = RotationControlNode(glfw.KEY_D, glfw.KEY_F, (0,1,0),phi1) 
        transform_forearm = Node(transform=translate(0,2,0) @ rotate((0,0,1),phi2))

        transform_forearm.add(forearm_shape)
        transform_arm.add(arm_sahpe,transform_forearm)
        transform_base.add(base_shape,transform_arm)

        node = RotationControlNode(glfw.KEY_LEFT, glfw.KEY_RIGHT, (0,1,0))
        node.add(transform_base)

        viewer.add(node)

    smoke_animation = PointAnimation(shader)
    lava_animation = LavaAnimation(shader_lava)
    viewer.add(smoke_animation)
    viewer.add(lava_animation)

    print("\n")
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("Appuyez sur 'espace' pour réinitialiser le temps à 0")
    print("Appuyez sur 'right' ou 'left' pour animer les drones autour du volcan")
    print("Appuyez sur 'D' ou 'F' pour animer les pales du rotor des drones")
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("\n")

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()                     # main function keeps variables locally scoped



















