import OpenGL.GL as GL              # standard Python OpenGL wrapper
from PIL import Image              # load texture maps
import numpy as np

# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, tex_file, wrap_mode=GL.GL_REPEAT,
                 mag_filter=GL.GL_LINEAR, min_filter=GL.GL_LINEAR_MIPMAP_LINEAR,
                 tex_type=GL.GL_TEXTURE_2D):
        self.glid = GL.glGenTextures(1)
        self.type = tex_type
        try:
            # imports image as a numpy array in exactly right format
            tex = Image.open(tex_file).convert('RGBA')
            GL.glBindTexture(tex_type, self.glid)
            GL.glTexImage2D(tex_type, 0, GL.GL_RGBA, tex.width, tex.height,
                            0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex.tobytes())
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MIN_FILTER, min_filter)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MAG_FILTER, mag_filter)
            GL.glGenerateMipmap(tex_type)
            print(f'Loaded texture {tex_file} ({tex.width}x{tex.height}'
                  f' wrap={str(wrap_mode).split()[0]}'
                  f' min={str(min_filter).split()[0]}'
                  f' mag={str(mag_filter).split()[0]})')
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % tex_file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


class Skybox:
    """ Helper class to create and automatically destroy skyboxes """
    def __init__(self, tex_files, wrap_mode=GL.GL_CLAMP_TO_EDGE,
                 mag_filter=GL.GL_LINEAR, min_filter=GL.GL_LINEAR,
                 tex_type=GL.GL_TEXTURE_CUBE_MAP):
        self.glid = GL.glGenTextures(1)
        self.type = tex_type
        try:
            GL.glBindTexture(tex_type, self.glid)
            for i, tex_file in enumerate(tex_files):
                # imports image as a numpy array in exactly right format
                tex = Image.open(tex_file).convert('RGB')
                GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGB, tex.width, tex.height,
                                0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, tex.tobytes())
                print(f'Loaded texture {tex_file} ({tex.width}x{tex.height}'
                      f' wrap={str(wrap_mode).split()[0]}'
                      f' min={str(min_filter).split()[0]}'
                      f' mag={str(mag_filter).split()[0]})')
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MIN_FILTER, min_filter)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MAG_FILTER, mag_filter)
            print(f'Created Skybox with {len(tex_files)} textures')
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % tex_file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

# -------------- Textured mesh decorator --------------------------------------
class Textured:
    """ Drawable mesh decorator that activates and binds OpenGL textures """
    def __init__(self, drawable, **textures):
        self.drawable = drawable
        self.textures = textures

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        for index, (name, texture) in enumerate(self.textures.items()):
            GL.glActiveTexture(GL.GL_TEXTURE0 + index)
            GL.glBindTexture(texture.type, texture.glid)
            uniforms[name] = index
        self.drawable.draw(primitives=primitives, **uniforms)

class SkyboxMesh:
    """ Drawable mesh decorator that activates and binds OpenGL skyboxes """
    def __init__(self, drawable, skybox):
        self.drawable = drawable
        self.skybox = skybox

    def draw(self, **uniforms):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(self.skybox.type, self.skybox.glid)
        GL.glDepthMask(GL.GL_FALSE)
        uniforms['skybox'] = 0
        self.drawable.draw(**uniforms)
        GL.glDepthMask(GL.GL_TRUE)

