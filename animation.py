# Python built-in modules
from bisect import bisect_left      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args

from core import Node
from transform import lerp, quaternion_slerp, quaternion_matrix,vec ,  translate, scale, identity


# -------------- Keyframing Utilities TP6 ------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """
        # 1. ensure time is within bounds else return boundary keyframe
        # 2. search for closest index entry in self.times, using bisect_left
        # 3. using the retrieved index, interpolate between the two neighboring
        # values in self.values, using the stored self.interpolate function
        
        n = len(self.times)
        if time <= self.times[0] :
            return self.values[0]
        elif time >= self.times[n-1] :
            return self.values[n-1]
        else :
            i = bisect_left(self.times,time)
            f = (time-self.times[i-1])/(self.times[i]-self.times[i-1])
            return self.interpolate(self.values[i-1],self.values[i],f)
        
class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.KeyFrameTranslate = KeyFrames(translate_keys)
        self.KeyFrameSclae = KeyFrames(scale_keys)
        self.KeyFrameRotate = KeyFrames(rotate_keys,quaternion_slerp)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        T = self.KeyFrameTranslate.value(time)
        R = self.KeyFrameRotate.value(time)
        S = self.KeyFrameSclae.value(time)

        
        R = quaternion_matrix(R)
        S = scale(S)
        T = translate(T)

        return T @ R @ S

        




class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, trans_keys, rot_keys, scale_keys, transform=identity()):
        super().__init__(transform=transform)
        self.keyframes = TransformKeyFrames(trans_keys, rot_keys, scale_keys)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(primitives=primitives, **uniforms)


# -------------- Linear Blend Skinning : TP7 ---------------------------------
class Skinned:
    """ Skinned mesh decorator, passes bone world transforms to shader """
    def __init__(self, mesh, bone_nodes, bone_offsets):
        self.mesh = mesh

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = np.array(bone_offsets, np.float32)

    def draw(self, **uniforms):
        world_transforms = [node.world_transform for node in self.bone_nodes]
        uniforms['bone_matrix'] = world_transforms @ self.bone_offsets
        self.mesh.draw(**uniforms)



if __name__ == "__main__":
    vector_keyframes = KeyFrames({0: np.array([1, 0, 0]), 3: np.array([0, 1, 0]), 6: np.array([0, 0, 1])})
    print(vector_keyframes.value(1.5)) 
    my_keyframes = KeyFrames({0: 1, 3: 7, 6: 20})
    print(my_keyframes.value(1.5))

    # Linear interpolation limitation :
    my_keyframes = KeyFrames({0:np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),1:np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])})
    print(my_keyframes.value(0.5))

    vector_keyframes = KeyFrames({0: vec(1, 0, 0), 3: vec(0, 1, 0), 6: vec(0, 0, 1)})
    print(vector_keyframes.value(1.5))   # should display numpy vector (0.5, 0.5, 0)