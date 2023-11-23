import bpy
from math import radians

from mathutils import Matrix
from mathutils import Vector
import pickle

import numpy
import sys
import random
import os


def add_hand(idx,hand_type,imgpath):

    # file_loc = f'/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/nodup_pose/{idx}_{hand_type}.obj'
    file_loc = f'/nvme/lijun/dataset/renderih/nodup_sample/{idx}_{hand_type}.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    obj_object = bpy.context.selected_objects[0] ####<--Fix
    #print('Imported name: ', obj_object.name)
    obj_object.rotation_euler = (radians(270), 0, 0)

    #add a new "SimpleSubsurf" modifier
    mod = obj_object.modifiers.new('SimpleSubsurf', 'SUBSURF')
    mod.subdivision_type = "CATMULL_CLARK"
    mod.levels = 3
    mod.render_levels=3

    #add texture
    mat = obj_object.material_slots[0].material

    ### load texture ###
    img = bpy.data.images.load(imgpath)
    mat.use_nodes=True

    #setup the node_tree and links as you would manually on shader Editor
    #to define an image texture for a material

    material_output = mat.node_tree.nodes.get('Material Output')
    principled_BSDF = mat.node_tree.nodes.get('Principled BSDF')
    principled_BSDF.inputs['Roughness'].default_value=1 #Roughness
    principled_BSDF.subsurface_method='BURLEY'
    principled_BSDF.inputs['Subsurface Color'].default_value=(0.8, 0.43, 0.411,1)
    principled_BSDF.inputs['Subsurface'].default_value=0.016


    tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_node.image = img

    mat.node_tree.links.new(tex_node.outputs[0], principled_BSDF.inputs[0])


def world(world_path):
    C = bpy.context
    scn = C.scene

    # Get the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    #node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property


    world_list=os.listdir(world_path)

    rand=random.randint(0,len(world_list)-1)
    print('environment:',rand)
    world_path=f'{world_path}{world_list[rand]}/'
    world_list=os.listdir(world_path)

    rand=random.randint(0,len(world_list)-1) ### change llj ###

    node_environment.image = bpy.data.images.load(f'{world_path}{world_list[rand]}') # Relative path
    node_environment.location = -300,0



    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    # Link all nodes
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_output.inputs["Surface"])
    #link = links.new(node_background.outputs["Background"], )




# Input: P 3x4 numpy matrix
# Output: K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
#
# Reference implementations:
#   - Oxford's visual geometry group matlab toolbox
#   - Scilab Image Processing toolbox
def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)

    K /= K[-1,-1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = numpy.diag(numpy.sign(numpy.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if (numpy.linalg.det(R) < 0):
       R = -R
    # C = -H\P[:,-1]
    C = numpy.linalg.lstsq(-H, P[:,-1])[0]
    T = -R*C
    return K, R, T

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations:
#   Oxford's visual geometry group matlab toolbox
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = numpy.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (numpy.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    return r, q

# Creates a blender camera consistent with a given 3x4 computer vision P matrix
# Run this in Object Mode
# scale: resolution scale percentage as in GUI, known a priori
# P: numpy 3x4
def get_blender_camera_from_3x4_P(P, scale,K1):
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(numpy.matrix(P))
    #print("======cal=========")
    #print(K,R_world2cv, T_world2cv)

    # K[0,2]=167
    # K[1,2]=256
    K=numpy.array(K1)

    scene = bpy.context.scene
    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0,0] / s_u
    # recover original resolution
    scene.render.resolution_x = int(resolution_x_in_px / scale)
    scene.render.resolution_y = int(resolution_y_in_px / scale)
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # http://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
    # R_bcam2cv = Matrix(
    #     ((-1, 0,  0),
    #      (0, 1, 0),
    #      (0, 0, 1)))

    R_cv2world = R_world2cv.T
    rotation =  Matrix(R_cv2world.tolist()) * R_bcam2cv
    location = -R_cv2world * T_world2cv

    #print('location:',location)

    # create a new camera
    bpy.ops.object.add(
        type='CAMERA',) #### llj
        #location=location)
    ob = bpy.context.object
    ob.name = 'CamFrom3x4PObj'
    cam = ob.data
    cam.name = 'CamFrom3x4P'

    # Lens
    cam.type = 'PERSP'
    cam.lens = f_in_mm
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width  = sensor_width_in_mm
    #lllj
    #ob.matrix_world = Matrix.Translation(location)*rotation.to_4x4()
    #print('lens:',cam.lens)

#    obj_object = bpy.context.selected_objects[0] ####<--Fix
    #print('Imported name: ', obj_object.name)
    ob.rotation_euler = (radians(90), 0, 0)

    #     cam.shift_x = -0.05
    #     cam.shift_y = 0.1
    #     cam.clip_start = 10.0
    #     cam.clip_end = 250.0
    #     empty = bpy.data.objects.new('DofEmpty', None)
    #     empty.location = origin+Vector((0,10,0))
    #     cam.dof_object = empty

    # Display
    cam.show_name = True
    # Make this the current camera
    scene.camera = ob
#    bpy.context.scene.update()

if __name__ == "__main__":
#load pickle data
    idx=int(sys.argv[12])

    texture_path='/path/to/texture/' 
    world_path='/path/to/HDRI/'
    with open(f'/path/to/pkl/', 'rb') as f:
    #with open(f'/nvme/lijun/dataset/renderih/nodup_sample/{idx}.pkl', 'rb') as f:
        data = pickle.load(f)
        
    R_world2cv=Matrix(data['camera']['R'])
    T_world2cv=Vector(data['camera']['t'][0])
    K=Matrix(data['camera']['camera'])


    scale=1

    #print("=======Pickle File=====")
#    print(K,R_world2cv,T_world2cv)
    RT1 = Matrix((
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],)
             ))
    P=K@RT1
#    print(K)

    #add camera
    get_blender_camera_from_3x4_P(P, scale,K)

    #add hand
    # color=random.randint(0,1)

    # texture_path='/home/lilijun/nvme/dataset/renderih/texture/'

    texture_list=os.listdir(texture_path)
    rand=random.randint(0,len(texture_list)-1) ### change llj ###

    imgpath=f'{texture_path}{texture_list[rand]}'


    add_hand(idx,'left',imgpath)
    add_hand(idx,'right',imgpath)
    #print('color:',imgpath)

    #add environment texture
    world(world_path=world_path)

    # render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.device='GPU'
    # bpy.context.scene.view_settings.look = 'High Contrast'

#for area in bpy.context.screen.areas:
#    if area.type == 'VIEW_3D':
#        area.spaces[0].viewport_shade = 'RENDERED'

#i=0
#bpy.context.scene.render.image_settings.file_format='PNG'
#bpy.context.scene.render.filepath = ".pic%0.2d.png"%i
#bpy.ops.render.render(use_viewport = True, write_still=True)
