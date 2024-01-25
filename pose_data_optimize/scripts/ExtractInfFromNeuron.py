from FbxCommon import *
from .util import *
def ListNegation(lis):
    lis_c = lis
    for idx in range(len(lis_c)):
        lis_c[idx] = -lis_c[idx]
    return lis_c

class AnimationLayerInf():
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.key_inf = {'TX':[[],[]], #key_time. key_value
                        'TY':[[],[]],
                        'TZ':[[],[]],
                        'RX':[[],[]],
                        'RY':[[],[]],
                        'RZ':[[],[]],}
    def ExtractLayerInf(self, pScene, pAnimLayer, bone_node):
        KFCURVENODE_T_X = "X"
        KFCURVENODE_T_Y = "Y"
        KFCURVENODE_T_Z = "Z"

        KFCURVENODE_R_X = "X"
        KFCURVENODE_R_Y = "Y"
        KFCURVENODE_R_Z = "Z"
        KFCURVENODE_R_W = "W"

        # Display general curves.
        lAnimCurve = bone_node.LclTranslation.GetCurve(pAnimLayer, KFCURVENODE_T_X)
        if lAnimCurve:
            self.ExractCurveInf(lAnimCurve, 'TX')
        lAnimCurve = bone_node.LclTranslation.GetCurve(pAnimLayer, KFCURVENODE_T_Y)
        if lAnimCurve:
            self.ExractCurveInf(lAnimCurve, 'TY')
        lAnimCurve = bone_node.LclTranslation.GetCurve(pAnimLayer, KFCURVENODE_T_Z)
        if lAnimCurve:
            self.ExractCurveInf(lAnimCurve, 'TZ')

        lAnimCurve = bone_node.LclRotation.GetCurve(pAnimLayer, KFCURVENODE_R_X)
        if lAnimCurve:
            self.ExractCurveInf(lAnimCurve, 'RX')
        lAnimCurve = bone_node.LclRotation.GetCurve(pAnimLayer, KFCURVENODE_R_Y)
        if lAnimCurve:
            self.ExractCurveInf(lAnimCurve, 'RY')
        lAnimCurve = bone_node.LclRotation.GetCurve(pAnimLayer, KFCURVENODE_R_Z)
        if lAnimCurve:
            self.ExractCurveInf(lAnimCurve, 'RZ')
        self.AxisConvert(bone_node)

    def ExractCurveInf(self, pCurve, target):
        lKeyCount = pCurve.KeyGetCount()
        for lCount in range(lKeyCount):
            self.key_inf[target][0].append(pCurve.KeyGetTime(lCount).GetSecondDouble())
            self.key_inf[target][1].append(pCurve.KeyGetValue(lCount))

    def AxisConvert(self, bone_node):
        # this code is not very proper
        # the neuron is euler ZYX (global axis, RX_RY_RZ)
        # but unreal is euler XYZ
        frame_idx = 0
        if len(self.key_inf['RX'][0]) == 0:
            return
        while frame_idx < len(self.key_inf['RX'][0]):
            roll = self.key_inf['RX'][1][frame_idx]
            pitch = self.key_inf['RY'][1][frame_idx]
            yaw = self.key_inf['RZ'][1][frame_idx]
            R = eulerAngles2rotationMat([roll, pitch, yaw], [], 'degree', order='ZYX', axis='right')
            if bone_node.GetName().find('hand_l') > -1:
                # R = R.dot(r_refine_rot)
                R = l_refine_rot.dot(AxisConvertFrom3dmax2Axis(R))
                # R = rot_x(90).dot(R)
            elif bone_node.GetName().find('hand_r') > -1:
                R = r_refine_rot.dot(AxisConvertFrom3dmax2Axis(R))
                # R = rot_x(90).dot(R)
            x,y,z = rotationMatrixToEulerAngles(R)
            self.key_inf['RX'][1][frame_idx] = x
            self.key_inf['RY'][1][frame_idx] = y
            self.key_inf['RZ'][1][frame_idx] = z
            frame_idx += 1
        # (X,Y,Z) -> (-X,-Z,-Y)
        # print self.key_inf['RX'][1]
        # self.key_inf['RX'][1] = ListNegation(self.key_inf['RX'][1])
        #
        # ry_key_time = self.key_inf['RY'][0]
        # ry_key_value = self.key_inf['RY'][1]
        # self.key_inf['RY'][0] = self.key_inf['RZ'][0]
        # self.key_inf['RY'][1] = self.key_inf['RZ'][1]
        #
        # self.key_inf['RZ'][0] = ry_key_time
        # self.key_inf['RZ'][1] = ListNegation(ry_key_value)
        # self.key_inf['RY'][1] = ListNegation(self.key_inf['RY'][1])
        # self.key_inf['RZ'][1] = ListNegation(self.key_inf['RZ'][1])

class AnimationStackInf():
    def __init__(self, stack_name):
        self.stack_name = stack_name
        self.animation_layers = []
    def append_animation_layers(self, layer):
        self.animation_layers.append(layer)

class BoneInf():
    def __init__(self, bone_name):
        self.name = bone_name
        self.children = []
        self.animation_stacks = []
    def append_child(self, child_bone):
        self.children.append(child_bone)
    def append_animation_stack(self, stack):
        self.animation_stacks.append(stack)

def GetBoneInf(pScene, node = None):
    if node is None:
        lRootNode = pScene.GetRootNode()
    else:
        lRootNode = node
    root_bone_inf = BoneInf(lRootNode.GetName())
    ExtractBoneAnimation(pScene, lRootNode, root_bone_inf)
    for i in range(lRootNode.GetChildCount()):
        root_bone_inf.append_child(GetBoneInf(pScene, lRootNode.GetChild(i)))
    return root_bone_inf

def OnlyPreserveHandAndRoot(bone_inf, preserved_bones):
    if bone_inf.name.find('root') > -1 or bone_inf.name.find('hand') > -1:
        preserved_bones.append(bone_inf)
    for i in range(bone_inf.children.__len__()):
        OnlyPreserveHandAndRoot(bone_inf.children[i], preserved_bones)

def OnlyPreserveHandAndRootAnim(bone_inf):
    if bone_inf.name.find('hand') > -1:
        return
    if bone_inf.name.find('root') == -1:
        for stack in bone_inf.animation_stacks:
            for layer in stack.animation_layers:
                layer.key_inf = {'TX':[[],[]], #key_time. key_value
                        'TY':[[],[]],
                        'TZ':[[],[]],
                        'RX':[[],[]],
                        'RY':[[],[]],
                        'RZ':[[],[]],}
    for i in range(bone_inf.children.__len__()):
        OnlyPreserveHandAndRootAnim(bone_inf.children[i])

def ExtractBoneAnimation(pScene, bone_node, root_bone_inf):
    for i in range(pScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))):
        lAnimStack = pScene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
        anim_stack_inf = AnimationStackInf(lAnimStack.GetName())
        nbAnimLayers = lAnimStack.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId))
        for l in range(nbAnimLayers):
            lAnimLayer = lAnimStack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), l)
            anim_layer_inf = AnimationLayerInf(lAnimLayer.GetName())
            anim_layer_inf.ExtractLayerInf(pScene, lAnimLayer, bone_node)

            anim_stack_inf.append_animation_layers(anim_layer_inf)
        root_bone_inf.append_animation_stack(anim_stack_inf)

def ExtractInfFromFbx(fbx_name):
    # Prepare the FBX SDK.
    lSdkManager, lScene = InitializeSdkObjects()
    # Load the scene.
    lResult = LoadScene(lSdkManager, lScene, fbx_name)
    print("\n\nUsage: ImportScene <FBX file name>\n")
    bones_inf = GetBoneInf(pScene=lScene, node=None)
    preserved_bones = []
    OnlyPreserveHandAndRoot(bones_inf, preserved_bones)
    bones_inf = preserved_bones[0]
    bones_inf.children = []
    idx = 1
    while idx < preserved_bones.__len__():
        bones_inf.append_child(preserved_bones[idx])
        print(preserved_bones[idx].name)
        print(preserved_bones[idx].animation_stacks[0].animation_layers[0].key_inf['RX'][1])
        print(preserved_bones[idx].animation_stacks[0].animation_layers[0].key_inf['RY'][1])
        print(preserved_bones[idx].animation_stacks[0].animation_layers[0].key_inf['RZ'][1])
        idx += 1
    # OnlyPreserveHandAndRootAnim(bones_inf)
    # SaveScene(lSdkManager, lScene, "xx.fbx")
    lSdkManager.Destroy()
    return bones_inf
if __name__ == "__main__":
    ExtractInfFromFbx("/home/tallery/fbx_sdk/samples/MY/take002_chr03_UNREAL.fbx")
    sys.exit(0)