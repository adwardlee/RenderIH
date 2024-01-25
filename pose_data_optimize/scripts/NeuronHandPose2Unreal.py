import sys
import yaml
import os
from ExtractInfFromNeuron import *
neuron_FBX_filename = "data/take003_chr03_UNREAL.fbx"
template_FBX_filename = "data/SK_Mannequin.FBX"
template_FBX_filename = ""
output_FBX_filename = 'Hand.fbx'
hand_root_translation_label = "KeyPointTrabsform.yaml"
hand_root_trans = {}
FPS = 30
def LoadHandRootTranslation():
    return
    f = open(hand_root_translation_label)
    keypoint_file = yaml.safe_load(f)
    num_frames = keypoint_file.keys()[-1] + 1
    key_frame_idx = 1
    hand_root_trans['hand_r'] = [[], [], []]
    hand_root_trans['hand_l'] = [[], [], []]
    while key_frame_idx < num_frames:
        loc_dict = keypoint_file[key_frame_idx]['hand_r']['Translation'][0]
        kp_loc = [loc_dict['X'], loc_dict['Y'], loc_dict['Z']]
        hand_root_trans['hand_r'][0].append(kp_loc[0])
        hand_root_trans['hand_r'][1].append(kp_loc[1])
        hand_root_trans['hand_r'][2].append(kp_loc[2])

        hand_root_trans['hand_l'][0].append(kp_loc[0]*2)
        hand_root_trans['hand_l'][1].append(kp_loc[1]*3)
        hand_root_trans['hand_l'][2].append(kp_loc[2]*1)
        key_frame_idx += 1

def CreateSkeletonAccordingtoNeuron(bones_inf):
    basic_skeleton = bones_inf.name
    lSkeletonBasicAttribute = FbxSkeleton.Create(lSdkManager, basic_skeleton)
    lSkeletonBasicAttribute.SetSkeletonType(FbxSkeleton.eRoot)
    lSkeletonBasic = FbxNode.Create(lSdkManager, basic_skeleton)
    lSkeletonBasic.SetNodeAttribute(lSkeletonBasicAttribute)
    lSkeletonBasic.LclTranslation.Set(FbxDouble3(0.0, 0.0, 0.0))
    for child_bone in bones_inf.children:
        lSkeletonBasic.AddChild(CreateSkeletonAccordingtoNeuron(child_bone))
    return lSkeletonBasic

def BonesInftoDict(bones_inf, bones_inf_dict):
    bones_inf_dict[bones_inf.name] = bones_inf
    for i in range(bones_inf.children.__len__()):
        BonesInftoDict(bones_inf.children[i], bones_inf_dict)

def AnimateHandRoot(bone_node, axis_id, lAnimCurve):
    lAnimCurve.KeyModifyBegin()
    lTime = FbxTime()
    frame_id = 0
    for value in hand_root_trans[bone_node.GetName()][axis_id]:
        lTime.SetSecondDouble(frame_id * 1.0/FPS)
        lKeyIndex = lAnimCurve.KeyAdd(lTime)[0]
        lAnimCurve.KeySetValue(lKeyIndex, value)
        lAnimCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)
        frame_id += 1
def AnimateSkeletonAccordingtoNeuron(bone_node, target_axis_animation_inf, lAnimCurve):
    num_frames = target_axis_animation_inf[0].__len__()
    if num_frames == 0:
        return
    lAnimCurve.KeyModifyBegin()
    lTime = FbxTime()
    # animation_inf = bones_inf_dict[bone_node.GetName()].animation_stacks[stack_id].animation_layers[layer_id][target_axis]
    key_frame_idx = 0
    while key_frame_idx < num_frames:
        lTime.SetSecondDouble(target_axis_animation_inf[0][key_frame_idx])
        lKeyIndex = lAnimCurve.KeyAdd(lTime)[0]

        lAnimCurve.KeySetValue(lKeyIndex, target_axis_animation_inf[1][key_frame_idx])
        lAnimCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)
        key_frame_idx += 1
    lAnimCurve.KeyModifyEnd()

def AnimateChildSkeleton(bone_node, bones_inf_dict, lAnimLayer, stack_id, layer_id):
    # bone_node.SetRotationOrder(FbxNode.eSourcePivot, eEulerZYX)
    bone_node.SetRotationOrder(FbxNode.eSourcePivot, eEulerZYX)
    if bone_node.GetName() not in bones_inf_dict.keys():
        print(bone_node.GetName())
    else:
        animation_inf = \
            bones_inf_dict[bone_node.GetName()].animation_stacks[stack_id].animation_layers[layer_id].key_inf
        if True or not bone_node.GetName().find('hand_') > -1:
        # if bone_node.GetName().find('hand') > -1:
        #     print bone_node.GetName()
        #     lAnimCurve = bone_node.LclTranslation.GetCurve(lAnimLayer, 'X', True)
        #     AnimateHandRoot(bone_node, 0, lAnimCurve)
        #     lAnimCurve = bone_node.LclTranslation.GetCurve(lAnimLayer, 'Y', True)
        #     AnimateHandRoot(bone_node, 1, lAnimCurve)
        #     lAnimCurve = bone_node.LclTranslation.GetCurve(lAnimLayer, 'Z', True)
        #     AnimateHandRoot(bone_node, 2, lAnimCurve)
        # else:
        #     lAnimCurve = bone_node.LclTranslation.GetCurve(lAnimLayer, 'X', True)
        #     AnimateSkeletonAccordingtoNeuron(bone_node, animation_inf['TX'], lAnimCurve)
        #     lAnimCurve = bone_node.LclTranslation.GetCurve(lAnimLayer, 'Y', True)
        #     AnimateSkeletonAccordingtoNeuron(bone_node, animation_inf['TY'], lAnimCurve)
        #     lAnimCurve = bone_node.LclTranslation.GetCurve(lAnimLayer, 'Z', True)
        #     AnimateSkeletonAccordingtoNeuron(bone_node, animation_inf['TZ'], lAnimCurve)
            lAnimCurve = bone_node.LclRotation.GetCurve(lAnimLayer, 'X', True)
            AnimateSkeletonAccordingtoNeuron(bone_node, animation_inf['RX'], lAnimCurve)
            lAnimCurve = bone_node.LclRotation.GetCurve(lAnimLayer, 'Y', True)
            AnimateSkeletonAccordingtoNeuron(bone_node, animation_inf['RY'], lAnimCurve)
            lAnimCurve = bone_node.LclRotation.GetCurve(lAnimLayer, 'Z', True)
            AnimateSkeletonAccordingtoNeuron(bone_node, animation_inf['RZ'], lAnimCurve)
    for i in range(bone_node.GetChildCount()):
        AnimateChildSkeleton(bone_node.GetChild(i), bones_inf_dict, lAnimLayer, stack_id, layer_id)

def AnimateSkeleton(pSdkManager, pScene, pSkeletonRoot, bones_inf_dict):
    for stack_inf_idx in range(bones_inf_dict['root'].animation_stacks.__len__()):
        stack_inf = bones_inf_dict['root'].animation_stacks[stack_inf_idx]
        lAnimStackName = stack_inf.stack_name
        lAnimStack = FbxAnimStack.Create(pScene, lAnimStackName)
        for layer_inf_idx in range(stack_inf.animation_layers.__len__()):
            layer_inf = stack_inf.animation_layers[layer_inf_idx]
            lAnimLayer = FbxAnimLayer.Create(pScene, layer_inf.layer_name)
            lAnimStack.AddMember(lAnimLayer)
            AnimateChildSkeleton(pSkeletonRoot, bones_inf_dict, lAnimLayer, stack_inf_idx, layer_inf_idx)

    # keypoints, num_frames = LoadPoseYaml(keypoint_file_name)
    # lKeyIndex = 0
    # lTime = FbxTime()
    #
    # lRoot = pSkeletonRoot.GetChild(0)
    #
    # # First animation stack.
    # lAnimStackName = "main anim"
    # lAnimStack = FbxAnimStack.Create(pScene, lAnimStackName)
    #
    # # The animation nodes can only exist on AnimLayers therefore it is mandatory to
    # # add at least one AnimLayer to the AnimStack. And for the purpose of this example,
    # # one layer is all we need.
    # lAnimLayer = FbxAnimLayer.Create(pScene, "Base Layer")
    # lAnimStack.AddMember(lAnimLayer)
    #
    # # Create the AnimCurve on the Rotation.Z channel
    # # lRoot.SetRotationOrder(FbxNode.eSourcePivot, eEulerXYZ)
    # lCurve_roll = lRoot.LclRotation.GetCurve(lAnimLayer, "X", True)
    # lCurve_pitch = lRoot.LclRotation.GetCurve(lAnimLayer, "Y", True)
    # lCurve_yaw = lRoot.LclRotation.GetCurve(lAnimLayer, "Z", True)
    # lCurve_tx = lRoot.LclTranslation.GetCurve(lAnimLayer, "X", True)
    # lCurve_ty = lRoot.LclTranslation.GetCurve(lAnimLayer, "Y", True)
    # lCurve_tz = lRoot.LclTranslation.GetCurve(lAnimLayer, "Z", True)
    #
    # # KeyAddToCurve(lCurve_roll, 'R', 0, keypoints, num_frames)
    # # KeyAddToCurve(lCurve_pitch, 'R', 1, keypoints, num_frames)
    # # KeyAddToCurve(lCurve_yaw, 'R', 2, keypoints, num_frames)
    # # KeyAddToCurve(lCurve_tx, 'T', 0, keypoints, num_frames)
    # # KeyAddToCurve(lCurve_ty, 'T', 1, keypoints, num_frames)
    # # KeyAddToCurve(lCurve_tz, 'T', 2, keypoints, num_frames)

def LoadCreatScene(pSdkManager, pScene, file_name):
    lResult = LoadScene(lSdkManager, lScene, file_name)
    lSkeletonRoot = lScene.GetRootNode()
    neuron_bone_inf = ExtractInfFromFbx(neuron_FBX_filename)
    LoadHandRootTranslation()
    bones_inf_dict = {}
    BonesInftoDict(neuron_bone_inf, bones_inf_dict)
    AnimateSkeleton(pSdkManager, pScene, lSkeletonRoot, bones_inf_dict)
    return lResult

def CreateScene(pSdkManager, pScene):
    # Create scene info
    lSceneInfo = FbxDocumentInfo.Create(pSdkManager, "SceneInfo")
    lSceneInfo.mTitle = "object motion scene"
    lSceneInfo.mSubject = "Object's animation captured by mocap."
    lSceneInfo.mAuthor = "Linrui Tian."
    lSceneInfo.mRevision = "rev. 1.0"
    lSceneInfo.mKeywords = "mocap"
    lSceneInfo.mComment = "no particular comments required."
    pScene.SetSceneInfo(lSceneInfo)

    neuron_bone_inf = ExtractInfFromFbx(neuron_FBX_filename)
    LoadHandRootTranslation()
    # lPatchNode = CreatePatch(pSdkManager, "Patch")
    lSkeletonRoot = CreateSkeletonAccordingtoNeuron(neuron_bone_inf)
    bones_inf_dict = {}
    BonesInftoDict(neuron_bone_inf, bones_inf_dict)
    # pScene.GetRootNode().AddChild(lPatchNode)
    pScene.GetRootNode().AddChild(lSkeletonRoot)

    # StoreBindPose(lSdkManager, lScene, lPatchNode, lSkeletonRoot)
    # StoreRestPose(lSdkManager, lScene, lSkeletonRoot)

    AnimateSkeleton(pSdkManager, pScene, lSkeletonRoot, bones_inf_dict)


if __name__ == "__main__":
    try:
        import FbxCommon
        from fbx import *
    except ImportError:
        print("Error: module FbxCommon and/or fbx failed to import.\n")
        print(
            "Copy the files located in the compatible sub-folder lib/python<version> into your python interpreter site-packages folder.")
        import platform

        if platform.system() == 'Windows' or platform.system() == 'Microsoft':
            print('For example: copy ..\\..\\lib\\Python27_x64\\* C:\\Python27\\Lib\\site-packages')
        elif platform.system() == 'Linux':
            print('For example: cp ../../lib/Python27_x64/* /usr/local/lib/python2.7/site-packages')
        elif platform.system() == 'Darwin':
            print(
                'For example: cp ../../lib/Python27_x64/* /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
        sys.exit(1)

    # Prepare the FBX SDK.
    (lSdkManager, lScene) = FbxCommon.InitializeSdkObjects()

    # Create the scene.
    if template_FBX_filename != "":
        lResult = LoadCreatScene(lSdkManager, lScene, template_FBX_filename)
    else:
        lResult = CreateScene(lSdkManager, lScene)

    if lResult == False:
        print("\n\nAn error occurred while creating the scene...\n")
        lSdkManager.Destroy()
        sys.exit(1)

    lResult = FbxCommon.SaveScene(lSdkManager, lScene, output_FBX_filename)
    print(output_FBX_filename)
    if lResult == False:
        print("\n\nAn error occurred while saving the scene...\n")
        lSdkManager.Destroy()
        sys.exit(1)

    # Destroy all objects created by the FBX SDK.
    lSdkManager.Destroy()

    sys.exit(0)