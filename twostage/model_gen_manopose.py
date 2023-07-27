import numpy as np

def get_bbox_from_pose(pose_2d, height=None, width=None, rate = 0.15):
    # this function returns bounding box from the 2D pose
    # here use pose_2d[:, -1] instead of pose_2d[:, 2]
    # because when vis reprojection, the result will be (x, y, depth, conf)
    validIdx = pose_2d[:, -1] > 0
    if validIdx.sum() == 0:
        return [0, 0, 100, 100, 0]
    y_min = int(min(pose_2d[validIdx, 1]))
    y_max = int(max(pose_2d[validIdx, 1]))
    x_min = int(min(pose_2d[validIdx, 0]))
    x_max = int(max(pose_2d[validIdx, 0]))
    # length = max(y_max - y_min, x_max - x_min)
    # center_x = (x_min + x_max) // 2
    # center_y = (y_min + y_max) // 2
    # y_min = center_y - length // 2
    # y_max = center_y + length // 2
    # x_min = center_x - length // 2
    # x_max = center_x + length // 2
    dx = (x_max - x_min)*rate
    dy = (y_max - y_min)*rate

    # 后面加上类别这些
    bbox = [x_min-dx, y_min-dy, x_max+dx, y_max+dy, 1]
    if height is not None and width is not None:
        bbox = [max(0, bbox[0]), max(0, bbox[1]), min(width - 1, bbox[2]), min(height - 1, bbox[3])]
    return bbox

def get_pose_shape(cur_frame, joints2d):
        single_flag = 0
        left_flag = 0
        right_flag = 0
        idx = 0
        total_time = 0
        end_time = start_time = 0

        height, width, _ = cur_frame.shape
        if joints2d['left'] is None or np.mean(joints2d['left'][:, 2]) < 0.2:
            left_flag = False
        else:
            left_points = joints2d['left'][:, :2]
            left_box = get_bbox_from_pose(left_points, height, width, rate=0.35)
            left_flag = True
        if joints2d['right'] is None or np.mean(joints2d['right'][:, 2]) < 0.2:
            right_flag = False
        else:
            right_points = joints2d['right'][:, :2]
            right_box = get_bbox_from_pose(right_points, height, width, rate=0.35)
            right_flag = True
        # right_box = [np.clip(min(right_points[:, 0]), 0, width), np.clip(min(right_points[:, 1]), 0, height),
        #              np.clip(max(right_points[:, 0]), 0, width), np.clip(max(right_points[:, 1]), 0, height)]

        if left_flag == False and right_flag == False:
            return

        out_img = cur_frame.copy()

        start_time = time.time()

        if left_flag and right_flag:
            if bbox_dist_ratio(left_box, right_box) == 0:
                single_flag = 1
            else:
                union_box = get_bbox_from_pose(np.concatenate((left_points, right_points), axis=0), height, width, rate=0.3)

                # track_path = 'track_{}'.format(track_id)
                # mkdir_with_check(os.path.join(crop_path, track_path))
                # mkdir_with_check(os.path.join(crop_ori_left, track_path))
                # mkdir_with_check(os.path.join(crop_ori_right, track_path))

                # if args.vis:
                #     cv2.imwrite(os.path.join(rgb_path, image_name), cur_frame)
                union_curbox, scale_union, union_bgbox, union_img, s2_union_img, s2_union_curbox, s2_union_bgbox = cropimg(
                    cur_frame,
                    union_box, )
                # os.path.join(
                #     crop_path,
                #     track_path,
                #     image_name))

        if left_flag:
            left_curbox, scale_left, left_bgbox, left_img, s2_left_img, s2_left_curbox, s2_left_bgbox = cropimg(cur_frame,
                                                                                                      left_box,)
                                                                                                      # os.path.join(
                                                                                                      #     crop_ori_left,
                                                                                                      #     track_path,
                                                                                                      #     image_name))

        if right_flag:
            right_curbox, scale_right, right_bgbox, right_img, s2_right_img, s2_right_curbox, s2_right_bgbox = cropimg(
                cur_frame, right_box,) #os.path.join(crop_ori_right, track_path, image_name))

        if left_flag and right_flag:
            if single_flag == 1:
                # if bbox_iou(s2_left_curbox, s2_right_curbox) > 0.02:
                #     overlap_box = bbox_intersection(s2_left_curbox, s2_right_curbox)
                #     left_overlap = Bbox(overlap_box.x1 - s2_left_curbox.x1 + s2_left_bgbox.x1,
                #                         overlap_box.y1 - s2_left_curbox.y1 + s2_left_bgbox.y1,
                #                         overlap_box.x2 - s2_left_curbox.x1 + s2_left_bgbox.x1,
                #                         overlap_box.y2 - s2_left_curbox.y1 + s2_left_bgbox.y1)###### overlap_box
                #     right_overlap = Bbox(overlap_box.x1 - s2_right_curbox.x1 + s2_right_bgbox.x1,
                #                         overlap_box.y1 - s2_right_curbox.y1 + s2_right_bgbox.y1,
                #                         overlap_box.x2 - s2_right_curbox.x1 + s2_right_bgbox.x1,
                #                         overlap_box.y2 - s2_right_curbox.y1 + s2_right_bgbox.y1)###### overlap_box

                proc_left = process_image(left_img)
                proc_right = process_image(right_img)
                inputs = {'img': torch.cat([proc_left, proc_right], dim=0).cuda()}
                out = mesh_predictor.run_mymodel(inputs['img'])
                pose_param_left = out['mano_pose_left'][0:1]
                pose_param_right = out['mano_pose_right'][1:2]
                shape_param_left = out['mano_shape_left'][0:1]
                shape_param_right = out['mano_shape_right'][1:2]
                v3d_left = out['v3d_left'][0:1]
                v3d_right = out['v3d_right'][1:2]
                outscale_left = out['scale_left'][0:1]
                outtrans2d_left = out['trans2d_left'][0:1]
                outscale_right = out['scale_right'][1:2]
                outtrans2d_right = out['trans2d_right'][1:2]
                scalelength_left = out['scalelength_left'][0:1]
                scalelength_right = out['scalelength_right'][1:2]

                ### smooth left params #################################################
                cur_left = Hand3dResult(
                    bbox=left_curbox,
                    global_orient=pose_param_left[0, :3],
                    poses=pose_param_left[0, 3:],
                    betas=shape_param_left[0],
                    camera_scale=outscale_left,
                    camera_tran=outtrans2d_left[0],
                    vertices=v3d_left[0],
                    scalelength=scalelength_left,
                )
                new_left = left_smoothcall(cur_left, prev_left_result)
                prev_left_result = new_left

                outscale_left = new_left.camera_scale
                outtrans2d_left = new_left.camera_tran[None, :]
                scalelength_left = new_left.scalelength
                #v3d_left = new_left.vertices[None, :]
                pose_param_left = torch.cat((new_left.global_orient[None, :],new_left.poses[None, :]), axis=1)
                shape_param_left = new_left.betas[None, :]
                v3d_left, j3d_left = mano_left_layer(rodrigues_batch(pose_param_left[:, :3]), pose_param_left[:, 3:], shape_param_left)
                v3d_left /= 1000
                v3d_left = v3d_left - j3d_left[:, 0:1, :] / 1000
                ####################################################################
                v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]),
                                                        pose_param_right[:, 3:],
                                                        shape_param_right)
                j3d_right = j3d_right - j3d_right[:, 0:1, :]
                j3d_right /= 1000
                j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right)
                pixel_length = (torch.tensor(
                    [right_curbox.x1 - left_curbox.x1, right_curbox.y1 - left_curbox.y1]).cuda().reshape(1, 2))
                root_rel = (((j2d_right[:, 0] /256 * scale_right + pixel_length) * 256 / scale_left - (outtrans2d_left * 128 + 128)) / (
                            outscale_left * 256)).reshape(1, 1, 2)
                # pixel_length = (torch.tensor(
                #     [right_prevbox.x1 - right_curbox.x1, right_prevbox.y1 - right_curbox.y1]).cuda().reshape(1, 2))
                # root_rel = (((j2d_rightprev + pixel_length) / scale_right * 256 - (outtrans2d_right * 128 + 128))/(outscale_right * 256)).reshape(1,1,2)
                #############################################
                # root_rel = (((cur_2dright_root - prev_2dright_root + j2d_rightprev + pixel_length) * 256 / scale_right - (
                #             outtrans2d_right * 128 + 128))/ (outscale_right * 256)).reshape(1,1,2)
                # prev_2dright_root = cur_2dright_root
                # right_prevbox = right_curbox

                root_right[:,:,:2] = root_rel
                ### smooth right params ###########################################
                cur_right = Hand3dResult(
                    bbox=right_curbox,
                    global_orient=pose_param_right[0, :3],
                    poses=pose_param_right[0, 3:],
                    betas=shape_param_right[0],
                    camera_scale=outscale_right,
                    camera_tran=outtrans2d_right[0],
                    vertices=v3d_right[0],
                    scalelength=scalelength_right,
                    rightrel=root_right[0,0]
                )
                new_right = right_smoothcall(cur_right, prev_right_result)
                prev_right_result = new_right

                outscale_right = new_right.camera_scale
                outtrans2d_right = new_right.camera_tran[None, :]
                scalelength_right = new_right.scalelength
                # v3d_right = new_right.vertices[None, :]
                pose_param_right = torch.cat((new_right.global_orient[None, :], new_right.poses[None, :]), axis=1)
                shape_param_right = new_right.betas[None, :]
                v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]), pose_param_right[:, 3:],
                                              shape_param_right)
                root_right = new_right.rightrel[None, None, :]
                v3d_right /= 1000
                v3d_right = v3d_right - j3d_right[:, 0:1, :] / 1000
                j3d_right = j3d_right - j3d_right[:, 0:1, :]
                j3d_right /= 1000
                j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right) * scale_right / 256
                j2d_rightprev = j2d_right[0, 0:1, :2]
                ###############################################################################
                if args.vis:
                    # img_left, img_right = rendering(render, outscale_left, outtrans2d_left, outscale_right, outtrans2d_right, v3d_left,
                    #                                 v3d_right, left_img=left_img, right_img=right_img, two=1, single=1)
                    # resize_img = cv2.resize(img_left, (scale_left, scale_left))
                    # out_img[left_curbox.y1:left_curbox.y2, left_curbox.x1:left_curbox.x2] = resize_img[left_bgbox.y1:left_bgbox.y2,
                    #                                                                         left_bgbox.x1:left_bgbox.x2, ]
                    #
                    # resize_img = cv2.resize(img_right, (scale_right, scale_right))
                    # out_img[right_curbox.y1:right_curbox.y2, right_curbox.x1:right_curbox.x2] = resize_img[right_bgbox.y1:right_bgbox.y2,
                    #                                                                             right_bgbox.x1:right_bgbox.x2, ]
                    img_left, img_right, left_hand, mask_left = rendering(s2_render, outscale_left* 0.5, outtrans2d_left* 0.5, outscale_right* 0.5,
                                                    outtrans2d_right* 0.5, v3d_left,
                                                    v3d_right, left_img=s2_left_img, right_img=s2_right_img, two=1,
                                                    single=1)
                    resize_img = cv2.resize(img_left, (2 * scale_left, 2 * scale_left))
                    resize_lefthand = cv2.resize(left_hand, (2 * scale_left, 2 * scale_left))
                    resize_leftmask = 1 - cv2.resize(mask_left, (2 * scale_left, 2 * scale_left))
                    out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] = resize_img[
                                                                                            s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                            s2_left_bgbox.x1:s2_left_bgbox.x2, ]

                    resize_img = cv2.resize(img_right, (2 * scale_right, 2 * scale_right))
                    out_img[s2_right_curbox.y1:s2_right_curbox.y2, s2_right_curbox.x1:s2_right_curbox.x2] = resize_img[
                                                                                                s2_right_bgbox.y1:s2_right_bgbox.y2,
                                                                                                s2_right_bgbox.x1:s2_right_bgbox.x2, ]

                    if classbox_iou(s2_left_curbox, s2_right_curbox) > 0.02:
                        out_img = out_img.astype(np.float32)
                        out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] *= resize_leftmask[
                                                                                                s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                s2_left_bgbox.x1:s2_left_bgbox.x2, None]
                        out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] += resize_lefthand[
                                                                                                s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                s2_left_bgbox.x1:s2_left_bgbox.x2, ]
                        out_img = out_img.astype(np.uint8)

                v3d_right = v3d_right + root_right.reshape(1, 1, 3)


                v3d_left *= scalelength_left
                v3d_right *= scalelength_right

            else:
                proc_union = process_image(union_img)
                inputs = {'img': proc_union.cuda()}
                targets = {}
                meta_info = {}
                out = mesh_predictor.run_mymodel(inputs['img'])
                pose_param_left = out['mano_pose_left'][0:1]
                pose_param_right = out['mano_pose_right'][0:1]
                shape_param_left = out['mano_shape_left'][0:1]
                shape_param_right = out['mano_shape_right'][0:1]
                v3d_left = out['v3d_left'][0:1]
                v3d_right = out['v3d_right'][0:1]
                outscale_left = out['scale_left']
                outtrans2d_left = out['trans2d_left']
                outscale_right = out['scale_right']
                outtrans2d_right = out['trans2d_right']
                scalelength_left = out['scalelength_left']
                scalelength_right = out['scalelength_right']
                right_rel = out['root_rel'].reshape(-1, 1, 3)
                root_right = right_rel

                ### smooth left params #################################################
                cur_left = Hand3dResult(
                    bbox=left_curbox,
                    global_orient=pose_param_left[0, :3],
                    poses=pose_param_left[0, 3:],
                    betas=shape_param_left[0],
                    camera_scale=outscale_left,
                    camera_tran=outtrans2d_left[0],
                    vertices=v3d_left[0],
                    scalelength=scalelength_left,
                )
                new_left = left_smoothcall(cur_left, prev_left_result)
                prev_left_result = new_left

                outscale_left = new_left.camera_scale
                outtrans2d_left = new_left.camera_tran[None, :]
                scalelength_left = new_left.scalelength
                # v3d_left = new_left.vertices[None, :]
                pose_param_left = torch.cat((new_left.global_orient[None, :], new_left.poses[None, :]), axis=1)
                shape_param_left = new_left.betas[None, :]
                v3d_left, j3d_left = mano_left_layer(rodrigues_batch(pose_param_left[:, :3]), pose_param_left[:, 3:],
                                              shape_param_left)
                v3d_left /= 1000
                v3d_left = v3d_left - j3d_left[:, 0:1, :] / 1000
                ####################################################################

                ### smooth right params ###########################################
                cur_right = Hand3dResult(
                    bbox=right_curbox,
                    global_orient=pose_param_right[0, :3],
                    poses=pose_param_right[0, 3:],
                    betas=shape_param_right[0],
                    camera_scale=outscale_right,
                    camera_tran=outtrans2d_right[0],
                    vertices=v3d_right[0],
                    scalelength=scalelength_right,
                    rightrel=root_right[0,0],
                )
                new_right = right_smoothcall(cur_right, prev_right_result)
                prev_right_result = new_right

                outscale_right = new_right.camera_scale
                outtrans2d_right = new_right.camera_tran[None, :]
                scalelength_right = new_right.scalelength
                root_right = new_right.rightrel[None, None, :]
                # v3d_right = new_right.vertices[None, :]
                pose_param_right = torch.cat((new_right.global_orient[None, :], new_right.poses[None, :]), axis=1)
                shape_param_right = new_right.betas[None, :]
                v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]), pose_param_right[:, 3:],
                                                shape_param_right)
                v3d_right /= 1000
                v3d_right = v3d_right - j3d_right[:, 0:1, :] / 1000

                j3d_right = j3d_right - j3d_right[:, 0:1, :]
                j3d_right /= 1000
                j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right) * scale_right / 256
                j2d_rightprev = j2d_right[0, 0:1, :2]
                ###############################################################################
                if args.vis:
                    # img = rendering(render, outscale_left, outtrans2d_left, outscale_right, outtrans2d_right,
                    #                                 v3d_left, v3d_right, union_img=union_img, two=1, single=0)
                    # resize_img = cv2.resize(img, (scale_union, scale_union))
                    # out_img[union_curbox.y1:union_curbox.y2, union_curbox.x1:union_curbox.x2] = resize_img[union_bgbox.y1:union_bgbox.y2, union_bgbox.x1:union_bgbox.x2,]

                    img = rendering(s2_render, outscale_left* 0.5, outtrans2d_left* 0.5, outscale_right* 0.5, outtrans2d_right* 0.5,
                                                    v3d_left, v3d_right, union_img=s2_union_img, two=1, single=0)
                    resize_img = cv2.resize(img, (2 * scale_union, 2 * scale_union))
                    out_img[s2_union_curbox.y1:s2_union_curbox.y2, s2_union_curbox.x1:s2_union_curbox.x2] = resize_img[s2_union_bgbox.y1:s2_union_bgbox.y2, s2_union_bgbox.x1:s2_union_bgbox.x2,]
                v3d_left *= scalelength_left
                v3d_right += root_right
                v3d_right *= scalelength_right
                right_prevbox = right_curbox

        elif left_flag:
            proc_left = process_image(left_img)
            inputs = {'img': proc_left.cuda()}
            out = mesh_predictor.run_mymodel(inputs['img'])
            pose_param_left = out['mano_pose_left'][0:1]
            shape_param_left = out['mano_shape_left'][0:1]
            v3d_left = out['v3d_left']
            outscale_left = out['scale_left']
            outtrans2d_left = out['trans2d_left']
            scalelength_left = out['scalelength_left'][0:1]

            ### smooth left params #################################################
            cur_left = Hand3dResult(
                bbox=left_curbox,
                global_orient=pose_param_left[0, :3],
                poses=pose_param_left[0, 3:],
                betas=shape_param_left[0],
                camera_scale=outscale_left,
                camera_tran=outtrans2d_left[0],
                vertices=v3d_left[0],
                scalelength=scalelength_left,
            )
            new_left = left_smoothcall(cur_left, prev_left_result)
            prev_left_result = new_left

            outscale_left = new_left.camera_scale
            outtrans2d_left = new_left.camera_tran[None, :]
            scalelength_left = new_left.scalelength
            # v3d_left = new_left.vertices[None, :]
            pose_param_left = torch.cat((new_left.global_orient[None, :], new_left.poses[None, :]), axis=1)
            shape_param_left = new_left.betas[None, :]
            v3d_left, j3d_left = mano_left_layer(rodrigues_batch(pose_param_left[:, :3]), pose_param_left[:, 3:],
                                          shape_param_left)
            v3d_left /= 1000
            v3d_left = v3d_left - j3d_left[:, 0:1, :] / 1000
            ####################################################################

            if args.vis:
                # img = rendering(render, outscale_left, outtrans2d_left, None, None,
                #                 v3d_left, None, left_img=left_img, two=0, single=0, left=1)
                # resize_img = cv2.resize(img, (scale_left, scale_left))
                # out_img[left_curbox.y1:left_curbox.y2, left_curbox.x1:left_curbox.x2] = resize_img[
                #                                                                         left_bgbox.y1:left_bgbox.y2,
                #                                                                         left_bgbox.x1:left_bgbox.x2, ]
                img = rendering(s2_render, outscale_left* 0.5, outtrans2d_left* 0.5,
                                                None,
                                                None, v3d_left,
                                                None, left_img=s2_left_img, two=0,
                                                single=0, left=1)
                resize_img = cv2.resize(img, (2 * scale_left, 2 * scale_left))
                out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] = resize_img[
                                                                                                    s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                    s2_left_bgbox.x1:s2_left_bgbox.x2, ]
            v3d_left *= scalelength_left

        elif right_flag:
            proc_right = process_image(right_img)
            inputs = {'img': proc_right.cuda()}
            out = mesh_predictor.run_mymodel(inputs['img'])
            pose_param_right = out['mano_pose_right']
            shape_param_right = out['mano_shape_right']
            v3d_right = out['v3d_right']
            # v3d_right = out['mesh_coord_cam_right']
            outscale_right = out['scale_right']
            outtrans2d_right = out['trans2d_right']
            scalelength_right = out['scalelength_right']
            ### smooth right params ###########################################
            cur_right = Hand3dResult(
                bbox=right_curbox,
                global_orient=pose_param_right[0, :3],
                poses=pose_param_right[0, 3:],
                betas=shape_param_right[0],
                camera_scale=outscale_right,
                camera_tran=outtrans2d_right[0],
                vertices=v3d_right[0],
                scalelength=scalelength_right,
                rightrel=root_right[0,0],
            )
            new_right = right_smoothcall(cur_right, prev_right_result)
            prev_right_result = new_right

            outscale_right = new_right.camera_scale
            outtrans2d_right = new_right.camera_tran[None, :]
            scalelength_right = new_right.scalelength
            # v3d_right = new_right.vertices[None, :]
            pose_param_right = torch.cat((new_right.global_orient[None, :], new_right.poses[None, :]), axis=1)
            shape_param_right = new_right.betas[None, :]
            root_right = new_right.rightrel[None, None, :]
            v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]), pose_param_right[:, 3:],
                                            shape_param_right)
            v3d_right /= 1000
            v3d_right = v3d_right - j3d_right[:, 0:1, :] / 1000

            j3d_right = j3d_right - j3d_right[:, 0:1, :]
            j3d_right /= 1000
            j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right) * scale_right / 256
            j2d_rightprev = j2d_right[0, 0:1, :2]
            ###############################################################################
            if args.vis:
                # img = rendering(render, None, None, outscale_right, outtrans2d_right,
                #                 None, v3d_right, right_img=right_img, two=0, single=0, left=0, right=1)
                # resize_img = cv2.resize(img, (scale_right, scale_right))
                # out_img[right_curbox.y1:right_curbox.y2, right_curbox.x1:right_curbox.x2] = resize_img[
                #                                                                             right_bgbox.y1:right_bgbox.y2,
                #                                                                             right_bgbox.x1:right_bgbox.x2, ]
                img1 = rendering(s2_render, None, None, outscale_right* 0.5, outtrans2d_right* 0.5,
                                None, v3d_right, right_img=s2_right_img, two=0, single=0, left=0, right=1)
                resize_img = cv2.resize(img1, (2 * scale_right, 2 * scale_right))
                # tmp = cur_frame.copy()
                out_img[s2_right_curbox.y1:s2_right_curbox.y2, s2_right_curbox.x1:s2_right_curbox.x2] = resize_img[
                                                                                            s2_right_bgbox.y1:s2_right_bgbox.y2,
                                                                                      s2_right_bgbox.x1:s2_right_bgbox.x2, ]
            v3d_right += root_right
            v3d_right *= scalelength_right