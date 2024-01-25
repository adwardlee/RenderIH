import numpy as np


def create_vertex_color(contact_info, mode="vertex_contact"):
    if mode == "vertex_contact":
        vertex_contact = contact_info["vertex_contact"]
        n_verts = vertex_contact.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[vertex_contact == 0] = np.array([57, 57, 57]) / 255.0
        vertex_color[vertex_contact == 1] = np.array([198, 198, 198]) / 255.0
        return vertex_color
    elif mode == "contact_region":
        contact_region = contact_info["hand_region"]
        n_verts = contact_region.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[contact_region == 0] = np.array([207, 56, 112]) / 255.0
        vertex_color[contact_region == 1] = np.array([226, 53, 74]) / 255.0
        vertex_color[contact_region == 2] = np.array([231, 91, 84]) / 255.0

        vertex_color[contact_region == 3] = np.array([235, 105, 79]) / 255.0
        vertex_color[contact_region == 4] = np.array([230, 109, 91]) / 255.0
        vertex_color[contact_region == 5] = np.array([202, 67, 99]) / 255.0

        vertex_color[contact_region == 6] = np.array([240, 162, 62]) / 255.0
        vertex_color[contact_region == 7] = np.array([244, 192, 99]) / 255.0
        vertex_color[contact_region == 8] = np.array([239, 179, 145]) / 255.0

        vertex_color[contact_region == 9] = np.array([224, 231, 243]) / 255.0
        vertex_color[contact_region == 10] = np.array([175, 186, 242]) / 255.0
        vertex_color[contact_region == 11] = np.array([195, 212, 240]) / 255.0

        vertex_color[contact_region == 12] = np.array([50, 115, 173]) / 255.0
        vertex_color[contact_region == 13] = np.array([82, 148, 200]) / 255.0
        vertex_color[contact_region == 14] = np.array([124, 191, 239]) / 255.0

        vertex_color[contact_region == 15] = np.array([144, 78, 150]) / 255.0
        vertex_color[contact_region == 16] = np.array([40, 76, 121]) / 255.0

        vertex_color[contact_region == 17] = np.array([255, 255, 0]) / 255.0
        return vertex_color
    else:
        raise ValueError(f"Unknown color mode: {mode}")


def paper_vertex_color(contact_info, mode="vertex_contact"):
    if mode == "vertex_contact":
        vertex_contact = contact_info["vertex_contact"]
        n_verts = vertex_contact.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[vertex_contact == 0] = np.array([57, 57, 57]) / 255.0
        vertex_color[vertex_contact == 1] = np.array([198, 198, 198]) / 255.0
        return vertex_color
    elif mode == "contact_region":
        contact_region = contact_info["hand_region"]
        n_verts = contact_region.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[contact_region == 0] = np.array([207, 56, 112]) / 255.0
        vertex_color[contact_region == 1] = np.array([226, 53, 74]) / 255.0
        vertex_color[contact_region == 2] = np.array([231, 91, 84]) / 255.0

        vertex_color[contact_region == 3] = np.array([235, 105, 79]) / 255.0
        vertex_color[contact_region == 4] = np.array([230, 109, 91]) / 255.0
        vertex_color[contact_region == 5] = np.array([202, 67, 99]) / 255.0

        vertex_color[contact_region == 6] = np.array([240, 162, 62]) / 255.0
        vertex_color[contact_region == 7] = np.array([244, 192, 99]) / 255.0
        vertex_color[contact_region == 8] = np.array([239, 179, 145]) / 255.0

        vertex_color[contact_region == 9] = np.array([224, 231, 243]) / 255.0
        vertex_color[contact_region == 10] = np.array([175, 186, 242]) / 255.0
        vertex_color[contact_region == 11] = np.array([195, 212, 240]) / 255.0

        vertex_color[contact_region == 12] = np.array([50, 115, 173]) / 255.0
        vertex_color[contact_region == 13] = np.array([82, 148, 200]) / 255.0
        vertex_color[contact_region == 14] = np.array([124, 191, 239]) / 255.0

        vertex_color[contact_region == 15] = np.array([144, 78, 150]) / 255.0
        vertex_color[contact_region == 16] = np.array([40, 76, 121]) / 255.0

        vertex_color[contact_region == 17] = np.array([255, 232, 246]) / 255.0

        return vertex_color
    else:
        raise ValueError(f"Unknown color mode: {mode}")


def debug_vertex_color(contact_info, mode="vertex_contact"):
    if mode == "vertex_contact":
        vertex_contact = contact_info["vertex_contact"]
        n_verts = vertex_contact.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[vertex_contact == 0] = np.array([57, 57, 57]) / 255.0
        vertex_color[vertex_contact == 1] = np.array([198, 198, 198]) / 255.0
        return vertex_color
    elif mode == "contact_region":
        contact_region = contact_info["hand_region"]
        n_verts = contact_region.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[contact_region == 0] = np.array([117, 0, 0]) / 255.0
        vertex_color[contact_region == 1] = np.array([255, 0, 0]) / 255.0
        vertex_color[contact_region == 2] = np.array([255, 138, 137]) / 255.0

        vertex_color[contact_region == 3] = np.array([117, 65, 0]) / 255.0
        vertex_color[contact_region == 4] = np.array([255, 144, 0]) / 255.0
        vertex_color[contact_region == 5] = np.array([255, 206, 134]) / 255.0

        vertex_color[contact_region == 6] = np.array([116, 117, 0]) / 255.0
        vertex_color[contact_region == 7] = np.array([255, 255, 0]) / 255.0
        vertex_color[contact_region == 8] = np.array([255, 255, 131]) / 255.0

        vertex_color[contact_region == 9] = np.array([0, 117, 0]) / 255.0
        vertex_color[contact_region == 10] = np.array([0, 255, 0]) / 255.0
        vertex_color[contact_region == 11] = np.array([145, 255, 133]) / 255.0

        vertex_color[contact_region == 12] = np.array([0, 60, 118]) / 255.0
        vertex_color[contact_region == 13] = np.array([0, 133, 255]) / 255.0
        vertex_color[contact_region == 14] = np.array([136, 200, 255]) / 255.0

        vertex_color[contact_region == 15] = np.array([70, 0, 118]) / 255.0
        vertex_color[contact_region == 16] = np.array([210, 135, 255]) / 255.0

        vertex_color[contact_region == 17] = np.array([255, 232, 246]) / 255.0
        return vertex_color
    else:
        raise ValueError(f"Unknown color mode: {mode}")


def view_vertex_contact(hodata):
    import cv2
    import pygame
    import open3d as o3d

    clock = pygame.time.Clock()
    idx = 0
    img_path = hodata.get_image_path(idx)
    hand_gt = hodata.get_hand_verts3d(idx)
    contact_info = hodata.get_processed_contact_info(idx)
    hand_mesh_cur = o3d.geometry.TriangleMesh()
    hand_mesh_cur.vertices = o3d.utility.Vector3dVector(hand_gt)
    hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hodata.get_hand_faces(idx))
    hand_mesh_cur.compute_vertex_normals()
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_verts_cur, obj_faces_cur = (
        hodata.get_obj_verts_transf(idx),
        hodata.get_obj_faces(idx),
    )
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_cur)
    obj_colors = create_vertex_color(contact_info, "contact_region")
    obj_mesh.compute_vertex_normals()
    # obj_mesh.paint_uniform_color([254 / 255.0, 77 / 255.0, 77 / 255.0])
    obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
    # vis_cur.add_geometry(obj_mesh)

    vis_cur = o3d.visualization.Visualizer()
    vis_cur.create_window(window_name="Runtime Hand", width=1280, height=720)

    vis_cur.add_geometry(obj_mesh)
    vis_cur.add_geometry(hand_mesh_cur)

    for idx in range(len(hodata)):
        img_path = hodata.get_image_path(idx)

        hand_gt = hodata.get_hand_verts3d(idx)
        contact_info = hodata.get_processed_contact_info(idx)

        hand_mesh_cur.vertices = o3d.utility.Vector3dVector(hand_gt)
        hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hodata.get_hand_faces(idx))
        hand_mesh_cur.compute_vertex_normals()

        obj_verts_cur, obj_faces_cur = (
            hodata.get_obj_verts_transf(idx),
            hodata.get_obj_faces(idx),
        )
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces_cur)
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_cur)
        obj_colors = create_vertex_color(contact_info, "contact_region")
        obj_mesh.compute_vertex_normals()
        obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
        vis_cur.update_geometry(obj_mesh)
        vis_cur.update_geometry(hand_mesh_cur)
        vis_cur.update_renderer()

        vis_cur.poll_events()

        img = hodata.get_image(idx)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("dataset", img)
        cv2.waitKey(1)

        clock.tick(30)
