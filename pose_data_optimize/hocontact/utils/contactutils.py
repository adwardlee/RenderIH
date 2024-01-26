import numpy as np

from manopth.anchorutils import get_rev_anchor_mapping


def get_padding_attr(anchor_mapping):
    rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping)
    anchor_id_background = len(anchor_mapping.keys())
    region_id_background = len(rev_anchor_mapping.keys())
    anchor_padding_len = max([len(v) for v in rev_anchor_mapping.values()])
    return anchor_id_background, region_id_background, anchor_padding_len


def process_contact_info(
    contact_info, anchor_mapping, pad_vertex=False, pad_anchor=False, dist_th=1000.0, elasti_th=0.00,
):
    anchor_id_background, region_id_background, anchor_padding_len = get_padding_attr(anchor_mapping)

    vertex_contact = [item["contact"] for item in contact_info]
    vertex_contact = np.array(vertex_contact, dtype=np.int)
    if pad_vertex and pad_anchor:
        # the return result will be the same length of vertex_contact
        hand_region = [item["region"] if item["contact"] == 1 else region_id_background for item in contact_info]
        hand_region = np.array(hand_region, dtype=np.int)
        # all the anchors will be padded to anchor_padding_len
        anchor_id = []
        anchor_dist = []
        anchor_elasti = []
        anchor_padding_mask = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_dist = item["anchor_dist"]
                item_anchor_elasti = item["anchor_elasti"]
                item_n_anchor = len(item_anchor_id)
                item_padding_len = anchor_padding_len - item_n_anchor

                item_anchor_padding_mask = np.zeros((anchor_padding_len,), dtype=np.int)
                item_anchor_padding_mask[:item_n_anchor] = 1
                item_anchor_id = item_anchor_id + ([anchor_id_background] * item_padding_len)
                item_anchor_dist = item_anchor_dist + ([dist_th] * item_padding_len)
                item_anchor_elasti = item_anchor_elasti + ([elasti_th] * item_padding_len)
            else:
                item_anchor_padding_mask = [0] * anchor_padding_len
                item_anchor_id = [anchor_id_background] * anchor_padding_len
                item_anchor_dist = [dist_th] * anchor_padding_len
                item_anchor_elasti = [elasti_th] * anchor_padding_len
            anchor_id.append(item_anchor_id)
            anchor_dist.append(item_anchor_dist)
            anchor_elasti.append(item_anchor_elasti)
            anchor_padding_mask.append(item_anchor_padding_mask)
        anchor_id = np.array(anchor_id, dtype=np.int)
        anchor_dist = np.array(anchor_dist)
        anchor_elasti = np.array(anchor_elasti)
        anchor_padding_mask = np.array(anchor_padding_mask, dtype=np.int)
        return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, anchor_padding_mask
    elif not pad_vertex and pad_anchor:
        # the return result will be the same length of ones in vertex_contact
        hand_region = [item["region"] for item in contact_info if item["contact"] == 1]
        hand_region = np.array(hand_region, dtype=np.int)
        # all the anchors will be padded to anchor_padding_len
        anchor_id = []
        anchor_dist = []
        anchor_elasti = []
        anchor_padding_mask = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_dist = item["anchor_dist"]
                item_anchor_elasti = item["anchor_elasti"]
                item_n_anchor = len(item_anchor_id)
                item_padding_len = anchor_padding_len - item_n_anchor

                item_anchor_padding_mask = np.zeros((anchor_padding_len,), dtype=np.int)
                item_anchor_padding_mask[:item_n_anchor] = 1
                item_anchor_id = item_anchor_id + ([anchor_id_background] * item_padding_len)
                item_anchor_dist = item_anchor_dist + ([dist_th] * item_padding_len)
                item_anchor_elasti = item_anchor_elasti + ([elasti_th] * item_padding_len)
            else:
                continue
            anchor_id.append(item_anchor_id)
            anchor_dist.append(item_anchor_dist)
            anchor_elasti.append(item_anchor_elasti)
            anchor_padding_mask.append(item_anchor_padding_mask)
        anchor_id = np.array(anchor_id, dtype=np.int)
        anchor_dist = np.array(anchor_dist)
        anchor_elasti = np.array(anchor_elasti)
        anchor_padding_mask = np.array(anchor_padding_mask, dtype=np.int)
        return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, anchor_padding_mask
    elif pad_vertex and not pad_anchor:
        # the return result will be the same length of vertex_contact
        hand_region = [item["region"] if item["contact"] == 1 else region_id_background for item in contact_info]
        hand_region = np.array(hand_region, dtype=np.int)
        # no anchors will be padded
        # will return list
        anchor_id = []
        anchor_dist = []
        anchor_elasti = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_dist = item["anchor_dist"]
                item_anchor_elasti = item["anchor_elasti"]
            else:
                item_anchor_id = []
                item_anchor_dist = []
                item_anchor_elasti = []
            anchor_id.append(item_anchor_id)
            anchor_dist.append(item_anchor_dist)
            anchor_elasti.append(item_anchor_elasti)
        return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, None
    else:
        # the return result will be the same length of ones in vertex_contact
        hand_region = [item["region"] for item in contact_info if item["contact"] == 1]
        hand_region = np.array(hand_region, dtype=np.int)
        # no anchors will be padded
        # will return list
        anchor_id = []
        anchor_dist = []
        anchor_elasti = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_dist = item["anchor_dist"]
                item_anchor_elasti = item["anchor_elasti"]
            else:
                continue
            anchor_id.append(item_anchor_id)
            anchor_dist.append(item_anchor_dist)
            anchor_elasti.append(item_anchor_elasti)
        return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, None


# ? this version is almost the same as above, except for it doesn't process anchor_dist
def dumped_process_contact_info(
    contact_info, anchor_mapping, pad_vertex=False, pad_anchor=False, elasti_th=0.00,
):
    anchor_id_background, region_id_background, anchor_padding_len = get_padding_attr(anchor_mapping)

    vertex_contact = [item["contact"] for item in contact_info]
    vertex_contact = np.array(vertex_contact, dtype=np.int)
    if pad_vertex and pad_anchor:
        # the return result will be the same length of vertex_contact
        hand_region = [item["region"] if item["contact"] == 1 else region_id_background for item in contact_info]
        hand_region = np.array(hand_region, dtype=np.int)
        # all the anchors will be padded to anchor_padding_len
        anchor_id = []
        anchor_elasti = []
        anchor_padding_mask = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_elasti = item["anchor_elasti"]
                item_n_anchor = len(item_anchor_id)
                item_padding_len = anchor_padding_len - item_n_anchor

                item_anchor_padding_mask = np.zeros((anchor_padding_len,), dtype=np.int)
                item_anchor_padding_mask[:item_n_anchor] = 1
                item_anchor_id = item_anchor_id + ([anchor_id_background] * item_padding_len)
                item_anchor_elasti = item_anchor_elasti + ([elasti_th] * item_padding_len)
            else:
                item_anchor_padding_mask = [0] * anchor_padding_len
                item_anchor_id = [anchor_id_background] * anchor_padding_len
                item_anchor_elasti = [elasti_th] * anchor_padding_len
            anchor_id.append(item_anchor_id)
            anchor_elasti.append(item_anchor_elasti)
            anchor_padding_mask.append(item_anchor_padding_mask)
        anchor_id = np.array(anchor_id, dtype=np.int)
        anchor_elasti = np.array(anchor_elasti)
        anchor_padding_mask = np.array(anchor_padding_mask, dtype=np.int)
        return vertex_contact, hand_region, anchor_id, anchor_elasti, anchor_padding_mask
    elif not pad_vertex and pad_anchor:
        # the return result will be the same length of ones in vertex_contact
        hand_region = [item["region"] for item in contact_info if item["contact"] == 1]
        hand_region = np.array(hand_region, dtype=np.int)
        # all the anchors will be padded to anchor_padding_len
        anchor_id = []
        anchor_elasti = []
        anchor_padding_mask = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_elasti = item["anchor_elasti"]
                item_n_anchor = len(item_anchor_id)
                item_padding_len = anchor_padding_len - item_n_anchor

                item_anchor_padding_mask = np.zeros((anchor_padding_len,), dtype=np.int)
                item_anchor_padding_mask[:item_n_anchor] = 1
                item_anchor_id = item_anchor_id + ([anchor_id_background] * item_padding_len)
                item_anchor_elasti = item_anchor_elasti + ([elasti_th] * item_padding_len)
            else:
                continue
            anchor_id.append(item_anchor_id)
            anchor_elasti.append(item_anchor_elasti)
            anchor_padding_mask.append(item_anchor_padding_mask)
        anchor_id = np.array(anchor_id, dtype=np.int)
        anchor_elasti = np.array(anchor_elasti)
        anchor_padding_mask = np.array(anchor_padding_mask, dtype=np.int)
        return vertex_contact, hand_region, anchor_id, anchor_elasti, anchor_padding_mask
    elif pad_vertex and not pad_anchor:
        # the return result will be the same length of vertex_contact
        hand_region = [item["region"] if item["contact"] == 1 else region_id_background for item in contact_info]
        hand_region = np.array(hand_region, dtype=np.int)
        # no anchors will be padded
        # will return list
        anchor_id = []
        anchor_elasti = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_elasti = item["anchor_elasti"]
            else:
                item_anchor_id = []
                item_anchor_elasti = []
            anchor_id.append(item_anchor_id)
            anchor_elasti.append(item_anchor_elasti)
        return vertex_contact, hand_region, anchor_id, anchor_elasti, None
    else:
        # the return result will be the same length of ones in vertex_contact
        hand_region = [item["region"] for item in contact_info if item["contact"] == 1]
        hand_region = np.array(hand_region, dtype=np.int)
        # no anchors will be padded
        # will return list
        anchor_id = []
        anchor_elasti = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
                item_anchor_elasti = item["anchor_elasti"]
            else:
                continue
            anchor_id.append(item_anchor_id)
            anchor_elasti.append(item_anchor_elasti)
        return vertex_contact, hand_region, anchor_id, anchor_elasti, None
