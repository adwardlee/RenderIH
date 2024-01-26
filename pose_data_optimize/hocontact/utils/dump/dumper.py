# Dumping Format
# [
#    {
#       "contact": 0/1,
#       "region": target_region,
#       "anchor_id": anchor_list,
#       "anchor_elasti": elasti_mat.tolist(),
#    }
# ]
# and
# {
#   "object_tsl": np.array(3,)
#   "object_rot": np.array(3,3)
#   "hand_tsl":   np.array(3,)
#   "hand_rot":   np.array(3,3)
#   "hand_shape": np.array(10,)
# }

import os
import pickle
from abc import ABC, abstractmethod

import torch
from manopth.anchorutils import anchor_load, get_rev_anchor_mapping

from hocontact.hodatasets.hoquery import BaseQueries, MetaQueries, CollateQueries


class Dumper(ABC):
    def __init__(self, dump_prefix, anchor_root):
        self.dump_prefix = dump_prefix
        _, _, _, self.anchor_mapping = anchor_load(anchor_root)
        self.rev_anchor_mapping = get_rev_anchor_mapping(self.anchor_mapping)
        self.counter = 0

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def feed_and_dump(self, sample, results):
        pass


class PicrOfflineDumper(Dumper):
    def __init__(self, dump_prefix, anchor_root):
        super().__init__(dump_prefix, anchor_root)
        self.type = "PicrOfflineDumper"

    def info(self):
        res = f"{self.type}\n"
        res += f"  prefix: {self.dump_prefix}\n"
        res += f"  count: {self.counter}"
        return res

    def feed_and_dump(self, sample, results, vc_thresh):
        # get sample identifier
        sample_identifier = sample[MetaQueries.SAMPLE_IDENTIFIER]
        n_sample = len(sample_identifier)
        collate_mask = sample[CollateQueries.PADDING_MASK]  # TENSOR[B, N]

        # assert fields in results
        assert "recov_vertex_contact" in results, f"{self.type}: vertex_contact not found"
        assert "recov_contact_region" in results, f"{self.type}: contact_region not found"
        assert "recov_anchor_elasti" in results, f"{self.type}: anchor_elasti not found"
        recov_vertex_contact = results["recov_vertex_contact"].detach()  # TENSOR[B, N]
        recov_contact_in_image_mask = results["recov_contact_in_image_mask"].detach()  # TENSOR[B, N]
        recov_contact_region = results["recov_contact_region"].detach()  # TENSOR[B, N, 17]
        recov_anchor_elasti_pred = results["recov_anchor_elasti"].detach()  # TENSOR[B, N, 4]

        recov_vertex_contact_pred = (torch.sigmoid(recov_vertex_contact) > vc_thresh).bool()  # TENSOR[B, N]
        recov_contact_region_pred = torch.argmax(recov_contact_region, dim=2)  # TENSOR[B, N]

        # iterate over samples, assemble dump dict
        for idx in range(n_sample):
            sample_id = sample_identifier[idx]
            sample_collate_mask = collate_mask[idx, :].bool()  # TENSOR[N, ]

            sample_vertex_contact = recov_vertex_contact_pred[idx, :]  # TENSOR[N, ]
            sample_contact_in_image_mask = recov_contact_in_image_mask[idx, :]  # TENSOR[N, ]
            combined_vertex_contact = sample_vertex_contact.bool() & sample_contact_in_image_mask.bool()  # TENSOR[N,]
            filtered_vertex_contact = combined_vertex_contact[sample_collate_mask]  # TENSOR[X, ]

            sample_contact_region = recov_contact_region_pred[idx, :]  # TENSOR[N, ]
            filtered_contact_region = sample_contact_region[sample_collate_mask]  # TENSOR[X, ]
            sample_anchor_elasti = recov_anchor_elasti_pred[idx, :, :]  # TENSOR[N, 4]
            filtered_anchor_elasti = sample_anchor_elasti[sample_collate_mask, :]  # TENSOR[X, 4]

            # transport from cuda to cpu
            filtered_vertex_contact = filtered_vertex_contact.cpu()
            filtered_contact_region = filtered_contact_region.cpu()
            filtered_anchor_elasti = filtered_anchor_elasti.cpu()

            # iterate over all points
            sample_res = []
            n_points = filtered_vertex_contact.shape[0]  # X
            for p_idx in range(n_points):
                p_contact = int(filtered_vertex_contact[p_idx])
                if p_contact == 0:
                    p_res = {
                        "contact": 0,
                    }
                else:  # p_contact == 1
                    p_region = int(filtered_contact_region[p_idx])
                    p_anchor_id = self.rev_anchor_mapping[p_region]
                    p_n_anchor = len(p_anchor_id)
                    p_anchor_elasti = filtered_anchor_elasti[p_idx, :p_n_anchor].tolist()
                    p_res = {
                        "contact": 1,
                        "region": p_region,
                        "anchor_id": p_anchor_id,
                        "anchor_elasti": p_anchor_elasti,
                    }
                sample_res.append(p_res)

            # save sample_res
            save_path = os.path.join(self.dump_prefix, f"{sample_id}.pkl")
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as fstream:
                pickle.dump(sample_res, fstream)
            self.counter += 1


class PicrDumper(Dumper):
    def __init__(self, dump_prefix, anchor_root):
        super().__init__(dump_prefix, anchor_root)
        self.type = "PicrDumper"
        self.honet_fields = [
            "hand_tsl",
            "hand_joints_3d",
            "hand_verts_3d",
            "hand_full_pose",
            "hand_shape",
            "obj_tsl",
            "obj_rot",
            "obj_verts_3d",
        ]

    def info(self):
        res = f"{self.type}\n"
        res += f"  prefix: {self.dump_prefix}\n"
        res += f"  count: {self.counter}"
        return res

    def feed_and_dump(self, sample, results, vc_thresh):
        # get sample identifier
        sample_identifier = sample[MetaQueries.SAMPLE_IDENTIFIER]
        n_sample = len(sample_identifier)
        collate_mask = sample[CollateQueries.PADDING_MASK]  # TENSOR[B, N]

        # ====== assert fields in results: contact related
        assert "recov_vertex_contact" in results, f"{self.type}: vertex_contact not found"
        assert "recov_contact_region" in results, f"{self.type}: contact_region not found"
        assert "recov_anchor_elasti" in results, f"{self.type}: anchor_elasti not found"
        recov_vertex_contact = results["recov_vertex_contact"].detach()  # TENSOR[B, N]
        recov_contact_in_image_mask = results["recov_contact_in_image_mask"].detach()  # TENSOR[B, N]
        recov_contact_region = results["recov_contact_region"].detach()  # TENSOR[B, N, 17]
        recov_anchor_elasti_pred = results["recov_anchor_elasti"].detach()  # TENSOR[B, N, 4]

        recov_vertex_contact_pred = (torch.sigmoid(recov_vertex_contact) > vc_thresh).bool()  # TENSOR[B, N]
        recov_contact_region_pred = torch.argmax(recov_contact_region, dim=2)  # TENSOR[B, N]

        # ====== assert fields in results: honet related
        for field in self.honet_fields:
            assert field in results, f"{self.type}: {field} not found"

        # iterate over samples, assemble dump dict
        for idx in range(n_sample):
            sample_id = sample_identifier[idx]

            # ==================== dump contact related info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            sample_collate_mask = collate_mask[idx, :].bool()  # TENSOR[N, ]
            sample_vertex_contact = recov_vertex_contact_pred[idx, :]  # TENSOR[N, ]
            sample_contact_in_image_mask = recov_contact_in_image_mask[idx, :]  # TENSOR[N, ]
            combined_vertex_contact = sample_vertex_contact.bool() & sample_contact_in_image_mask.bool()  # TENSOR[N,]
            filtered_vertex_contact = combined_vertex_contact[sample_collate_mask]  # TENSOR[X, ]

            sample_contact_region = recov_contact_region_pred[idx, :]  # TENSOR[N, ]
            filtered_contact_region = sample_contact_region[sample_collate_mask]  # TENSOR[X, ]
            sample_anchor_elasti = recov_anchor_elasti_pred[idx, :, :]  # TENSOR[N, 4]
            filtered_anchor_elasti = sample_anchor_elasti[sample_collate_mask, :]  # TENSOR[X, 4]

            # transport from cuda to cpu
            filtered_vertex_contact = filtered_vertex_contact.cpu()
            filtered_contact_region = filtered_contact_region.cpu()
            filtered_anchor_elasti = filtered_anchor_elasti.cpu()

            # iterate over all points
            sample_res = []
            n_points = filtered_vertex_contact.shape[0]  # X
            for p_idx in range(n_points):
                p_contact = int(filtered_vertex_contact[p_idx])
                if p_contact == 0:
                    p_res = {
                        "contact": 0,
                    }
                else:  # p_contact == 1
                    p_region = int(filtered_contact_region[p_idx])
                    p_anchor_id = self.rev_anchor_mapping[p_region]
                    p_n_anchor = len(p_anchor_id)
                    p_anchor_elasti = filtered_anchor_elasti[p_idx, :p_n_anchor].tolist()
                    p_res = {
                        "contact": 1,
                        "region": p_region,
                        "anchor_id": p_anchor_id,
                        "anchor_elasti": p_anchor_elasti,
                    }
                sample_res.append(p_res)

            # save sample_res
            save_path = os.path.join(self.dump_prefix, f"{sample_id}_contact.pkl")
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as fstream:
                pickle.dump(sample_res, fstream)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # ==================== dump honet related info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            honet_res = {}
            for field in self.honet_fields:
                if field in ["obj_verts_3d"]:
                    honet_res[field] = results[field][idx, ...][sample_collate_mask, :].detach().cpu().numpy()
                honet_res[field] = results[field][idx, ...].detach().cpu().numpy()
            honet_res["image_path"] = sample[BaseQueries.IMAGE_PATH][idx]
            honet_save_path = os.path.join(self.dump_prefix, f"{sample_id}_honet.pkl")
            with open(honet_save_path, "wb") as fstream:
                pickle.dump(honet_res, fstream)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            self.counter += 1
