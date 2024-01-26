import os
import pickle

from matplotlib import pyplot as plt

from hocontact.utils import eval as evalutils
from hocontact.visualize import samplevis


def summarize_evaluator_picr(evaluator: evalutils.Evaluator, exp_id, epoch_idx, train=False):
    img_folder = os.path.join(exp_id, "images")
    os.makedirs(img_folder, exist_ok=True)
    prefix = "train" if train else "val"
    fig = plt.figure(figsize=(10, 10))
    save_dict = {}

    for loss_name, avg_meter in evaluator.loss_meters.items():
        save_dict[loss_name] = {}
        loss_val = avg_meter.avg
        save_dict[loss_name][prefix] = loss_val

    eval_results = evaluator.parse_evaluators()

    for eval_name, eval_res in eval_results.items():
        target_meter = evaluator.eval_meters[eval_name]
        if isinstance(target_meter, evalutils.EvalUtil):
            for met in ["epe_mean", "auc"]:
                loss_name = f"{eval_name}_{met}"
                # Filter nans, since Numpy has : np.nan != np.nan
                if eval_res[met] != eval_res[met]:
                    continue
                save_dict[loss_name] = {}
                save_dict[loss_name][prefix] = eval_res[met]

    img_filepath = f"{prefix}_epoch{epoch_idx:04d}_eval.png"
    save_img_path = os.path.join(img_folder, img_filepath)

    eval_results_no_nans = {}
    eval_results_for_viz = {}
    for eval_name, res in eval_results.items():
        target_meter = evaluator.eval_meters[eval_name]
        if isinstance(target_meter, evalutils.EvalUtil):
            # Filter out Nan pck curves
            if res["epe_mean"] != res["epe_mean"]:
                continue
            eval_results_no_nans[eval_name] = res
            eval_results_for_viz[eval_name] = res

    samplevis.eval_vis(eval_results_for_viz, save_img_path, fig=fig)
    plt.close(fig)

    pickle_path = save_img_path.replace(".png", ".pkl")
    with open(pickle_path, "wb") as p_f:
        pickle.dump(eval_results_no_nans, p_f)

    return save_dict
