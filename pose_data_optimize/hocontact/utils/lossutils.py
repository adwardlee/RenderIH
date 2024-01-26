def update_loss(ls_losses):
    # ls_losses = LIST[DICT]
    # this function sums over common keys
    # if list of losses is 0, return empty dict
    if len(ls_losses) == 0:
        return dict()

    # ? first, get the common keys of all dicts
    keys_list = []
    for loss_dict in ls_losses:
        for key in loss_dict.keys():
            keys_list.append(key)
    keys = list(set(keys_list))

    # ? init an dict with default value zeros
    res = dict()
    for key in keys:
        res[key] = 0.0

    # ? iterate over all loss dicts
    for loss_dict in ls_losses:
        for key in keys:
            res[key] += loss_dict[key]
    return res
