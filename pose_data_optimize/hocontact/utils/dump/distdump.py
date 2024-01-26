def summarize_dumper_list(dumper_list, target_count):
    total_count = 0
    for dumper in dumper_list:
        total_count += dumper.counter
    if total_count == target_count:
        msg = f"OK: dumped {total_count}, target {target_count}"
        color = "green"
    else:
        msg = f"WARNING: dumped {total_count}, target {target_count}\n"
        msg += "WARNING: this is possibly caused by DDP bug "
        msg += "(see pytorch issue https://github.com/pytorch/pytorch/issues/25162)"
        color = "red"
    return msg, color
