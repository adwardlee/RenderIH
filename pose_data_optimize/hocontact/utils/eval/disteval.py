from typing import List, Dict

from hocontact.utils.eval import Evaluator, EvalUtil
from hocontact.utils.eval.evalutils import AverageMeter


def dispatch_evaluator(
    evaluator: Evaluator, lossmeter_dict: Dict[str, List[AverageMeter]], evalutil_dict: Dict[str, List[EvalUtil]],
):
    this_loss_meter = evaluator.loss_meters
    for meter_name, meter in this_loss_meter.items():
        if meter_name not in lossmeter_dict:
            lossmeter_dict[meter_name] = [meter]
        else:
            lossmeter_dict[meter_name].append(meter)
    this_eval_util = evaluator.eval_meters
    for util_name, util in this_eval_util.items():
        if util_name not in evalutil_dict:
            evalutil_dict[util_name] = [util]
        else:
            evalutil_dict[util_name].append(util)


def merge_evaluator(evaluator_list: List[Evaluator]):
    lm_dict = dict()
    eu_dict = dict()
    for evaluator in evaluator_list:
        dispatch_evaluator(evaluator, lm_dict, eu_dict)

    for lm_key, lm_meter_list in lm_dict.items():
        if len(lm_meter_list) < 1:
            raise RuntimeError("unexpected 0 length meter list, check dispath function above")
        lm_dict[lm_key] = merge_averagemeter(lm_meter_list)

    for eu_key, eu_meter_list in eu_dict.items():
        if len(eu_meter_list) < 1:
            raise RuntimeError("unexpected 0 length meter list, check dispath function above")
        test_meter = eu_meter_list[0]
        if isinstance(test_meter, EvalUtil):
            eu_dict[eu_key] = merge_evalutil(eu_meter_list)

    res = Evaluator()
    res.loss_meters = lm_dict
    res.eval_meters = eu_dict
    return res


def merge_averagemeter(averagemeter_list: List[AverageMeter]):
    res = AverageMeter()
    res.val = averagemeter_list[-1].val
    for averagemeter in averagemeter_list:
        res.sum += averagemeter.sum
        res.count += averagemeter.count
    res.avg = res.sum / res.count
    return res


def merge_evalutil(evalutil_list: List[EvalUtil]):
    res = EvalUtil()
    for evalutil in evalutil_list:
        for i in range(res.num_kp):
            res.data[i].extend(evalutil.data[i])
    return res
