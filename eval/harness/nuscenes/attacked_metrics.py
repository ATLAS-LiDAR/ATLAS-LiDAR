"""Official nuScenes AP / mAP / NDS restricted to the ATLAS attacked frames.

The devkit's NuScenesEval always scores a whole split. Here its initialization is
replaced by an exact-token subset; accumulation, matching and the metric definitions are
the devkit's own. Every real object in an attacked frame is a ground truth; injected
phantoms are not. Frames with no predictions stay in the evaluation as empty samples.
"""
import copy
from pathlib import Path


def subset_boxes(boxes, tokens):
    """Copy of `boxes` restricted to `tokens`, keeping empty samples."""
    tokens = set(tokens)
    if not tokens or not tokens <= set(boxes.sample_tokens):
        raise ValueError('Empty token set or tokens outside ground truth')
    out = type(boxes)()
    for token in sorted(tokens):
        out.add_boxes(token, copy.deepcopy(boxes[token]))
    return out


def filter_preserving_empty_samples(nusc, boxes, class_range):
    from nuscenes.eval.common.loaders import filter_eval_boxes
    nonempty = type(boxes)()
    for token in boxes.sample_tokens:
        if boxes[token]:
            nonempty.add_boxes(token, boxes[token])
    # Older devkits inspect the first sample's first box, so empty samples are left out
    # during filtering only, then restored.
    if nonempty.sample_tokens:
        nonempty = filter_eval_boxes(nusc, nonempty, class_range, verbose=True)
    result = type(boxes)()
    for token in boxes.sample_tokens:
        result.add_boxes(token, nonempty[token] if token in nonempty.sample_tokens else [])
    return result


def make_evaluator(nusc, cfg, prediction_path, output, tokens, gt_boxes):
    """A NuScenesEval over exactly `tokens`; call .main() on it as usual."""
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.common.loaders import load_prediction, add_center_dist
    from nuscenes.eval.detection.data_classes import DetectionBox
    output = Path(output)
    evaluator = NuScenesEval.__new__(NuScenesEval)
    evaluator.nusc, evaluator.cfg = nusc, cfg
    evaluator.result_path, evaluator.eval_set = str(prediction_path), 'val'
    evaluator.output_dir, evaluator.verbose = str(output), True
    evaluator.plot_dir = str(output / 'plots')
    Path(evaluator.plot_dir).mkdir(parents=True, exist_ok=True)
    evaluator.pred_boxes, evaluator.meta = load_prediction(
        str(prediction_path), cfg.max_boxes_per_sample, DetectionBox, verbose=True)
    if set(evaluator.pred_boxes.sample_tokens) != set(tokens):
        raise ValueError('Prediction token set does not match attack records')
    evaluator.gt_boxes = subset_boxes(gt_boxes, tokens)
    evaluator.pred_boxes = add_center_dist(nusc, evaluator.pred_boxes)
    evaluator.gt_boxes = add_center_dist(nusc, evaluator.gt_boxes)
    evaluator.pred_boxes = filter_preserving_empty_samples(nusc, evaluator.pred_boxes, cfg.class_range)
    evaluator.gt_boxes = filter_preserving_empty_samples(nusc, evaluator.gt_boxes, cfg.class_range)
    evaluator.sample_tokens = evaluator.gt_boxes.sample_tokens
    if set(evaluator.pred_boxes.sample_tokens) != set(evaluator.sample_tokens):
        raise ValueError('Prediction/GT token mismatch after filtering')
    return evaluator
