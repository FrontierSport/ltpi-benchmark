import math
import logging

from tracklab.pipeline import Evaluator as EvaluatorBase

import pandas as pd

log = logging.getLogger(__name__)


class ReidEvaluator(EvaluatorBase):
    """
    Requires camera parameters and ground truth lines in tracker_state for evaluating homography
    """

    def __init__(self, csis, *args, **kwargs):
        self.csis_args = csis

    def run(self, tracker_state):
        metrics = {}
        results = {'right': 0, 'wrong': 0, 'unk': 0}
        
        # TODO: добавить сортировку по убыванию конфиденса
        
        # Find wrong, right and unknown subtrack identifications
        print(tracker_state.detections_pred)
        
        for _, dets in tracker_state.detections_pred.groupby('track_id'):
            assert len(dets['pred_id'].unique()) == 1
            pred_id = dets['pred_id'].iloc[0]
            if pd.isna(pred_id):
                results['unk'] += 1
            elif len(dets['gt_track_id'].unique()) != 1:
                results['wrong'] += 1
            elif pred_id != dets['gt_track_id'].iloc[0]:
                results['wrong'] += 1
            else:
                results['right'] += 1
        
        # CSIS
        cost = self.csis_args.cost_wrong * results['wrong'] + self.csis_args.cost_unk * results['unk']
        cost_avg = cost / (results['wrong'] + results['right'] + results['unk'])
        metrics['csis'] = 1 - (cost_avg / self.csis_args.cost_wrong)
        
        # UNKRate
        metrics['UnkRate'] = results['unk'] / (results['wrong'] + results['right'] + results['unk'])
        metrics['Coverage'] = 1 - metrics['UnkRate']

        # MisID
        metrics['MisID'] = results['wrong'] / (results['wrong'] + results['right'] + results['unk'])

        print(metrics)
        # experiment_logger.experiment_logger.log_dict(metrics, "Reid metrics")
