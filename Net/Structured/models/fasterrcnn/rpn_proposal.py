import tensorflow as tf

class RPNProposal:
    """Transforms anchors and RPN predictions into object proposals.
    Using the fixed anchors and the RPN predictions for both classification
    and regression (adjusting the bounding box), we return a list of objects
    sorted by relevance.
    Besides applying the transformations (or adjustments) from the prediction,
    it tries to get rid of duplicate proposals by using non maximum supression
    (NMS).
    """
    def __init__(self, config, anchors_count):
        self.config         = config
        self.anchors_count  = anchors_count

        pass


    def get_obj_proposals(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, img_shape):
        pass