from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector



model = RcnnDetector()


print("Anchor sizes:", model.rpn.anchor_generator.sizes)
print("Anchor aspect ratios:", model.rpn.anchor_generator.aspect_ratios)
print("Anchors per location:", model.rpn.anchor_generator.num_anchors_per_location())