import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size
    result = []

    for j in range(feature_size):
        for i in range(feature_size):
            cx = (i + 0.5) * stride
            cy = (j + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)

                    result.append([
                        cx - w/2,
                        cy - h/2,
                        cx + w/2,
                        cy + h/2
                    ])

    return result