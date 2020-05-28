from baseline.models.FeatherNet import FeatherNetA, FeatherNetB, conv_bn

def FeatherNetA_5ch(**kwargs):
    model = FeatherNetA(**kwargs)
    model.features[0] = conv_bn(5, 32, 2)
    return model