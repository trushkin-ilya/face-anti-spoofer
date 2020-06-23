from baseline.models.FeatherNet import FeatherNetA, FeatherNetB, conv_bn


def _FeatherNetA(in_ch: int, **kwargs):
    model = FeatherNetA(**kwargs)
    model.features[0] = conv_bn(in_ch, 32, 2)
    return model

def FeatherNetA_4ch(**kwargs):
    return _FeatherNetA(4, **kwargs)

def FeatherNetA_5ch(**kwargs):
    return _FeatherNetA(5, **kwargs)