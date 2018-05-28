import numpy as np


class _CONF:
    def __init__(self, kw):
        self.__dict__ = kw


def mkflags(opts_dict):
    """Create namespace with flags from dictionary"""
    return _CONF(opts_dict)


def to_sec(timestr):
    """
    Converts `hh:mm:ss` to duration in seconds.
    """
    ts = [i for i in map(int, timestr.split(":"))]
    l = len(ts)
    k = 1
    sec = 0
    ts.reverse()
    for v in ts:
        sec += v * k
        k *= 60
    return sec


def gen_crop_wnds(img_sz, wnd_sz):
    """
    Generate half-overlap crop windows.
    """
    w_offs = np.linspace(0, img_sz[1] - wnd_sz, 2 * img_sz[1] // wnd_sz, dtype=np.int32)
    h_offs = np.linspace(0, img_sz[0] - wnd_sz, 2 * img_sz[0] // wnd_sz, dtype=np.int32)
    crops = []
    for offh in h_offs:
        for offw in w_offs:
            crops += [(offh, offw, wnd_sz, wnd_sz)]
    return [crops]


def get_image_paths(paths):
    """
    Get all `*.jpg` files from given paths.
    This also scans whole directories.
    """
    _paths = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".jpg"):
            _paths += [path]
        elif os.path.isdir(path):
            for ps, ds, fs in os.walk(path):
                for f in fs:
                    if f.endswith(".jpg"):
                        _paths += [os.path.join(ps,f)]
    return _paths

