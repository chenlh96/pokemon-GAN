from .base import Illustration2VecBase
import traceback

caffe_available = False
chainer_available = False

try:
    from .caffe_i2v import CaffeI2V, make_i2v_with_caffe
    caffe_available = True
except ImportError:
    pass

try:
    from .chainer_i2v import ChainerI2V, make_i2v_with_chainer
    chainer_available = True
    print(chainer_available)
except Exception:
    traceback.print_exc()
    pass

if not any([caffe_available, chainer_available]):
    raise ImportError('i2v requires caffe or chainer package')
