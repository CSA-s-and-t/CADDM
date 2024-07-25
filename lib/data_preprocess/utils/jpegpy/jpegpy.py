#!/usr/bin/env mdl
import cv2
import numpy as np
import importlib


def jpeg_encode(img: np.array, quality=80):
    _jpegpy = importlib.import_module("._jpegpy", package="lib.data_preprocess.utils")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _jpegpy.encode(img, quality)


def jpeg_decode(code: bytes):
    _jpegpy = importlib.import_module("._jpegpy", package="lib.data_preprocess.utils")
    img = _jpegpy.decode(code)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# vim: ts=4 sw=4 sts=4 expandtab
