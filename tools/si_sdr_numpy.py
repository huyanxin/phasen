#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)
# to cal si_sdr


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = np.mean(signal)
    signal -= mean
    return signal


def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))


def pow_norm(s1, s2):
    return np.sum(s1 * s2)


def si_sdr(estimated, original):
    estimated = remove_dc(estimated)
    original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))


def permute_si_sdr(e1, e2, c1, c2):
    sdr1 = si_sdr(e1, c1) + si_sdr(e2, c2)
    sdr2 = si_sdr(e1, c2) + si_sdr(e2, c1)
    if sdr1 > sdr2:
        return sdr1 * 0.5
    else:
        return sdr2 * 0.5
