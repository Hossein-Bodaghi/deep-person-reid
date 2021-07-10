#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:58:00 2020

@author: hossein
"""

from torchreid.utils import RankLogger
s = 'market1501'
t = 'market1501'
ranklogger = RankLogger(s, t)
ranklogger.write(t, 10, 0.5)
ranklogger.write(t, 20, 0.7)
ranklogger.write(t, 30, 0.9)
ranklogger.show_summary()
# You will see:
# => Show performance summary
# market1501 (source)
# - epoch 10   rank1 50.0%
# - epoch 20   rank1 70.0%
# - epoch 30   rank1 90.0%
# If there are multiple test datasets
t = ['market1501', 'dukemtmcreid']
ranklogger = RankLogger(s, t)
ranklogger.write(t[0], 10, 0.5)
ranklogger.write(t[0], 20, 0.7)
ranklogger.write(t[0], 30, 0.9)
ranklogger.write(t[1], 10, 0.1)
ranklogger.write(t[1], 20, 0.2)
ranklogger.write(t[1], 30, 0.3)
ranklogger.show_summary()
# You can see:
# => Show performance summary
# market1501 (source)
# - epoch 10   rank1 50.0%
# - epoch 20   rank1 70.0%
# - epoch 30   rank1 90.0%
# dukemtmcreid (target)
# - epoch 10   rank1 10.0%
# - epoch 20   rank1 20.0%
# - epoch 30   rank1 30.0%