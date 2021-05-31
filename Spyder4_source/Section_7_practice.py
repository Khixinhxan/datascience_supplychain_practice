#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:13:07 2021

@author: dang
"""

import pandas as pd

from datetime import datetime
import numpy as np

retail = pd.read_csv('./online_retail2.csv')

retail = retail.drop_duplicates()

retail = retail.dropna(axis=0, how='any')