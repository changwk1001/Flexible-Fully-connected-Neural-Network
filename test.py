# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 00:25:21 2021

@author: user
"""
from parser_ import parser
args = parser.get_parser().parse_args()
help(parser)
print(args)