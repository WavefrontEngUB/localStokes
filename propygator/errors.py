#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERRORS FOR PROPYGATOR (name pend.)
"""

class Error(Exception):
    """Base class"""
    pass

class DimensionError(Error):
    def __init__(self, message):
        self.message = message
