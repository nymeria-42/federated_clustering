"""Adapter module exposing DBSCANLearner at package level.

NVFlare may import "dbscan_learner.DBSCANLearner" from the job root. This file
re-exports the implementation from the job's app.custom package so the
configured import path resolves regardless of sys.path layout.
"""
from app.custom.dbscan_learner import DBSCANLearner

__all__ = ["DBSCANLearner"]
