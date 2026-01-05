"""Adapter module exposing DBSCANAssembler at package level.

NVFlare may import "dbscan_assembler.DBSCANAssembler" from the job root. This file
re-exports the assembler implemented under app.custom so the import works regardless
of whether NVFlare adds app/ or app_server/ to sys.path.
"""
from app.custom.dbscan_assembler import DBSCANAssembler

__all__ = ["DBSCANAssembler"]
