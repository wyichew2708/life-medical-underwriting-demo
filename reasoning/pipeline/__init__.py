"""Reasoning pipeline for the hybrid underwriting demo.

Stages, in order: validate the case, evaluate typed routing rules, screen with the
deployed ML model if one is configured, retrieve knowledge, reason with the LLM under a
fixed contract, validate the output, allow one bounded revision, then apply guardrails
that no model output can loosen.
"""
from .knowledge import KnowledgeBase, KnowledgeError, load
from .engine import Pipeline, PipelineError, assess

__all__ = ['KnowledgeBase', 'KnowledgeError', 'load', 'Pipeline', 'PipelineError', 'assess']
