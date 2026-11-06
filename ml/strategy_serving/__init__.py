"""Serving, replay and prospective checks for the pit-strategy models.

Kept outside ``ml.strategy`` on purpose: the frozen candidate's fingerprint covers
every file in ``ml/strategy``, so serving code must not live there.
"""
