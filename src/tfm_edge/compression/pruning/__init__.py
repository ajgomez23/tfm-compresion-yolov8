"""Poda estructurada de canales.

La poda elimina canales físicamente; con máscaras no bajaría la latencia. El análisis de
dependencias lo hace torch-pruning (DepGraph), encapsulado en ``dependency``.
"""
