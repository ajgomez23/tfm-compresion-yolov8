"""Cuantización: PTQ y QAT.

QAT entrena con cuantización simulada en PyTorch eager y el INT8 se materializa después con cada
runtime, igual que en PTQ. La conversión eager no es posible porque PyTorch no tiene kernel
cuantizado para SiLU.
"""
