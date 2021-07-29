#!/usr/bin/env python3
from setuptools import setup

# installation script, install using
# pip3 install -e .

setup(
	name='pyludo',
	version='1.0.0',
	install_requires=[
		'numpy',
		'pyglet',
	]
)
