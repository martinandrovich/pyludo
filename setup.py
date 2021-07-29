#!/usr/bin/env python3
from setuptools import setup

# installation script, install using
# pip3 install -e .

setup(
	name='pyludo',
	version='1.0.0',
	description='A python LUDO game simulator. Forked from RasmusHaugaard.',
	author='Martin Androvich',
	url='https://github.com/martinandrovich/pyludo',
	packages=['pyludo'],
	install_requires=[
		'numpy',
		'pyglet',
	]
)
