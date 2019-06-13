from setuptools import setup

with open('README.md', 'r') as f:
	long_description = f.read()

setup(
	name='parscanning',
	version='0.1',
	description='Parameter scanning',
	license='MIT',
	author='Jorge Alda',
	author_email='jalda@unizar.es',
	url='https://github.com/Jorge-Alda/parscanning',
	packages=['parscanning'],
)
