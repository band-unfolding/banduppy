## python3 setup.py bdist_wheel
## python3 -m twine upload dist/* 
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


from banduppy import __version__ as version

setuptools.setup(
     name='banduppy',  
     version=version,
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@uzh.ch",
     description="BandUPpy: Python interfaceof the BandUP code",
     long_description=long_description,
     long_description_content_type="text/markdown",
     install_requires=['numpy', 'scipy >= 1.0', 'matplotlib' ,'irrep>=1.5.1'],
     url="https://www.ifm.liu.se/theomod/compphys/band-unfolding/",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
 )