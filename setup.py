## python3 setup.py bdist_wheel
## python3 -m twine upload dist/* 
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='banduppy',  
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@ehu.eus",
     maintainer="Badal Mondal",
     maintainer_email="badalmondal.chembgc@gmail.com",
     description="BandUPpy: Python interface of the BandUP code",
     long_description=long_description,
     long_description_content_type="text/markdown",
     install_requires=['numpy', 'scipy >= 1.0', 'matplotlib' ,'irrep>=1.6.2'],
     url="https://www.ifm.liu.se/theomod/compphys/band-unfolding/",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
 )
