# ISS Project 2020/21
Project for course [ISS](https://www.fit.vut.cz/study/course/ISS/.en) (Signals and Systems) at FIT BUT 2020
## Description
- Script for analysis & simulation of the impact of face masks on human speech; and production of graphs for output protocol
- [Documentation of implementation & Protocol of achieved results](https://github.com/bix-1/ISS/blob/master/doc/xbartk07.pdf) *(in Slovak)*
## [Assignment](https://github.com/bix-1/ISS/blob/master/doc/assignment.pdf) *(in Czech)*
- Procude recordings of of single tone & single sentence with and without face mask
- Analyse the recordings, produce estimates of parameters of the face mask & use them for simulation of the mask's impact on the recordings without the mask
## Authors
- Jakub Bartko xbartk07@stud.fit.vutbr.cz
## Installation
- Download [tar file](https://github.com/bix-1/ISS/blob/master/src/xbartk07.tar.gz)
- Unpack using `tar -xf xbartk07.tar.gz`
- Navigate to `src`
- Install **dependencies** using `python3 -m pip install -r requirements.txt`
## Usage
- **Run** using `python3 solution.py`
- **Generate graphs** using `python3 solution.py -g`
## NOTES
- At *merlin.fit.vutbr.cz* run with `/usr/local/bin/python3.8` instead of `python3`
- **SoundFile** *v0.10.0* was used, but due to *merlin* not recognizing its version specification I was forced to ommit it from *requirements.txt*
