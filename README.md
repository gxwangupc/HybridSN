# HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification
-------------------------------------------------
## Introduction
 * This is a keras implementation of the IEEE GRSL paper 'HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification'.<br>
 * See https://github.com/gokriznastic/HybridSN for the original implementation in the jupyter notebook form.<br>

 * I have adapted the code into the form of python script, making it easier for running directly.<br>
 
## Environment & Requirements
* CentOS Linux release 7.2.1511 (Core)<br>
* python 3.6.5<br>
* keras 2.3.1<br>
* spectral<br>

## Usage
### Download hyperspectral data.<br>
Download the Inidan Pines, the Salinas and the Pavia University and put them into *dataset*:<br>
### Train&Test the Inidan Pines:<br>

    python hybridsn.py -GPU 0 
	
You can change the namespace and conduct experiements under your preferred setting.

## References 
https://github.com/gokriznastic/HybridSN <br>
