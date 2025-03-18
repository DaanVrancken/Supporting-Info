# PySCF_calculations

This folder contains the code for downfolding trans-polyacetylene and polythiophene using PySCF.

## Installation

In order to run the CRPA-code locally several packages need to be installed: 

* PySCF: [https://github.com/pyscf/pyscf.git](https://github.com/pyscf/pyscf.git)
* PyWannier90 (Using Wannier90): [https://github.com/hungpham2017/pyWannier90.git](https://github.com/hungpham2017/pyWannier90.git), Installation is most easily done as mentioned in one of the issues in the pywannier90 library, but repeated here:

    Note: No separate wannier90 installation is needed.

    on Linux:

    `git clone https://github.com/hungpham2017/libwannier90.git`

    `cd libwannier90`

    `pip install .`

    on mac:

    `pip install libwannier90`

    The wrapper pywannier90.py (via PySCF or mcu ) should see this library so no need to set any path in pywannier90.py

    Try` import libwannier90` to see if it works, you should see no error. On Linux, if you run into problem like undefined symbol: zgesv_, you can force load the lapack library:

    `export LD_PRELOAD=/path/to/your/liblapack.so`

* generic packages such as matplotlib, scipy and numpy are also used.

## Usage

Run the scripts in the ```scripts``` folder to perform the DFT, Wannierization, and cRPA in one go. The computed tight-binding hopping parameters and interaction parameters are stored in the files `tmn.npy` and `Wmn.npy`, respectively. 

- In `tmn.npy`, the first index represents the destination site of the electron hop (e.g., index 0 corresponds to site `[-3,0,0]` and index 7 to site `[3,0,0]` by default), while the second and third indices define the corresponding hopping matrix.  
- The file `Wmn.npy` contains interaction parameters with eight indices: the first four each specify a site, and the last four correspond to the orbital at the respective site.
