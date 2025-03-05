# VASP_calculations

This folder contains the VASP input for geometry relaxations, hybrid functional and GW computations of trans-polyacetylene, as well as the downfolding of Sr<sub>2</sub>CuO<sub>3</sub>.

## Geometry_relaxation
Contains the input files to optimize the geometries of trans-polyacetylene (tPA1 with PBE and tPA2 with B3LYP), polythiophene, and Sr<sub>2</sub>CuO<sub>3</sub>.

## Hybrids_tPA
Contains the input files for hybrid functional and GW calculations of three trans-polyacetylene geometries. To perform the calculation with the PBE, HSE06, or B3LYP functional, run the simulation in the corresponding folder. The GW calculations are performed in the folder ```MBPT``` as follows:
1. Run the initial SCF calculation in the folder ```SCF```. Copy the resulting WAVECAR file to the folder ```LWL```.
2. Run the DFT calculation with a large number of bands in the folder ```LWL```. Copy the WAVECAR and WAVEDER files to both the folders ```GW0``` and ```GW```.
3. Run the calculation in ```GW0``` for a partially self-consistent and that in ```GW``` for a fully self-consistent calculation.
Note that for the GW calculations the vacuum around the chains was reduced from 20 Å to 12.5 Å to lower the required memory. 

## Downfolding_Sr2CuO3
Contains the input files for performing cRPA on Sr<sub>2</sub>CuO<sub>3</sub>. Several steps have to be taken:
1. Run the calculation in the folder ```SCF```. Copy the WAVECAR file to the folder ```LWL```.
2. Run the calculation in the folder ```LWL```. Copy the WAVECAR and CHGCAR files to ```BANDS``` and the WAVECAR and WAVEDER to both ```MLWFs``` and ```cRPA```.
3. You can compute the band structure by running the calculation in ```BANDS```. This step is optional.
4. Run the calculation in ```MLWFs``` to compute the maximally localized Wannier functions. Copy the WANPROJ file to ```cRPA```.
5. Run the HOPPING.py script in ```MLWFs``` to obtain the hopping parameters.
6. Run the calculation in ```cRPA```. The interaction parameters can be found in ```UIJKL.1```.
