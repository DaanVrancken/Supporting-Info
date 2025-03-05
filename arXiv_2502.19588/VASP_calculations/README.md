# VASP_calculations

This folder contains the VASP input for geometry relaxations, hybrid functional and GW computations of trans-polyacetylene, as well as the downfolding of Sr<sub>2</sub>CuO<sub>3</sub>.

## Geometry_relaxation
Contains the input files to optimize the geometries of trans-polyacetylene (tPA1 with PBE and tPA2 with B3LYP), polythiophene, and Sr<sub>2</sub>CuO<sub>3</sub>.

## Hybrids_tPA
Contains the input files for hybrid functional and GW calculations of trans-polyacetylene. For the GW calculations, initial SCF calculations should be performed (one with low and one with high number of bands).The output files should be reused as follows
```
cp ./SCF/WAVECAR ./LWL/WAVECAR
cp ./LWL/WAVECAR ./GW/WAVECAR

```

## Downfolding_Sr2CuO3
Contains the input files for performing cRPA on Sr<sub>2</sub>CuO<sub>3</sub>. Several steps have to be taken:
1. Run the calculation in the folder ```SCF```. Copy the WAVECAR file to the folder ```LWL```.
2. Run the calculation in the folder ```LWL```. Copy the WAVECAR and CHGCAR files to ```BANDS``` and the WAVECAR and WAVEDER to both ```MLWFs``` and ```cRPA```.
3. You can compute the band structure by running the calculation in ```BANDS```. This step is optional.
4. Run the calculation in ```MLWFs``` to compute the maximally localized Wannier functions. Copy the WANPROJ file to ```cRPA```.
5. Run the HOPPING.py script in ```MLWFs``` to obtain the hopping parameters.
6. Run the calculation in ```cRPA```. The interaction parameters can be found in ```UIJKL.1```.
