# VASP_calculations

This folder contains the VASP input for geometry relaxations, hybrid functional and GW computations of trans-polyacetylene, as well as the downfolding of Sr<sub>2</sub>CuO<sub>3</sub>.

## Geometry_relaxation
Contains the input files to optimize the geometries of trans-polyacetylene (tPA1 with PBE and tPA2 with B3LYP), polythiophene, and Sr<sub>2</sub>CuO<sub>3</sub>.

## Hybrids_tPA
Contains the input files for hybrid functional and GW calculations of trans-polyacetylene. For the GW calculations, initial SCF calculations should be performed (one with low and one with high number of bands).The output files should be reused as follows
> cp ./SCF/WAVECAR ./LWL/WAVECAR
> cp ./LWL/WAVECAR ./GW/WAVECAR