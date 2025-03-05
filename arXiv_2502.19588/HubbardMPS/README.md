# HubbardMPS

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> HubbardMPS

It is authored by DaanVrancken.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

   This will install all necessary packages for you to be able to run the scripts and
   everything should work out of the box, including correctly finding local paths.

   You may notice that most scripts start with the commands:
   ```julia
   using DrWatson
   @quickactivate "HubbardMPS"
   ```
   which auto-activate the project and enable local path handling from DrWatson.

2. The code for the tensor network calculations of the three materials trans-polyacetylene (tPA), polythiophene (PT), and Sr<sub>2</sub>CuO<sub>3</sub> can be found in the folder ```scripts```. Run them with the required arguments provided as explained in the scripts. You may want to compute the ground state (which is automatically saved) first and perform a second run for the excitations.
