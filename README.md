# Accelerating Deep Neural Networks using SVD

An implementation of using Singular Value Decomposition to accelerate training of neural networks.

Implemented in [Julia](https://julialang.org/) using the [Flux ML Library](https://github.com/FluxML/Flux.jl).

## Details

See the [Report](Report.pdf) for technical details and comparisons (Skip to section 3.3 for the relevant parts)

## Usage

See [Examples](Examples).

## Requirements

Tested on Julia 1.3.0 and 1.4.1 with the following required packages.
 - Flux: `v0.10.4`
 - Zygote: `v0.4.20`
 - TimerOutputs: `v0.5.5` (TimerOutputs is required to quantify the difference in speeds between a normal DNN and SVD-DNN, it can be easily modified to not require TimerOutputs)

IJulia is required for the jupyter notebook examples.

Some of the examples may require the following packages (version numbers indicate the ones tested on, it may work on other versions too).
 - HDF5: `v0.13.2`
 - Images: `v0.22.2`
 - MLDatasets: `v0.5.2`
 - Plots: `v1.2.4`

## Acknowledgements

Thanks to [Abhijit Mohanty](https://github.com/mohantyabhijit074) and [Srinidhi Temkar](https://github.com/srinidhi-temkar) for helping in the project.
