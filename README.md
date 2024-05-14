Essential bicycle dynamics for microscopic simulation - experiments & plots
==============================



<img title="a title" src="file:///U:/PhDConnectedVRU/Publications/P2_BicycleDynamics_BMD2023/experiments/figures/review/test-scenarios_encroachemnt-subfig.png" alt="Alt text" width="390">

    *Snippet of Fig. 9 in Schmidt et al. (2023).*



Simulation experiments and plots of our publication "Essential bicycle dynamics for microscopic simulation - An example using the social force model." (Schmidt et al., 2023). 

Three python scripts create the content of our publication:

- **test_scenarios.py:** Simulation experiments, Figures 8 - 10, and Table 1.

- **stablility_limits_symbolic.py:** Figures 5, 6 and verification of the equations of the paper. 

- **repulsive_force_fileds.py:** Figure 3.

This software was created in the context of my PhD project at Delft University of Technology. 

## Requirements

Executing the scripts requires the following modules / packages. They are available from their own repositories or come bundled with the archived version of this software available at the 4TU.ResearchData repository.

- Python 3.11

- [cyclistsocialforce v1.1.1](https://github.com/chrismo-schmidt/cyclistsocialforce/releases/tag/v1.1.1-bmd2023proceedingspaper-review)

- [pypaperutils](https://github.com/chrismo-schmidt/pypaperutils.git)

- [pytrafficutils](https://github.com/chrismo-schmidt/pytrafficutils.git)

## Installation

1. Install the requirements in a new virtual environment of your preference. 

2. Run the scripts. 

## Authors

- Christoph M. Schmidt, c.m.schmidt@tudelft.nl (algorithm and software development, experiment design)

- Azita Dabiri (Supervision, consultation, review)

- Frederik Schulte (Supervision, consultation, review)

- Rieder Happee (Supervision, consultation, review)

- Jason K. Moore (Supervision, consultation, review, software development)

License
--------------------

This software is licensed under the terms of the MIT License.

### Additional Material

An archived version of this code can be found at the 4TU.ResearchData repository.

## Citation

If you use this software in your research, please cite it in your publications as:

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2023). Essential Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the Social Force Model [Preprint]. The Evolving Scholar - BMD 2023, 5th Edition. DOI: [10.59490/65a5124da90ad4aecf0ab147](https://doi.org/10.59490/65a5124da90ad4aecf0ab147)
