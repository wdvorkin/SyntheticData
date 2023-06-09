# SyntheticData: Private Synthetic Data for Power Systems 

This repository stores data and code to replicate the results from the following paper: 

[*Differentially Private Algorithms for Synthetic Power System Datasets*](https://wdvorkin.github.io/media/L-CCS-2023-SynData.pdf)

by Vladimir Dvorkin and Audun Botterud (Massachusetts Institute of Technology, LIDS and Energy Initiative)

[published](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10146422) in the IEEE Control Systems Letters journal.

---

**Abstarct:** While power systems research relies on the availability of real-world network datasets, data owners (e.g., system operators) are hesitant to share data due to security and privacy risks. To control these risks, we develop privacy-preserving algorithms for the synthetic generation of optimization and machine learning datasets. Taking a real-world dataset as input, the algorithms output its noisy, synthetic version which preserves the accuracy of the real data on a specific downstream model or even a large population of those. We control the privacy loss using Laplace and Exponential mechanisms of differential privacy and preserve data accuracy using a post-processing convex optimization. We apply the algorithms to release synthetic network parameters and wind power data.

---


<table align="center">
    <tr>
        <td align="center" width="500"><img src="https://user-images.githubusercontent.com/31773955/225124111-59df9b3e-7bff-4d1f-ab48-29bb0b904730.gif">
        Wind power obfuscation
        </td>
        <td align="center" width="500"><img src="https://user-images.githubusercontent.com/31773955/225128660-cf9f4b65-0e61-4afc-829f-59925ceede6e.gif">
        Transmission capacity obfuscation
        </td>
        <td align="center" width="500"><img src="https://user-images.githubusercontent.com/31773955/236578392-b490f95f-675e-44b0-908a-abaeb1914f08.gif">
        Electric load obfuscation
        </td>
    </tr>
</table>


---

## Installation and usage

All models are implemented in Julia Language v.1.8 using [JuMP](https://github.com/jump-dev/JuMP.jl) modeling language for mathematical optimization and commercial [Mosek](https://github.com/MOSEK/Mosek.jl) and [Gurobi](https://github.com/jump-dev/Gurobi.jl) optimization solvers, which need to be licensed (free for academic use). 

The codes to implement the two algorithms are placed in ```WPO``` and ```TCO``` folders, respectively. Make sure to active project environment using ```Project.toml``` and ```Manifest.toml``` located in each folder. 

To run the WPO algorithm, ```cd``` to ```WPO``` and type the following command in the terminal:

```julia wpo_main.jl -a 15.0 -e 1.0```

which asks to compute the results for adjacency parameter 15.0 and privacy loss 1.0. 

Similarly, ```cd``` to ```TCO``` and type:

```julia tco_main.jl -a 15.0 -e 1.0```

to run the TCO algorithm. For more information on the settings, type

```julia tco_main.jl --help```

---

## Citation
```
@article{dvorkin2023differentially,
  title={Differentially Private Algorithms for Synthetic Power System Datasets},
  author={Dvorkin, Vladimir and Botterud, Audun},
  journal={arXiv preprint arXiv:2303.11079},
  year={2023}
}
```
