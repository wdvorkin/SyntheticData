# SynthericData: Private Synthetic Dataset Generation for Power Systems 

This repository stores data and code to replicate the results from the following preprint: 

[*Differentially Private Algorithms for Synthetic Power System Datasets*]()

by Vladimir Dvorkin and Audun Botterud (Massachusetts Institute of Technology, LIDS and Energy Initiative)

---

**Abstarct:** While power systems research relies on the availability of real-world network datasets, data owners (e.g., system operators) are hesitant to share data due to security and privacy risks. To control these risks, we develop privacy-preserving algorithms for the synthetic generation of optimization and machine learning datasets. Taking a real-world dataset as input, the algorithms output its noisy, synthetic version which preserves the accuracy of the real data on a specific downstream model or even a large population of those. We control the privacy loss using Laplace and Exponential mechanisms of differential privacy and preserve data accuracy using a post-processing convex optimization. We apply the algorithms to release synthetic network parameters and wind power data.

---


<table align="center">
    <tr>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/225117192-09c424fe-b071-4aff-ab1b-dda804cf9f58.gif">
        Wind power obfuscation (WPO) algorithm
        </td>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557705-11c922f0-59b8-4ad9-bb97-80e31e34f8ab.gif">
        Transmission capacity obfuscation (TCO) algorithm
        </td>
    </tr>
</table>


