# Machine-Learning-for-5D-Calorimetry
Machine learning and statistical modeling for timing reconstruction, detector response characterization, and precision measurements in 5D calorimetry.
## Connection to the ATLAS 5D Calorimetry Project

This work is part of my contribution to the ATLAS 5D Calorimetry project, which incorporates precision timing information as an additional dimension in event reconstruction. The project combines timing measurements from the Liquid Argon (LAr) calorimeter, Tile calorimeter, and tracking detectors to improve event reconstruction performance in high-luminosity environments.

The broader goals of the project include improving Primary Vertex (PV) reconstruction in hard-scattering events and integrating timing-based refinements into the ATLAS Particle Flow (PFlow) algorithm. Studies have evolved from single-particle samples, such as charged pions, to jet reconstruction under both low and realistic pile-up conditions.

My contribution focuses on the application of machine learning techniques, particularly neural networks, to improve calorimeter timing resolution. This work involves extensive Monte Carlo-based studies to identify optimal signal selections using particle energy, position, and detector-response information. In addition, I investigate the impact of slow neutrons on timing measurements in hadronic jet events and develop methods to better understand and isolate timing contributions from secondary neutrons produced in hadronic cascades.

These studies support ongoing ATLAS research and development efforts aimed at enhancing detector performance for future high-luminosity operation.

## Workflow

```text
Calorimeter Cell Timing Data
             │
             ▼
      Data Cleaning
             │
             ▼
     Feature Extraction
             │
             ▼
 Machine Learning Models
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
 Neural   Gaussian  Polynomial
Network      Fit       Fit
      │      │      │
      └──────┴──────┘
             │
             ▼
 Timing Distribution Modeling
             │
             ▼
 Timing Resolution Estimation
             │
             ▼
 5D Calorimeter Performance Studies
```
