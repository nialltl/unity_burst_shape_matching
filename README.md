Burst Softbodies - Parallelised Shape-Matching
========

This is a commented implementation of the paper **[Meshless Deformations Based on Shape Matching]** in Unity, using the **C# Job System** and **Burst compiler** available in Unity 2018.1 onwards. It also uses **GPU based indirect drawing**, to speedily visualise points pulling from a texture. It can handle several hundred thousand particles in under 16ms (machine-dependent!).

![gif](https://i.imgur.com/DadbwwZ.gif)

[Meshless Deformations Based on Shape Matching]: http://www.matthias-mueller-fischer.ch/publications/MeshlessDeformations_SIG05.pdf

Implementation
---
- I'm using a **Position Based Dynamics** integration scheme, kind of arbitrarily -- as a result the system's stiffness is iteration-dependent but generally quite friendly to deal with, and suffers from some inherent motion damping.
- Particles are processed in parallel using the **job system**. Additionally, steps of the algorithm requiring linear summations have been partially parallelised by splitting sums into batches.
- This implementation includes support for **linear deformation**, extending the range of possible motion (stretch and shear). Support for twisting and bending would require the quadratic extension, detailed in the original paper.
- I've only tested this in Unity 2018.2.0f1! I hope it works for you too!!

Misc notes
---
- I tried using IJobParallelForBatch to perform the center-of-mass and shape matrix summations, but couldn't quite get this working. Maybe you can?
- There's a little bit of drift occuring when performing the parallel center-of-mass summation which doesn't occur in the linear equivalent. I think this is due to floating point precision, but is maybe worth investigating.
- This project is under an MIT license, feel free to do whatever you like with it -- let me know if you make anything cool! x