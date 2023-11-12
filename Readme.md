# Implementation and discussion of the "Pith Estimation on Rough Log End Images using Local Fourier Spectrum Analysis" method

Repository for the IPOL paper "Implementation and discussion of the "Pith Estimation on Rough Log End Images using Local Fourier Spectrum Analysis" method". Submitted on 

IPol Demo: [IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000472)

UruDendro ImageSet: [UruDendro][link_urudendro].

[link_urudendro]: https://iie.fing.edu.uy/proyectos/madera/


Version 1.0
Last update: 27/08/2023

Authors: 
-	Henry Marichal, henry.marichal@fing.edu.uy
-   Diego Passarella, diego.passarella@cut.edu.uy
-   Gregory Randall, randall@fing.edu.uy

## Get started

#### 1. Folders
All the python source files are in lib/ folder.

Algorithm 1 is implemented in the file **lib/pith_detector.py**

Algorithm 2 is implemented in the file **lib/fft_local_orientation.py**

Algorithm 3 is implemented in the file **lib/accumulator_space.py**

Algorithm 4 is implemented in the file **lib/peak_estimator.py**
## Installation

```bash
pip3 install --no-cache-dir -r requirements.txt
```

## Examples of usage

Here some examples of usage:
```bash
python main.py --filename ./Input/F02c.png --output_dir Output/
```
`

## License
License for th source code: [MIT](./LICENSE)

