To compile the source code, first clone the Chrono repository, pointing at commit cc72eb4b10fd4b9fe498673375e587f1560c4c95
Then add the compressible_sph folder into the "src/chrono_fsi" folder inside the Chrono source directory, and replace the corresopnding CmakeLists.txt with the provided one.
Compile the Chrono libraries following the official guide, enabling the option CHRONO_FSI.
This project was built using Cuda version 12.6 and Visual Studio Enterprise 2022, LTSC 17.6, Version 17.6.21
The Test_compressibility folder can be added directly inside the "src" folder; it provides some simulation test cases.
