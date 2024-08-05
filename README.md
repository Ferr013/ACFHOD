# ACFHOD (Angular Correlation Function with Halo Occupation Distribution)

``ACFHOD`` is a software package to calculate the angular correlation function using the HOD formalism.

Installation
------------

``pip install git+https://github.com/Ferr013/ACFHOD``

Note on `gfortran`:

The code dependencies on ``hmf`` and ``scipy`` require a Fortran compiler.
I had some issues with conda linking of `gfortran` in MacOS Sonoma.
You might have to bypass conda with ``conda deactivate`` and install it with ``brew install gcc``.

Don't hesitate to reach out if you have questions!

email: [gferrami@student.unimelb.edu.au](mailto::gferrami@student.unimelb.edu.au)