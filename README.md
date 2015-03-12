cosmogenic_dating
=================

A few IPython notebooks to show examples of inference of
exposure times and erosion rates from measured
cosmogenic nuclides concentrations vs. depth.

It compares both Bayesian and Frequentist (Maximum
Likelihood Estimation) approachs
with different optimization and/or sampling algorithms
(grid search, non-linear fitting, MCMC...).


Links to static versions of the notebooks
-----------------------------------------

- [General notes about the statistical inference methods][1]
- [A Model of nuclides concentration vs. depth][2] 
- [Datasets][3]

Maximum Likelihood Estimation w/ Grid Search:
- [2 free parameters][4]
- [4 free parameters][5]

Maximum Likelihood Estimation w/ non-linear fitting:
- [2 free parameters][6]
- [4 free parameters][7]

Bayesian approach w/ MCMC:
- [2 free parameters][8]
- [4 free parameters][9]

Bayesian approach w/ MCMC (more robust, based on [emcee][10]:
- [2 free parameters][11]
- [4 free parameters][12]


[1]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/Inference_Notes.ipynb
[2]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/Models.ipynb
[3]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/Datasets.ipynb
[4]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/GS_test_2params.ipynb
[5]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/GS_test_4params.ipynb
[6]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/MLE_test_2params.ipynb
[7]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/MLE_test_4params.ipynb
[8]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/Bayes_test_2params.ipynb
[9]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/Bayes_test_4params.ipynb
[10]: http://dan.iel.fm/emcee/current/
[11]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/emcee_test_2params.ipynb
[12]: http://nbviewer.ipython.org/github/benbovy/cosmogenic_dating/blob/master/emcee_test_4params.ipynb


Licence
-------

Author: B. Bovy

License: MIT (see LICENSE)
