[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fshgo%2Fgamtl.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fshgo%2Fgamtl?ref=badge_shield)
[![Build Status](https://travis-ci.com/shgo/gamtl.svg?branch=master)](https://travis-ci.com/shgo/gamtl)

# GAMTL

**Paper accepted for the 28th International Joint Conference on Artificial Intelligence - IJCAI 2019**
Title: **Group LASSO with Asymmetric Structure Estimation for Multi-Task Learning**
Authors: *Saullo Oliveira, André Gonçalves, Fernando Von Zuben**.

Abstract:
    Group LASSO is a widely used regularization that imposes sparsity considering groups of covariates. When used in Multi-Task Learning (MTL) formulations, it makes an underlying assumption that if one group of covariates is not relevant for one or a few tasks, it is also not relevant for all tasks, thus implicitly assuming that all tasks are related.
    This implication can easily lead to negative transfer if this assumption does not hold for all tasks.
    Since for most practical applications we hardly know a priori how the tasks are related, several approaches have been conceived in the literature to ($i$) properly capture the transference structure, ($ii$) improve interpretability of the tasks interplay, and ($iii$) penalize potential negative transfer.
    Recently, the automatic estimation of asymmetric structures inside the learning process was capable of effectively avoiding negative transfer.
    Our proposal is the first attempt in the literature to conceive a Group LASSO with asymmetric transference formulation, looking for the best of both worlds in a  framework that admits the overlap of groups. 
    The resulting optimization problem is solved by an alternating procedure with fast methods. 
    We performed experiments using synthetic and real datasets to compare our proposal with state-of-the-art approaches, evidencing the promising predictive performance and distinguished interpretability of our proposal.
    The real case study involves the prediction of cognitive scores for  Alzheimer's disease progression assessment.

For replication instructions see [Reproducing](reproducing.md).

Date: 06/2019
License: GNU General Public License v3.0

Acknowledgements
We acknowledge the grants \#141881\/2015-1 and \#307228\/2018-5 from the Brazilian National Council for Scientific and Technological Development (CNPq), grant \#2013\/07559-3 from São Paulo Reseach Foundation (FAPESP), and the Coordination for the Improvement of Higher Education Personnel (CAPES).
