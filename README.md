# TBSOC (former name Wannier_Add_onsite_SOC)
This library is to support direct calculation the  spin-orbital coupling strength in materials and used for on site matrix for Wannier TB models.

This code is for the system which is not possible or not easy to directly get the ab initio WannierTB with SOC, such as some topological system or systems with strong correlations. For these systems the non-soc calculations is stable and easy to perform, then you can use this code to calculate the SOC strength.

Copyright: Qiangqiang Gu, International Center for Quantum Materials, School of Physics, Peking University. 
Email:     guqq@pku.edu.cn

## feature:
- calculate the $ab$ $initio$ SOC strength for $p$-and $d$-orbital.
- output the SOC TB model, with basis defined $| s_z \rangle$ basis.
- supports for custom local axis for Wannier TB. 
- support the custom spin quantization axis for  Wannier TB.


### Install: two ways to install
1. export PYTHONPATH=$PYTHONPATH:/yourpath/TBSOC/lib

2. 
```python
import sys
sys.path.append('/yourpath/TBSOC/lib')
```

### How to use:
  **Please see the tutorial and run it using jupiter-notebook.**

