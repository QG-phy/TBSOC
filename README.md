# Wannier_Add_onsite_SOC
This library is to add on site spin-orbital coupling effect.

This code is for the system which is not possible or not easy to directly get the ab initio WannierTB with SOC, such as some topological system or systems with strong correlations. For these systems the non-soc calculations is stable and easy to perform, then you can use this code to add the SOC back on.

Copyright: Qiangqiang Gu, International Center for Quantum Materials, School of Physics, Peking University. 

Email:     guqq@pku.edu.cn

**
Note: up to now, this library only supports p- and d-orbital.
**

### Install: two ways to install
1. export PYTHONPATH=$PYTHONPATH:/yourpath/Wannier_Add_onsite_SOC/lib

2. 
```python
import sys
sys.path.append('/yourpath/Wannier_Add_onsite_SOC/lib')
```

### How to use:
  **Please see the tutorial and run it using jupiter-notebook.**

