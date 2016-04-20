AFM and AFM+S in Python
=======================

This is a python implementation of afm and afm+s, wherever possible I tried to
maintain scikit-learn convention, so that the code would be compatible with
their helper functions.

custom\_logistic.py is an estimator that can be used to implement AFM.
bounded\_logistic.py is an estimator that can be used to implement AFM+S.

process\_datashop.py can be used to read a student step export from datashop
and to run AFM and AFM+S on it. 

Citing this Software 
====================

If you use this software in a scientific publiction, then we would appreciate
citation of the following paper:

MacLellan, C.J., Liu, R., Koedinger, K.R. (2015) Accounting for Slipping and
Other False Negatives in Logistic Models of Student Learning. In O.C. Santos et
al. (Eds.), Proceedings of the 8th International Conference on Educational Data
Mining. Madrid, Spain: International Educational Data Mining Society [(pdf)](http://christopia.net/media/publications/maclellan2-2015.pdf).

Bibtex entry:

    @inproceedings{afmslip:2015,
    author={MacLellan, C.J. and Liu, R. and Koedinger, K.R.},
    title={Accounting for Slipping and Other False Negatives in Logistic Models
    of Student Learning.},
    booktitle={Proceedings of the 8th International Conference on Educational
    Data Mining},
    editor={Santos, O.C. and Boticario, J.G. and Romero, C. and Pechenizkiy, M.
    and Merceron, A. and Mitros, P. and Luna, J.M. and Mihaescu, C. and Moreno,
    P. and Hershkovitz, A. and Ventura S. and Desmarais, M.},
    year={2015},
    publisher={Interational Educational Data Mining Society},
    address={Madrid, Spain}
    }

