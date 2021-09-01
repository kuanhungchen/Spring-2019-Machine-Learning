import matplotlib.pyplot as plt
import numpy as np


# p0 = [1.1204014, 1.1325435, 1.1331347, 1.1439200, 1.1574480]
# p1 = [1.0249238, 1.0208748, 1.0195541, 1.0176683, 1.0134736]
# p2 = [1.0574598, 1.0373649, 1.0361757, 1.0328835, 1.0304045]
# p3 = [1.0595380, 1.0436696, 1.0514365, 1.0415148, 1.0371799]
# p4 = [1.0816304, 1.0618980, 1.0592042, 1.0582547, 1.0545450]

# p0 = [1.1329724, 1.1406240, 1.1256884, 1.1334824, 1.1170379]
# p1 = [0.9625389, 0.9562185, 0.9541032, 0.9535967, 0.9544435]
# p2 = [0.9619335, 0.9530141, 0.9533895, 0.9526119, 0.9542985]
# p3 = [0.9678584, 0.9684765, 0.9638261, 0.9596978, 0.9557523]
# p4 = [0.9575865, 0.9575865, 0.9583439, 0.9569934, 0.9546289]

p0 = [1.3520962
,1.3511810
,1.3509623
,1.3500678
,1.3459536]
p1 = [1.3480199
,1.3485086
,1.3482138
,1.3462695
,1.3459536]
p2=[1.3607491,
    1.3599562,
    1.3599181,
    1.3604251,
    1.3592932]
p3 = [1.3668930,
      1.3643353,
      1.3651682,
      1.3666343,
      1.3665135]
p4 = [1.3717462,
      1.3730675,
      1.3711506,
      1.3704212,
      1.3709724]

n = [150, 250, 350, 450, 550]


plt.figure()
plt.plot(n, p0, label='1 group')
plt.plot(n, p1, label='2 groups')
plt.plot(n, p2, label='3 groups')
plt.plot(n, p3, label='4 groups')
plt.plot(n, p4, label='5 groups')
plt.xlabel("Number of estimators")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

"""

Base: 4, n_estimators: 150, criterion: etpy ==> 0.9575865
Base: 4, n_estimators: 250, criterion: etpy ==> 0.9575065
Base: 4, n_estimators: 350, criterion: etpy ==> 0.9583439
Base: 4, n_estimators: 450, criterion: etpy ==> 0.9569934
Base: 4, n_estimators: 550, criterion: etpy ==> 0.9546289
"""