
import numpy as np




a = [1,2,3,4,5,6,7,8,9]

for x in range(len(a)):

    m = a[:x] + a[x+1:]
    print(m)

print(min(a))

