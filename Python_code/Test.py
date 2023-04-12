import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]
inner_coord = [(i,j) for i in range(1,5) for j in range(1,5)]

print(inner_coord)

# print(range((5,5)))

#for i, j in np.ndindex(5, 5):
#for i,j in range((3,2)):
for i,j in inner_coord:
    print(i,j)
    #print(i,j)