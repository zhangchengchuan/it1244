import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([
4.52,
4.23,
2.47,
4.71,
3.40,
8.55,
6.87,
10.30,
8.92,
9.06])

x2 = np.array([
3.46,
2.91,
4.50,
5.81,
4.00,
5.83,
4.66,
3.64,
4.55,
1.92
])

def dist(a1, a2, b1, b2, type):
    if type == 'e':
        return np.sqrt(np.power(a1-b1,2)+np.power(a2-b2, 2))
    
    elif type == 'm':
        return abs(a1-b1)+abs(a2-b2)

    else:
        print("error")

point = (5.21, 4.32)
best_pair = None
min_d = 10000
for i in range(len(x1)):
    d = dist(point[0], point[1], x1[i], x2[i], 'e')
    print(d)
    if d < min_d:
        min_d = d
        best_pair = (x1[i], x2[i])


# standardization
standardized_x1 = (x1 - np.mean(x1)) / np.std(x1)
standardized_x2 = (x2 - np.mean(x2)) / np.std(x2)

normalized_x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
normalized_x2 = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))

print(standardized_x1, standardized_x2)
print(normalized_x1, normalized_x2)

plt.scatter(standardized_x1, standardized_x2)
plt.show()
plt.scatter(normalized_x1,normalized_x2)
plt.show()
# print(best_pair)

# knn
point = (9.09, 4.36)
res = [dist(a,b,point[0],point[1], 'e') for (a,b) in zip(x1, x2)]
sorted_res = np.argsort(res)
print(sorted_res)
