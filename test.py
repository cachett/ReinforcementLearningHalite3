







a = [[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4], [5,5,5,5,5]]
sight_window = 17
ship_x = 0
ship_y = 0
array = []
for i in range(-(sight_window//2), sight_window//2 + 1):
    for j in range(-(sight_window//2) + abs(i), sight_window//2 - abs(i) + 1):
        array.append((i+ship_x,j+ship_y))

print(array)
print(len(array))
