# This is the main file for HW 3

import numpy as np
from utils import integrate, extract

## Part 1
'''
step_size = 0.05
x_array = np.linspace(1, 0, 21)

print(x_array)

integrate = integrate(x_array,step_size)
integrate.integrate([1,2,3,4])
integrate.plot()


'''
## part 2
#file_location = 'Empty2.asc'
file_location = 'TableWithObjects2.asc'
file_location = 'CSE.asc'
extractor = extract()
extractor.load_file(file_location)
extractor.ransac(n = 5)
extractor.plot_data_2()
