import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc


class gen_bc:
    
    def __init__(self, case_location, orig_location):
        
        self.case_location = case_location
        self.orig_location = orig_location
        
        
    def _linear_variation(self, x, a, b):
        
        return a * x + b
    
    def _sinusoidal_variation(self, x, a, b, k):
        
       #b = np.abs(a)
        return a * np.sin(k * x) + b

    def _parabolic_variation(self, x, a, b, c):
        
        return a * x**2 + b * x + c


    def _add_noise(self, y, noise_level):
        
        noise = np.random.normal(0, noise_level, y.shape)
        return y + noise
    
    def _bc_values(self):
        
        range_ = np.linspace(0,1,20)
        a = np.random.uniform(0, 10)
        b = np.random.uniform(0, 10)
        c = np.random.uniform(0, 10)
        k = np.random.uniform(0, 10)
        noise = np.random.uniform(0, 100)
        
        r = np.random.choice([0, 1, 2])
        if r == 0: 
            y = self._linear_variation(range_, a, b)
            y = self._add_noise(y, noise)
        elif r == 1:
            y = self._parabolic_variation(range_, a, b, c)
            y = self._add_noise(y, noise)
        else:
            y = self._sinusoidal_variation(range_, a, b, k)
            y = self._add_noise(y, noise)
        
        for i in y:
            if i > 30:
                i == np.random.uniform(0, 30)

        if np.random.uniform(0,1) < 0.5:
            y = y[::-1]

        return np.abs(y)

        
    def generate(self):
       # print (self.orig_location)
        read_file = open('%sT.orig' % self.orig_location)
        data = read_file.readlines()
        data = [data[i].split() for i in range(len(data))]
        start = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] == 'value':
                    start = i+3

        len_ = float(data[start-2][0])
        stop = int(start+len_-1)


        y = self._bc_values()
        
        
        for i in range(start,stop+1):
            data[i] = str(np.round(y[abs(start-i)], decimals=3))

        read_file_2 = open('%sT.orig' % self.orig_location).readlines()
        for i in range(start,stop+1):
            read_file_2[i] = 2*'\t' + str(np.round(y[abs(start-i)], decimals=3)) + '\n'


        write_file = open('.%s/0/T' % self.case_location, 'w')
        for i in read_file_2:
            write_file.write(i)
        write_file.close()

        return y


