import numpy as np
import subprocess
import sys
import time

sys.path.insert(0, './src')

import generate_bc as gbc


class farm:
    
    def __init__(self, loops):
        self.loops = loops
    
    def run(self):
        
        inputs = []
        outputs = []
        
        for i in range(self.loops):
            print (i)
            run = gbc.gen_bc('/bc_problem', './orig/')
            y = run.generate()

    
            
            r = np.random.randint(0, 100000)
            subprocess.run(["./bc_problem/run.sh"],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            probes = np.loadtxt('bc_problem/postProcessing/probes1/0/T') 
            scalar_values = probes[-1,1:]
            inputs.append(y)
            outputs.append([scalar_values])


   
        
        # print (np.array(inputs))
        # print (outputs)
        
        return np.array(inputs), outputs
    
    def save_data(self):
        
        inputs, outputs = self.run()
        
        r = np.random.randint(0, 100000)
        
        
        # print (inputs)
        print (outputs)
        
        np.savetxt('bcs.txt', inputs)
        np.savetxt('probes.txt', outputs)

        
        
test = farm(10).save_data()



        
