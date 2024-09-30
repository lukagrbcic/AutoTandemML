import numpy as np
import subprocess
import sys
import time

sys.path.insert(0, '/src')
sys.path.insert(1, '/bc_problem')


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
            
            # r = np.random.randint(0, 100000)
            # subprocess.run(["source ./bc_problem/run.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # subprocess.run(["source ./bc_problem/run.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            command = "cd bc_problem;rm -rf postProcess*; laplacianFoam"
            process = subprocess.Popen(
            [command],  shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate()
            # 
            probes = np.loadtxt('bc_problem/postProcessing/probes1/0/T') 
            scalar_values = probes[-1,1:]
            # print (scalar_values)
            inputs.append(y)
            outputs.append(scalar_values)


   
        
        # print (np.array(inputs))
        # print (outputs)
        
        return np.array(inputs), outputs
    
    def save_data(self):
        
        inputs, outputs = self.run()
        
        r = np.random.randint(0, 100000)
        
        
        print (inputs)
        print (outputs)
        
        np.savetxt('initial_data/bcs.txt', inputs)
        np.savetxt('initial_data/probes.txt', np.array(outputs))

        
        
test = farm(20000).save_data()



        
