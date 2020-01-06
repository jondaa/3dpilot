import os
from exputil import run_exp

device='0'
acc_fac=10

init='sticks'
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=False)
init='radial'
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=False)

init='2dradial'
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=False)


acc_fac=10
weight=0.1
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=weight,wvel=weight*4,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
weight=0.3
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=weight,wvel=weight*4,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
weight=0.6
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=weight,wvel=weight*4,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)