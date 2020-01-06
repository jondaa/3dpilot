import os
from exputil import run_exp

device='1'
acc_fac=10


init='stackofstars'
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.01,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=False)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.01,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.001,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.5,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)

acc_fac=5
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.01,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
acc_fac=20
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.01,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
acc_fac=40
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.01,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
acc_fac=100
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
run_exp(init,device,acc_fac, res=120,rec_lr=0.001,lr=0.01,batch_size=4,num_epochs=50,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True)
