from brian2 import *


adEx = dict(eqs=Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - VT)/DeltaT) + I - w ) * (1./C) : volt (unless refractory)
        dw/dt = ( a*(v - EL) - w ) / tauw : amp (unless refractory)
        Vcut = VT + (DeltaT * 5) : volt 
        gL = C/taum : siemens
        tauw : second
        a : siemens
        b : amp
        C : farad
        taum : second
        EL : volt
        VT : volt
        VR : volt
        DeltaT : volt
        refrac : second
        I = in_current(t) : amp
        '''), threshold='v>Vcut', reset='v=VR; w+=b', refractory='refrac', method='euler', init_var=dict(v='EL'))

asqEx = dict(eqs=Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - vc)/DeltaT) + I - w ) * (1./C) : volt (unless refractory)
        dw/dt = ( a*(v - EL) - w ) / tauw : amp (unless refractory)
        dvc/dt = (VT - vc) / tauVT : volt (unless refractory)
        Vcut = VT + (DeltaT * 5) : volt (constant over dt)
        gL = C/taum : siemens (constant over dt)
        tauw : second
        tauVT : second
        a : siemens
        b : amp
        C : farad
        taum : second
        EL : volt
        VT : volt
        VR : volt
        DeltaT : volt
        refrac : second
        bVT : volt
        I = in_current(t) : amp
        '''), threshold='v>Vcut', reset='v=VR; w+=b; vc+=bVT', refractory='refrac', method='euler', init_var=dict(vc='VT', v='EL')) #+ (DeltaT * 5)

adIF = dict(eqs=Equations('''
        dv/dt = ( gL*(EL-v) + I - w ) * (1./Cm) : volt (unless refractory)
        dw/dt = ( a*(v - EL) - w ) / tauw : amp (unless refractory)
        tauw : second
        a : siemens
        b : amp
        C : farad
        taum : second
        gL : siemens
        EL : volt
        VT : volt
        VR : volt
        refrac : second
        I = in_current(t) : amp
        '''), threshold='v>VT', reset='v=VR; w+=b', refractory='refrac', method='euler')

