from utils import *
from brian2 import *


adEx = dict(eqs=Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - VT)/DeltaT) + I - w ) * (1./C) : volt
        dw/dt = ( a*(v - EL) - w ) / tauw : amp
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
        '''), threshold='v>Vcut', reset='v=VR; w+=b', refractory='refrac', method='euler')

asqEx = dict(eqs=Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - V_T)/DeltaT) + I - w ) * (1./C) : volt
        dw/dt = ( a*(v - EL) - w ) / tauw : amp
        dV_T/dt = (V_T - VT)/tauVT : volt
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
        '''), threshold='v>Vcut', reset='v=VR; w+=b; V_T+=bVT', refractory='refrac', method='euler')

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

