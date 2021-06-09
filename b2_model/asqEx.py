from utils import *
from brian2 import *
from b2_model.brian2_model import brian2_model

#defaultclock.dt = 0.01*ms

class adExModel(brian2_model):
    
    
    def __init__(self, param_dict=None):
        ## Default Values
        self.N = 1
        self._run_time = 3
        self.Cm = 18.
        self.EL = -65.
        self.VT = -50.
        self.VR = self.EL
        self.taum = 200.
        self.tauw = 150.
        self.a = 0.01
        self.b = 5.
        self.refractory = 1
        self.DeltaT = 5
        self.dt = 0.01
        self.activeSweep = 0
        self.realX = np.full((4,4), np.nan)
        self.realY = np.full((4,4), np.nan)
        self.realC = np.full((4,4000), 0)
        self.spike_times = np.full((4,4), np.nan)
        self.subthresholdSweep = 0
        self.spikeSweep = 5
        self.eqs = Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - V_T)/DeltaT) + I - w ) * (1./C) : volt
        dw/dt = ( a*(v - EL) - w ) / tauw : amp
        dV_T/dt = (VT - V_T)/tauVT : volt
        tauw : second
        tauVT : second
        a : siemens
        b : amp
        C : farad
        taum : second
        gL : siemens
        EL : volt
        VT : volt
        VR : volt
        Vcut : volt
        DeltaT : volt
        refrac : second
        bVT : volt
        I = in_current(t) : amp
        ''')
        

        #passed params
        if param_dict is not None:
            self.__dict__.update(param_dict) #if the user passed in a param dictonary we can update the objects attributes using it.
                # This will overwrite the default values if the user passes in for example {'Cm:' 99} the cm value above will be overwritten


    def run_current_sim(self, sweepNumber=None, param_dict=None):
        
        if param_dict is not None:
            self.__dict__.update(param_dict)
        if sweepNumber is not None and sweepNumber != self.activeSweep:
            self.activeSweep = sweepNumber
        
        
        start_scope()
        
        temp_in = self.realC[self.activeSweep,:]
        in_current = TimedArray(values = temp_in * pamp, dt=self.dt * ms)
        self.CRH = NeuronGroup( self.N, model=self.eqs, threshold='v>Vcut', reset='v=VR; w+=b; V_T+=bVT', refractory='refrac', method='euler' )
 
        # build network
        
        CRH = self.CRH; 
        #CRH.set_states(self.__dict__)
        CRH.tauw = self.tauw *ms; 
        CRH.b = self.b * pA; 
        CRH.a = self.a * nS; 
        CRH.C = self.Cm * pF; 
        CRH.taum = self.taum *ms;
        CRH.gL = (self.Cm * pF) / (self.taum * ms); 
        CRH.EL = self.EL* mV;
        CRH.VT = self.VT * mV;
        CRH.bVT = self.bVT *mV;
        CRH.Vcut = (np.array(self.VT) + 5 * np.array(self.DeltaT)) * mV; #0 *mV # 
        CRH.refrac = self.refractory*ms;
        CRH.DeltaT = self.DeltaT * mV; 
        CRH.tauVT = self.tauVT *ms;
        CRH.VR = self.VR *mV
        

        # init
        CRH.v = self.EL * mV
        

        # monitor
        M = SpikeMonitor( CRH )
        V = StateMonitor( CRH, ["v", "w"], record=True, dt=self.dt * ms, when='start')
        run(self._run_time * second)
        return M,V
       

