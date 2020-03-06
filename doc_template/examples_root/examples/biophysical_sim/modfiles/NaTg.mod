:Reference :Colbert and Pan 2002

NEURON	{
	SUFFIX NaTg
	USEION na READ ena WRITE ina
	RANGE gNaTgbar, gNaTg, ina, vshifth, vshiftm, slopeh, slopem
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTgbar = 0.00001 (S/cm2)
	vshifth = 0 (mV)
	vshiftm = 0 (mV)
	slopeh = 6
	slopem = 6
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTg	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNaTg = gNaTgbar*m*m*m*h
	ina = gNaTg*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((34-21)/10)
	
  UNITSOFF
    if(v == -38){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- (-38+vshiftm)))/(1-(exp(-(v- (-38+vshiftm))/slopem)))
		mBeta  = (0.124 * (-v + (-38+vshiftm)))/(1-(exp(-(-v + (-38+vshiftm))/slopem)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = mAlpha/(mAlpha + mBeta)

    if(v == -66){
      v = v + 0.0001
    }

		hAlpha = (-0.015 * (v- (-66+vshifth)))/(1-(exp((v- (-66+vshifth))/slopeh)))
		hBeta  = (-0.015 * (-v +(-66+vshifth)))/(1-(exp((-v +(-66+vshifth))/slopeh)))
		hTau = (1/(hAlpha + hBeta))/qt
		hInf = hAlpha/(hAlpha + hBeta)
	UNITSON
}