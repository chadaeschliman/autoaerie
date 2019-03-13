import numpy as np
from scipy import optimize

def ctof(c):
    return 1.8*c + 32.
def ftoc(f):
    return (f-32.0)/1.8

def est_clo(to_f, chair=0.15):
    to = ftoc(to_f)
    # converted from JS here: http://comfort.cbe.berkeley.edu/
    if to < -5.0:
        clo = 1.0
    elif to < 5.0:
        clo = 0.8181 - 0.0364*to
    elif to < 26.0:
        clo = np.power(10, -0.1635-0.0066*to)
    else:
        clo = 0.46
    return clo+chair

def calc_pmv_ppd_f(ta_f, rh, tr_f=None, tr_alpha=0.05, met=1.1, wme=0, vel=0.1, clo=None, to_f=None):
    # converted from JS here: http://comfort.cbe.berkeley.edu/
    if (tr_f is None or clo is None) and to_f is None:
        return None
    if tr_f is None:
        tr_f = tr_alpha*to_f + (1.0-tr_alpha)*ta_f
    to = ftoc(to_f)
    if clo is None:
        clo = est_clo(to)
    ta = ftoc(ta_f)
    tr = ftoc(tr_f)
    print 'clo',clo

    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (ta + 235))

    icl = 0.155 * clo # thermal insulation of the clothing in M2K/W
    m = met * 58.15 # metabolic rate in W/M2
    w = wme * 58.15 # external work in W/M2
    mw = m - w # internal heat production in the human body
    if (icl <= 0.078):
        fcl = 1 + (1.29 * icl)
    else:
        fcl = 1.05 + (0.645 * icl)

    # heat transf. coeff. by forced convection
    hcf = 12.1 * np.sqrt(vel)
    taa = ta + 273.0
    tra = tr + 273.0
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * np.power(tra / 100, 4)
    xn = tcla / 100.0
    xf = tcla / 50.0
    eps = 0.00015

    for n in xrange(150):
        xf = (xf + xn) / 2
        hcn = 2.38 * np.power(abs(100.0 * xf - taa), 0.25)
        if (hcf > hcn):
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * np.power(xf, 4)) / (100.0 + p3 * hc)
        if (abs(xf-xn) <= eps):
            break

    tcl = 100.0 * xn - 273.0

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    # heat loss by sweating
    if (mw > 58.15):
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - ta)
    # heat loss by radiation
    hl5 = 3.96 * fcl * (np.power(xn, 4) - np.power(tra / 100.0, 4))
    # heat loss by convection
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * np.power(pmv, 4.0) - 0.2179 * np.power(pmv, 2.0))

    return pmv, ppd

def objective(ta_f, target_ppd, sign, rh, tr_f=None, tr_alpha=0.05, met=1.1, wme=0, vel=0.1, clo=None, to_f=None):
    pmv, ppd = calc_pmv_ppd_f(ta_f, rh, tr_f, tr_alpha, met, wme, vel, clo, to_f)
    if np.sign(pmv) != np.sign(sign):
        return 100.0 - target_ppd + sign*pmv
    else:
        return abs(ppd - target_ppd)

def calc_min_set_f(max_ppd, rh, tr_f=None, tr_alpha=0.05, met=1.1, wme=0, vel=0.1, clo=None, to_f=None):
    if (tr_f is None or clo is None) and to_f is None:
        return None

    x0 = ctof(20)
    res = optimize.fmin(objective, x0, args=(max_ppd, -1, rh, tr_f, tr_alpha, met, wme, vel, clo, to_f), xtol=0.01, full_output=True)
    return res

def calc_max_set_f(max_ppd, rh, tr_f=None, tr_alpha=0.05, met=1.1, wme=0, vel=0.1, clo=None, to_f=None):
    if (tr_f is None or clo is None) and to_f is None:
        return None

    x0 = ctof(30)
    res = optimize.fmin(objective, x0, args=(max_ppd, 1, rh, tr_f, tr_alpha, met, wme, vel, clo, to_f), xtol=0.01, full_output=True)
    return res

def find_saturated_vapor_pressure_torr(T):
    # converted from JS here: http://comfort.cbe.berkeley.edu/
    # calculates Saturated Vapor Pressure (Torr) at Temperature T  (C)
    return np.exp(18.6686 - 4030.183 / (T + 235.0))

def calculate_set(ta_f, tr_f, vel, rh, met, clo, wme):
    ta = ftoc(ta_f)
    tr = ftoc(tr_f)
    # converted from JS here: http://comfort.cbe.berkeley.edu/
    VaporPressure = rh * find_saturated_vapor_pressure_torr(ta) / 100.0
    AirVelocity = max(vel, 0.1)
    KCLO = 0.25
    BODYWEIGHT = 69.9
    BODYSURFACEAREA = 1.8258
    METFACTOR = 58.2
    SBC = 0.000000056697 # Stefan-Boltzmann constant (W/m2K4)
    CSW = 170.0
    CDIL = 120.0
    CSTR = 0.5

    TempSkinNeutral = 33.7 # setpoint (neutral) value for Tsk
    TempCoreNeutral = 36.8 # setpoint value for Tcr
    TempBodyNeutral = 36.49 # setpoint for Tb (.1*TempSkinNeutral + .9*TempCoreNeutral)
    SkinBloodFlowNeutral = 6.3 # neutral value for SkinBloodFlow

    # INITIAL VALUES - start of 1st experiment
    TempSkin = TempSkinNeutral
    TempCore = TempCoreNeutral
    SkinBloodFlow = SkinBloodFlowNeutral
    MSHIV = 0.0
    ALFA = 0.1
    ESK = 0.1 * met

    # Start new experiment here (for graded experiments)
    # UNIT CONVERSIONS (from input variables)

    p = 101325.0 / 1000.0 # TH : interface?

    PressureInAtmospheres = p * 0.009869
    LTIME = 60
    TIMEH = LTIME / 60.0
    RCL = 0.155 * clo
    # AdjustICL(RCL, Conditions)  TH: I don't think this is used in the software

    FACL = 1.0 + 0.15 * clo # % INCREASE IN BODY SURFACE AREA DUE TO CLOTHING
    LR = 2.2 / PressureInAtmospheres # Lewis Relation is 2.2 at sea level
    RM = met * METFACTOR
    M = met * METFACTOR

    if (clo <= 0):
        WCRIT = 0.38 * np.power(AirVelocity, -0.29)
        ICL = 1.0
    else:
        WCRIT = 0.59 * np.power(AirVelocity, -0.08)
        ICL = 0.45

    CHC = 3.0 * np.power(PressureInAtmospheres, 0.53)
    CHCV = 8.600001 * np.power((AirVelocity * PressureInAtmospheres), 0.53)
    CHC = max(CHC, CHCV)

    # initial estimate of Tcl
    CHR = 4.7
    CTC = CHR + CHC
    RA = 1.0 / (FACL * CTC) # resistance of air layer to dry heat transfer
    TOP = (CHR * tr + CHC * ta) / CTC
    TCL = TOP + (TempSkin - TOP) / (CTC * (RA + RCL))

    # ========================  BEGIN ITERATION
    #
    # Tcl and CHR are solved iteratively using: H(Tsk - To) = CTC(Tcl - To),
    #  where H = 1/(Ra + Rcl) and Ra = 1/Facl*CTC
    #

    TCL_OLD = TCL
    flag = True
    for TIM in xrange(1,LTIME+1):
        for _ in xrange(10000):
            if flag:
                TCL_OLD = TCL
                CHR = 4.0 * SBC * pow(((TCL + tr) / 2.0 + 273.15), 3.0) * 0.72
                CTC = CHR + CHC
                RA = 1.0 / (FACL * CTC) # resistance of air layer to dry heat transfer
                TOP = (CHR * tr + CHC * ta) / CTC
            TCL = (RA * TempSkin + RCL * TOP) / (RA + RCL);
            flag = True
            if abs(TCL - TCL_OLD) <= 0.01:
                break

        flag = False
        DRY = (TempSkin - TOP) / (RA + RCL)
        HFCS = (TempCore - TempSkin) * (5.28 + 1.163 * SkinBloodFlow)
        ERES = 0.0023 * M * (44.0 - VaporPressure)
        CRES = 0.0014 * M * (34.0 - ta)
        SCR = M - HFCS - ERES - CRES - wme
        SSK = HFCS - DRY - ESK
        TCSK = 0.97 * ALFA * BODYWEIGHT
        TCCR = 0.97 * (1 - ALFA) * BODYWEIGHT
        DTSK = (SSK * BODYSURFACEAREA) / (TCSK * 60.0) #deg C per minute
        DTCR = SCR * BODYSURFACEAREA / (TCCR * 60.0) #deg C per minute
        TempSkin = TempSkin + DTSK
        TempCore = TempCore + DTCR
        TB = ALFA * TempSkin + (1 - ALFA) * TempCore
        SKSIG = TempSkin - TempSkinNeutral
        WARMS = (SKSIG > 0) * SKSIG
        COLDS = ((-1.0 * SKSIG) > 0) * (-1.0 * SKSIG)
        CRSIG = (TempCore - TempCoreNeutral)
        WARMC = (CRSIG > 0) * CRSIG
        COLDC = ((-1.0 * CRSIG) > 0) * (-1.0 * CRSIG)
        BDSIG = TB - TempBodyNeutral
        WARMB = (BDSIG > 0) * BDSIG
        COLDB = ((-1.0 * BDSIG) > 0) * (-1.0 * BDSIG)
        SkinBloodFlow = (SkinBloodFlowNeutral + CDIL * WARMC) / (1 + CSTR * COLDS)
        if (SkinBloodFlow > 90.0):
            SkinBloodFlow = 90.0
        if (SkinBloodFlow < 0.5):
            SkinBloodFlow = 0.5
        REGSW = CSW * WARMB * np.exp(WARMS / 10.7)
        if (REGSW > 500.0):
            REGSW = 500.0
        ERSW = 0.68 * REGSW
        REA = 1.0 / (LR * FACL * CHC) # evaporative resistance of air layer
        RECL = RCL / (LR * ICL) # evaporative resistance of clothing (icl=.45)
        EMAX = (find_saturated_vapor_pressure_torr(TempSkin) - VaporPressure) / (REA + RECL)
        PRSW = ERSW / EMAX
        PWET = 0.06 + 0.94 * PRSW
        EDIF = PWET * EMAX - ERSW
        ESK = ERSW + EDIF
        if (PWET > WCRIT):
            PWET = WCRIT
            PRSW = WCRIT / 0.94
            ERSW = PRSW * EMAX
            EDIF = 0.06 * (1.0 - PRSW) * EMAX
            ESK = ERSW + EDIF

        if (EMAX < 0):
            EDIF = 0
            ERSW = 0
            PWET = WCRIT
            PRSW = WCRIT
            ESK = EMAX

        ESK = ERSW + EDIF
        MSHIV = 19.4 * COLDS * COLDC
        M = RM + MSHIV
        ALFA = 0.0417737 + 0.7451833 / (SkinBloodFlow + .585417)

    # Define new heat flow terms, coeffs, and abbreviations
    STORE = M - wme - CRES - ERES - DRY - ESK  # rate of body heat storage

    HSK = DRY + ESK # total heat loss from skin
    RN = M - wme # net metabolic heat production
    ECOMF = 0.42 * (RN - (1 * METFACTOR))
    if (ECOMF < 0.0):
        ECOMF = 0.0 #from Fanger
    EREQ = RN - ERES - CRES - DRY
    EMAX = EMAX * WCRIT
    HD = 1.0 / (RA + RCL)
    HE = 1.0 / (REA + RECL)
    W = PWET
    PSSK = find_saturated_vapor_pressure_torr(TempSkin)
    # Definition of ASHRAE standard environment... denoted "S"
    CHRS = CHR
    if (met < 0.85):
        CHCS = 3.0
    else:
        CHCS = 5.66 * pow(((met - 0.85)), 0.39)
        if (CHCS < 3.0):
            CHCS = 3.0

    CTCS = CHCS + CHRS
    RCLOS = 1.52 / ((met - wme / METFACTOR) + 0.6944) - 0.1835
    RCLS = 0.155 * RCLOS
    FACLS = 1.0 + KCLO * RCLOS
    FCLS = 1.0 / (1.0 + 0.155 * FACLS * CTCS * RCLOS)
    IMS = 0.45
    ICLS = IMS * CHCS / CTCS * (1 - FCLS) / (CHCS / CTCS - FCLS * IMS)
    RAS = 1.0 / (FACLS * CTCS)
    REAS = 1.0 / (LR * FACLS * CHCS)
    RECLS = RCLS / (LR * ICLS)
    HD_S = 1.0 / (RAS + RCLS)
    HE_S = 1.0 / (REAS + RECLS)

    # SET* (standardized humidity, clo, Pb, and CHC)
    # determined using Newton#s iterative solution
    # FNERRS is defined in the GENERAL SETUP section above

    DELTA = .0001
    dx = 100.0
    X_OLD = TempSkin - HSK / HD_S #lower bound for SET
    while (abs(dx) > .01):
        ERR1 = (HSK - HD_S * (TempSkin - X_OLD) - W * HE_S * (PSSK - 0.5 * find_saturated_vapor_pressure_torr(X_OLD)))
        ERR2 = (HSK - HD_S * (TempSkin - (X_OLD + DELTA)) - W * HE_S * (PSSK - 0.5 * find_saturated_vapor_pressure_torr((X_OLD + DELTA))))
        X = X_OLD - DELTA * ERR1 / (ERR2 - ERR1)
        dx = X - X_OLD
        X_OLD = X

    return ctof(X)

def reverse_set(target, to, alpha, vel, rh, met, clo, wme):
    x0 = target
    if clo is None:
        clo = est_clo(to)
    res = optimize.fmin(lambda x: (target-calculate_set(x, (1-alpha)*x+alpha*to, vel, rh, met, clo, wme))**2, x0=x0, xtol=0.01, full_output=False, disp=False)
    return res[0]
