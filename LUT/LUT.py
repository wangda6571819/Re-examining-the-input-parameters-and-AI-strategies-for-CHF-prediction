import numpy as np
from iapws import IAPWS97
from scipy.interpolate import RegularGridInterpolator

# # Example usage:
# H = 1000  # Example value for H (J/kg)
# D_exp = 0.01  # Example value for D (10 mm)
# P_exp = 2e6  # Example value for P (2 MPa)
# G_exp = 1000  # Example value for G (kg/m²s)
# H_in = 800  # Example value for H_in (J/kg)
# L_h = 1  # Example value for L_h (m)

def getLUTCHF(H = 1000, P_exp= 1e6, D_exp = 0.008, G_exp = 1000, H_in = 800, L_h = 1) : 
    print(f'H = {H} , D = {D_exp} , P = {P_exp} , G = {G_exp}, H = {H_in}, L = {L_h}')
    # Constants
    D_ref = 8e-3  # Reference diameter in meters (8 mm)
    CHF_initial = 500e3  # Initial CHF estimate in W/m²
    tolerance = 1e-3  # Convergence tolerance

    # Load LUT data from file
    q_raw = np.loadtxt('./2006LUTdata.txt') * 1e3
    P = np.array((0.10, 0.30, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 21.0)) * 1e6
    G = np.array((0, 50, 100, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000))
    x = np.array((-0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00))


    # Define the functions to calculate H_f, H_fg, and X using IAPWS97
    def H_f(P):
        # Enthalpy of liquid water at pressure P
        water = IAPWS97(P=P/1e6, x=0)  # P in MPa, x=0 for saturated liquid
        return water.h * 1000  # Convert to J/kg

    def H_fg(P):
        # Latent heat of vaporization at pressure P
        water = IAPWS97(P=P/1e6, x=0)  # Saturated liquid
        steam = IAPWS97(P=P/1e6, x=1)  # Saturated vapor
        return (steam.h - water.h) * 1000  # Convert to J/kg

    def calculate_quality(H, q_est, L_h, G, P_exp, D_exp, H_in):
        H_f_P = H_f(P_exp)
        H_fg_P = H_fg(P_exp)
        X = (4 * q_est * L_h / (G * H_fg_P * D_exp)) - (H_in / H_fg_P)
        return X

    # Iterative CHF calculation with LUT interpolation
    def iterative_chf(H, D_exp, P_exp, G_exp, H_in, L_h):
        ii=0
        q_est = CHF_initial
        converged = False
        relaxation_factor = 0.1

        # Reshape LUT data
        lenG = len(G)
        lenx = len(x)
        lenP = len(P)
        q = np.zeros((lenG, lenx, lenP))
        for i in range(lenG):
            for j in range(lenx):
                for k in range(lenP):
                    q[i, j, k] = q_raw[i + k * lenG, j]

        # Create an interpolator for the LUT
        interpolator = RegularGridInterpolator((G, x, P), q)

        while not converged:
            # Step 2: Calculate quality X
            X = calculate_quality(H, q_est, L_h, G_exp, P_exp, D_exp, H_in)
            
            # Step 3: Predict CHF using LUT and apply diameter correction
            q_pred = interpolator((G_exp, X, P_exp))

            print(f'q_pred : {q_pred} x= {X}')
            q_pred = q_pred * (D_exp / D_ref) ** -0.5
            
            # Step 4: Update q_est as the average of q_pred and previous q_est
            q_new = (1 - relaxation_factor) * q_est + relaxation_factor * q_pred
            print(ii,q_new,q_est, q_pred,'/n')
            ii+= 1
            # Step 5: Check for convergence
            if abs(q_new - q_est) < tolerance:
                converged = True
            q_est = q_new

        return q_est

    chf_lut_value = iterative_chf(H, D_exp, P_exp, G_exp, H_in, L_h)
    X2 = calculate_quality(H, chf_lut_value, L_h, G_exp, P_exp, D_exp, H_in)
    print(f"Calculated LUT CHF: {chf_lut_value/1000} kw/m²")
    return chf_lut_value/1000,X2


if __name__ == "__main__":
    getLUTCHF()
