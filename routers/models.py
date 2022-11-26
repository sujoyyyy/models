import uuid
from fastapi import APIRouter, status, HTTPException
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys

router = APIRouter()




def deriv_sirm(y, t, N, alpha, beta, gamma, delta, b, c1, c2, c3, c4, epsilon):
    S, I, R, M = y
    dSdt = -(alpha * S * I) - (delta * b * S) + ((c1 - c2) * epsilon)
    dIdt = -(beta * I)  + (alpha * S * I) - (gamma * b * I) - (c3 * epsilon)
    dRdt =  (beta * I) - (c4 * epsilon)
    dMdt =  (delta * b * S) + (gamma * b * I)
    return dSdt, dIdt, dRdt, dMdt


def deriv_seirmz(y, t, N, alpha, beta, gamma, delta, a, c, d, f, g, b, c1, c2, c3, c4, c5, c6, epsilon):
    S, E, I, R, M, Z = y
    dSdt = -(alpha * S * E) - (alpha * I * S) + ((c1 - c2) * epsilon)
    dEdt = (alpha * S * E) + (alpha * I * S) - (a * E) - (b * delta * E) - (c * E) - (c3 * epsilon)
    dIdt = (a * E - beta * I) - (b * gamma * I) - (d * I) + (f * Z)  - (c4 * epsilon)
    dRdt = (beta * I) + (g * Z) - (c5 * epsilon)
    dMdt = (delta * b * E) + (gamma * b * I)
    dZdt = (c * E) + (d * I) - (f * Z) - (g * Z) - (c6 * epsilon)
    return dSdt, dEdt, dIdt, dRdt, dMdt, dZdt


# Api call to add new expense in a group
@router.post("/model")
def model(model_type: str, N: int, epsilon: int, b: float):
    resp = {}
    try:
        if model_type == "SIRM":
            ############################################ Empirical Variables
            # Initial number of infected, recovered and mortal individuals, I0, R0 and M0.
            #Equilibrium (Local)- (1,0,0)- only one individual is infected and share fake news
            I0, R0, M0 = 1, 0, 0
            # Everyone else, S0, is susceptible to infection initially.
            S0 = N - I0 - R0 - M0
            # Contact rate- alpha, mean recovery rate- beta, mortality rates- delta and gamma (in 1/days).
            alpha, beta, gamma, delta = 0.5, 0.05, 0.025, 0.025 
            #bias and population change rates(c1- growth and c2, c3, c4- exit rates)

            c1, c2, c3, c4 = 1,0, 0, 0.9
            # A grid of time points (in days)
            t = np.linspace(0, 160, 160)
            # Initial conditions vector
            y0 = S0, I0, R0, M0
            # Integrate the SIR equations over the time grid, t.
            ret = odeint(deriv_sirm, y0, t, args=(N, alpha, beta, gamma, delta, b, c1, c2, c3, c4, epsilon))
            S, I, R, M = ret.T

            # Plot the data on four separate curves for S(t), I(t), R(t) and M(t)
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
            ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
            ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
            ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
            ax.plot(t, M, 'y', alpha=0.5, lw=2, label='Mortality')
            ax.set_xlabel('Time /days')
            ax.set_ylabel('Population')
            ax.set_ylim(0,N)
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.grid(b=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            #plt.show()
            plt.savefig("static/SIRMoutput.jpg")
            resp["url"] = "/static/SIRMoutput.jpg"
        else:
            E0, I0, R0, M0, Z0 = 1, 0, 0, 0, 0
            # Everyone else, S0, is susceptible to infection initially.
            S0 = N - I0 - R0 - M0 - Z0 - E0
            # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
            alpha, beta, gamma, delta = 0.5, 0.05, 0.025, 0.025 
            a,c,d,f,g = 0.5, 0.05, 0.025, 0.025 , 0.025
            #bias and population changes
            c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0.001
            # A grid of time points (in days)
            t = np.linspace(0, 160, 160)

            # The SEIRMZ model differential equations.


            # Initial conditions vector
            y0 = S0, E0, I0, R0, M0, Z0
            # Integrate the SIR equations over the time grid, t.
            ret = odeint(deriv_seirmz, y0, t, args=(N, alpha, beta, gamma, delta, a, c, d, f, g, b, c1, c2, c3, c4, c5, c6, epsilon))
            S, E, I, R, M, Z = ret.T

            # Plot the data on four separate curves for S(t), I(t), R(t) and M(t)
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
            ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
            ax.plot(t, E, 'c', alpha=0.5, lw=2, label='Exposed')
            ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
            ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
            ax.plot(t, M, 'y', alpha=0.5, lw=2, label='Mortality')
            ax.plot(t, Z, 'k', alpha=0.5, lw=2, label='Skeptic')
            ax.set_xlabel('Time /days')
            ax.set_ylabel('Population')
            ax.set_ylim(0,N)
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.grid(b=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            #plt.show()
            plt.savefig("static/SEIRMZoutput.jpg")
            resp["url"] = "/static/SEIRMZoutput.jpg"
        return resp
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("model:", e, "at", str(exc_tb.tb_lineno))
        raise HTTPException(500)