import numpy as np

mu = np.array([0, 0, 0])
a = 0.3
b = 0.8
c = 0.5

alpha = 1
beta = 7
gamma = 0.1

EXPECTATIONS = [
    [0, 1, 0, 3, 0, 15, 0, 105],
    [1, 0, 3, 0, 15, 0, 105],
    [(1, 2), (0, 0), (3, 12), (0, 0), (15, 90), (0, 0), (105, 840)],
    [(9, 6), (0, 0), (45, 60)],
]


def exp_poly(p1, p2, rho=a):
    if p1 > p2:
        p1, p2 = p2, p1

    if p1 == 0:
        return EXPECTATIONS[0][p2]
    if p1 == 1:
        return EXPECTATIONS[1][p2 - p1] * rho
    if p1 == 2:
        x, y = EXPECTATIONS[2][p2 - p1]
        return x + y * rho**2
    if p1 == 3:
        x, y = EXPECTATIONS[3][p2 - p1]
        return x * rho + y * rho**3
    if p1 == 4:
        return 9 + 72 * (rho**2) + 24 * (rho**4)

    raise ValueError("Invalid input")


def analytical_mse(alpha_, beta_, gamma_):
    a_diff = alpha_ - alpha
    b_diff = beta_ - beta
    c_diff = gamma_ - gamma
    return (
        a_diff**2 +
        3*b_diff**2 +
        c_diff**2*exp_poly(2, 8, b) +
        2*a_diff*c_diff*exp_poly(2, 4, b)
    )


VAR_F = alpha**2 + 2*beta**2 + (gamma**2)*(105 + 840*(b**2)) + 2*alpha*gamma*(3 + 12*(b**2))

sig2_1 = (1 - (a**2 - 2*a*b*c + b**2) / (1 - c**2))
D1 = (alpha**2 + 6*alpha*gamma + 105*(gamma**2)) * sig2_1

sig2_2 = 1 - (a**2 - 2*a*b*c + c**2) / (1 - b**2)
d = a - b * c
e = c - a * b
f = 1 / (1 - b**2)

D2 = 4*(beta**2) * sig2_2 * (f**2) * (d**2 + e**2 + 2*d*e*b) + 2*(beta**2)*(sig2_2**2)

sig2_3 = 1 - (b**2 - 2*a*b*c + c**2) / (1 - a**2)
f = 1 / (1 - a**2)
d = b - a * c
e = c - a * b
Ex2mu6 = (f**6) * (105*(d**6) + 6*(d**5)*e*exp_poly(7, 1) + 15*(d**4)*(e**2)*exp_poly(6, 2) + 20*(d**3)*(e**3)*exp_poly(5, 3) + 15*(d**2)*(e**4)*exp_poly(4, 4) + 6*d*(e**5)*exp_poly(3, 5) + (e**6)*exp_poly(2, 6))
Ex2mu4 = (f**4) * (15*(d**4) + 4*(d**3)*e*exp_poly(5, 1) + 6*(d**2)*(e**2)*exp_poly(4, 2) + 4*d*(e**3)*exp_poly(3, 3) + (e**4)*exp_poly(2, 4))
Ex2mu2 = (f**2) * (3*(d**2) + 6*a*d*e + (e**2)*exp_poly(2, 2))
D3 = (gamma**2) * (16*Ex2mu6*sig2_3 + 168*Ex2mu4*(sig2_3**2) + 384*Ex2mu2*(sig2_3**3) + 96*(sig2_3**4))

S1 = D1/VAR_F
S2 = D2/VAR_F
S3 = D3/VAR_F


if __name__ == "__main__":
    print(S1, S2, S3)
