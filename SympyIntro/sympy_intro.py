# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name> Sam Carpenter
<Class>
<Date> 1/19/21
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x, y = sy.symbols('x, y')
    return sy.Rational(2,5) * sy.exp(x ** 2 - y) * sy.cosh(x + y) + sy.Rational(3, 7) * sy.log(x * y + 1)

#print(prob1())

# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    i, j, x = sy.symbols('i, j, x')
    expression = sy.product(sy.summation(j*(sy.sin(x) + sy.cos(x)), (j, i, 5)), (i, 1, 5))
    return sy.simplify(expression)

#print(prob2())

# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    x, n, y = sy.symbols('x, n, y')
    expression = sy.summation((x**n)/sy.factorial(n), (n, 0, N))
    #print(expression)

    subExp = expression.subs(x, -(y**2))
    #print(subExp)
    lambExp = sy.lambdify(y, subExp)

    domain = np.linspace(-2, 2, 200)
    plt.plot(domain, lambExp(domain))
    plt.plot(domain, np.exp(-(domain**2)))
    plt.legend(('Maclaurin Series', 'e^(-y^2)'))
    plt.show()

#prob3(10)

# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x, y, r, theta = sy.symbols('x, y, r, theta')
    expr = 1 - ((x**2 + y**2)**(sy.Rational(7, 2)) + 18*(x**5)*y - 60*(x**3)*(y**3) + 18*x*(y**5)) / (x**2 + y**2)**3
    #print(expr)
    subExpr = expr.subs({x: r * sy.cos(theta), y: r * sy.sin(theta)})
    polarExpr = sy.simplify(subExpr)

    #print(polarExpr)
    rSolution = sy.simplify(sy.solve(polarExpr, r)[0])

    #print(rSolution)
    #rExpr = sy.simplify(polarExpr.subs(r, rSolution))
    rLamb = sy.lambdify(theta, rSolution, "numpy")

    #print(rExpr)
    domain = np.linspace(0, 2 * np.pi, 200)

    plt.plot(rLamb(domain) * np.cos(domain), rLamb(domain) * np.sin(domain))
    plt.show()

#prob4()

# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x, y, lamb = sy.symbols('x, y, lamb')
    A = sy.Matrix([[x-y,   x,   0], [x, x-y,   x], [0,   x, x-y]])
    iLamb = lamb * sy.eye(3)
    solveA = (A - iLamb)

    #Compute eigenvalues
    expr = sy.det(solveA)
    eVals = sy.solve(expr, lamb)
    #Compute eigenvectors
    eVects = [solveA.subs(lamb, eVal).nullspace()[0] for eVal in eVals]

    valVectDict = {eVals[i]: eVects[i] for i in range(len(eVals))}
    #print(valVectDict)
    return valVectDict

def prob5test():
    x, y, lamb = sy.symbols('x, y, lamb')
    A = sy.Matrix([[x - y, x, 0], [x, x - y, x], [0, x, x - y]])
    eigenVects = A.eigenvects()
    print(eigenVects)

#prob5()
#print()
#print()
#prob5test()

# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    x = sy.symbols('x')
    p = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    pLamb = sy.lambdify(x, p, "numpy")

    #Take derivatives
    dx = sy.diff(p, x)
    dxdx = sy.diff(dx, x)
    dxdxLamb = sy.lambdify(x, dxdx)

    critPts = sy.solve(dx, x)
    # True if max (Red), False if min (Blue)
    isMax = [dxdxLamb(critPt) < 0 for critPt in critPts]

    mins = set()
    maxs = set()

    domain = np.linspace(-5, 5, 200)
    plt.plot(domain, pLamb(domain))
    for i, critPt in enumerate(critPts):
        if isMax[i]: # Local max
            plt.plot(critPt, pLamb(critPt), 'r.')
            maxs.add(critPt)
        else: # Local min
            plt.plot(critPt, pLamb(critPt), 'b.')
            mins.add(critPt)
    plt.legend(('p(x)', 'Blue = Min', 'Red = Max' ))
    plt.show()
    return mins, maxs
#print(prob6())
#({1, 3, -4}, {2, -2})


# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    x, y, z, r, theta, phi, rho = sy.symbols('x, y, z, r, theta, phi, rho')
    f = (x**2 + y**2 + z**2)**2

    #Jacobian
    J = sy.Matrix([[rho*sy.sin(phi)*sy.cos(theta)],
                   [rho*sy.sin(phi)*sy.sin(theta)],
                   [rho*sy.cos(phi)]]).jacobian([rho, theta, phi])

    #Inner Expression
    expr = f.subs({x: rho*sy.sin(phi)*sy.cos(theta),
                   y: rho*sy.sin(phi)*sy.sin(theta),
                   z: rho*sy.cos(phi)}) * -1 * sy.det(J)

    #Compute integral
    int1 = sy.integrate(sy.simplify(expr), (rho, 0, r))
    int2 = sy.integrate(sy.simplify(int1), (theta, 0, 2 * sy.pi))
    int3 = sy.integrate(sy.simplify(int2), (phi, 0, sy.pi))
    final = sy.simplify(int3)

    sphereIntegral = sy.lambdify(r, final, "numpy")

    rDomain = np.linspace(0, 3, 200)
    plt.plot(rDomain, sphereIntegral(rDomain))
    plt.show()

    return sphereIntegral(2)

#print(prob7())
