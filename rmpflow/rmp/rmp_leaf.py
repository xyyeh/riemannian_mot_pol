from rmpflow.rmp.rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as Rot
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd


class GoalAttractorUni(RMPLeaf):
    """
    Goal attractor RMP leaf
    """

    def __init__(
        self,
        name,
        parent,
        y_g,
        w_u=10,
        w_l=1,
        sigma_gamma=1,
        alpha=1,
        eta=2,
        gain=1,
        tol=0.005,
    ):
        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)

        N = y_g.size
        psi = lambda y: (y - y_g)
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))

        def RMP_func(x, x_dot):
            x_norm = norm(x)

            # -ve gradient of eta-scaled softmax potential function (25)
            s = (1 - np.exp(-2 * alpha * x_norm)) / (1 + np.exp(-2 * alpha * x_norm))

            # functions in Appendix D's "Metric options"
            gamma = np.exp(-(x_norm ** 2) / 2 / (sigma_gamma ** 2))
            w = (w_u - w_l) * sigma_gamma + w_l

            # metric options (27)
            G = np.eye(N) * w
            if x_norm > tol:
                grad_Phi = s / x_norm * w * x * gain
            else:
                grad_Phi = 0
            Bx_dot = eta * w * x_dot
            grad_w = -beta * (w_u - w_l) / sigma ** 2 * x

            # since gradient is simple, xi is hand-computed, M_stretch is a bit more complicated, we'll be using a differentiation library like AutoGrad
            x_dot_norm = norm(x_dot)
            xi = -0.5 * (
                x_dot_norm ** 2 * grad_w - 2 * np.dot(np.dot(x_dot, x_dot.T), grad_w)
            )

            # Use G(x, dx) = M(x), since there is no dependence on velocity, so \Xi = 0
            M = G
            f = -grad_Phi - Bx_dot - xi

            return (f, M)  # Natural form

        super().__init__(name, parent, None, psi, J, J_dot, RMP_func)

    def update_goal(self, y_g):
        """
        Updates the position of the goal
        """
        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)

        N = y_g.size
        self.psi = lambda y: (y - y_g)
        self.J = lambda y: np.eye(N)
        self.J_dot = lambda y, y_dot: np.zeros((N, N))


class Damper(RMPLeaf):
    """
    Damper
    """

    def __init__(self, name, parent, w=1, eta=1):
        psi = lambda y: y
        J = lambda y: np.eye(y.size)
        J_dot = lambda y, y_dot: np.zeros((y.size, y.size))

        def RMP_func(x, x_dot):
            G = w
            Bx_dot = eta * w * x_dot
            M = G
            f = -Bx_dot

            return (f, M)

        super().__init__(name, parent, None, psi, J, J_dot, RMP_func)


class CollisionAvoidance(RMPLeaf):
    """
    Collision avoidance
    """

    def __init__(
        self,
        name,
        parent,
        parent_param,
        c=np.zeros(2),
        R=1,
        epsilon=0.2,
        alpha=1e-5,
        eta=0,
    ):
        self.R = R
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        if parent_param:
            psi = None
            J = None
            J_dot = None
        else:
            if c.ndim == 1:
                c = c.reshape(-1, 1)

            N = c.size

            # R is the radius of the obstacle point
            psi = lambda y: np.array(norm(y-c)-R).reshape(-1,1)
            J = lambda y: 1.0 / norm(y - c) * (y - c).transpose()
            J_dot = lambda y, y_dot: np.dot(
                y_dot.T,
                (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).transpose())
                 + 1 / norm(y - c) * np.eye(N)))

            def RMP_func(x, x_dot):
                # using D.3, x is d(x)
                s = x
                w = 