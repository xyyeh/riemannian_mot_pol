from rmpflow.rmp.rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as Rot

# from jax import grad, jit, vmap, jacfwd
# import jax.numpy as jnp

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
        N = y_g.size
        psi = lambda y: (y - y_g)
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))

        def RMP_func(x, x_dot):
            # x and x_dot refers to the error and error_dot
            x_norm = norm(x)

            # -ve gradient of eta-scaled softmax potential function (25)
            s = (1 - np.exp(-2 * alpha * x_norm)) / (1 + np.exp(-2 * alpha * x_norm))

            # functions in Appendix D's "Metric options"
            gamma = np.exp(-(x_norm ** 2) / 2 / (sigma_gamma ** 2))
            w = (w_u - w_l) * gamma + w_l

            # metric options (27)
            G = np.eye(N) * w
            if x_norm > tol:
                grad_Phi = s / x_norm * w * x * gain
            else:
                grad_Phi = 0
            Bx_dot = eta * w * x_dot
            grad_w = -gamma * (w_u - w_l) / sigma_gamma ** 2 * x

            # since gradient is simple, xi is hand-computed
            # M_stretch is a bit more complicated, we'll need to use a differentiation library like AutoGrad
            x_dot_norm = norm(x_dot)
            xi = -0.5 * (
                x_dot_norm ** 2 * grad_w
                - 2 * np.dot(np.dot(x_dot, x_dot.transpose()), grad_w)
            )

            # Use G(x, dx) = M(x), since there is no dependence on velocity, so \Xi = 0
            M = G
            f = - grad_Phi - Bx_dot - xi

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
    Obstacle avoidance RMP leaf using sphere primitive
    """

    def __init__(
        self,
        name,
        parent,
        parent_param,
        c=np.zeros(3),
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
            psi = lambda y: np.array(norm(y - c) - R).reshape(-1, 1)
            J = lambda y: 1.0 / norm(y - c) * (y - c).transpose()
            J_dot = lambda y, y_dot: np.dot(
                y_dot.transpose(),
                (
                    -1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).transpose())
                    + 1 / norm(y - c) * np.eye(N)
                ),
            )

            def RMP_func(x, x_dot):
                # using D.3, x is d(x)
                s = x
                w = max(r_w - s, 0) / (s - R) if (s - R) >= 0 else 1e10
                grad_w = (
                    (((r_w - s) > 0) * -1 * (s - R) - max(r_w - s, 0.0)) / (s - R) ** 2
                    if (s - R) >= 0
                    else 0
                )

                # epsilon is the constant value when moving away from the obstacle
                u = epsilon + (
                    1.0 - np.exp(-(x_dot ** 2) / 2.0 / sigma ** 2) if x_dot < 0 else 0.0
                )
                g = w * u

                grad_u = (
                    np.exp(-(x_dot ** 2) / 2.0 / sigma ** 2) * x_dot / sigma ** 2
                    if x_dot < 0
                    else 0.0
                )

                grad_Phi = alpha * w * grad_w
                xi = 0.5 * x_dot ** 2 * u * grad_w

                # upper-case xi calculation is included here
                M = g + 0.5 * x_dot * w * grad_u
                M = np.minimum(np.maximum(M, -1e5), 1e5)

                Bx_dot = eta * g * x_dot

                f = np.minimum(np.maximum(-grad_Phi - xi - Bx_dot, -1e10), 1e10)

                return (f, M)

            super().__init__(name, parent, parent_param, psi, J, J_dot, RMP_func)


class CollisionAvoidanceBox(RMPLeaf):
    """
    Obstacle avoidance RMP leaf using box primitive
    """

    def __init__(
        self,
        name,
        parent,
        parent_param,
        c,
        r,
        R,
        xyz=np.zeros((3, 1)),
        epsilon=0.2,
        alpha=1e-5,
        eta=0,
        r_w=0.07,
        sigma=0.5,
    ):
        r = np.abs(r)
        rot_w_box = Rot.from_euler("zyx", zyx.flatten())

        # rotation transformation from global frame to box
        rot_box_w = inv(rot_w_box.as_matrix())

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

            # graphics people solved this one already:
            # https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
            # https://www.youtube.com/watch?v=62-pRVZuS5c
            # note: we normalize by R

            def psi(y):
                q = np.abs(np.dot(rot_box_w, y - c)) - r
                return np.array(norm(np.maximum(q, 0.0))).reshape(-1, 1)

            # leverage chain rule for Jacobian
            def J(y):
                p = np.dot(rot_inv, y - c)
                q = np.abs(p) - r
                sdf = norm(np.maximum(q, 0.0))

                if sdf <= 0.0:
                    print(self.name + " resulted in zero sdf")

                return np.dot(
                    np.repeat(1 / sdf, 3).reshape(1, 3),
                    np.dot(
                        np.diag(np.maximum(q, 0.0).flatten()),
                        np.dot(np.diag(np.sign(p).flatten()), rot_inv),
                    ),
                )

            # ... and J dot (this was done by multiplying out Jacobian and using quotient rule)
            def J_dot(y, y_dot):
                p = np.dot(rot_inv, y - c)
                q = np.abs(p) - r
                sdf = norm(np.maximum(q, 0.0))

                p_dot = np.dot(rot_inv, y_dot)
                q_dot = np.sign(p) * p_dot
                max_dot = (q > 0) * q_dot

                sdf_dot = np.sum(np.maximum(q, 0.0) * max_dot) / sdf
                return np.dot(
                    (
                        np.sign(p)
                        * (max_dot * sdf - sdf_dot * np.maximum(q, 0.0))
                        / sdf ** 2
                    ).transpose(),
                    rot_inv,
                )

        def RMP_func(x, x_dot):
            w = max(r_w - x, 0) / (x - R) if (x - R) >= 0 else 1e10
            grad_w = (
                (((r_w - x) > 0) * -1 * (x - R) - max(r_w - x, 0.0)) / (x - R) ** 2
                if (x - R) >= 0
                else 0
            )

            # epsilon is the constant value when moving away from the obstacle
            u = epsilon + (
                1.0 - np.exp(-(x_dot ** 2) / 2.0 / sigma ** 2) if x_dot < 0 else 0.0
            )
            g = w * u

            grad_u = (
                np.exp(-(x_dot ** 2) / 2.0 / sigma ** 2) * x_dot / sigma ** 2
                if x_dot < 0
                else 0.0
            )

            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            # upper-case xi calculation is included here
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, -1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = -grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, -1e10), 1e10)

            return (f, M)

        super().__init__(name, parent, parent_param, psi, J, J_dot, RMP_func)
