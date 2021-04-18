# Rmpflow basic classes

import numpy as np

class RMPNode:
    """
    A generic rmp node
    """
    def __init__(self, name, parent, psi, J, J_dot, verbose=False):
        self.name = name
        self.parent = parent
        self.children = []

        # connecting this node as its parent's child
        if self.parent is not None:
            self.parent.add_child(self)

        # mapping J and J_dot for the edge from parent to this node
        self.psi = psi
        self.J = J
        self.J_dot = J_dot

        # state
        self.x = None
        self.x_dot = None

        # RMP
        self.f = None  # desired force map
        self.a = None  # desired acceleration policy
        self.M = None  # riemannian metric

        # print information
        self.verbose = verbose

    def add_child(self, child):
        """
        Add a child to the current node
        @param child    The child to be specified as one of the children of this node
        """
        self.children.append(child)

    def pushforward(self):
        """ "
        Apply a pushforward operation recursively where (x,dx) -> (y_i, dy_i) = (psi(x), J(x)*dx)
        """
        if self.verbose:
            print("{}: pushforward".format(self.name))

        if self.psi is not None and self.J is not None:
            self.x = self.psi(self.parent.x)
            self.x_dot = np.dot(self.J(self.parent.x), self.x_dot)

        # recursion
        [child.pushforward() for child in self.children]

    def pullback(self):
        """
        Apply a pullback operation recursively
        """
        # recursion
        [child.pullback() for child in self.children]

        if self.verbose:
            print("{}: pullback".format(self.name))

        # aggregate results
        f = np.zeros_like(self.x, dtype="float64")
        M = np.zeros((self.x.shape[0], self.x.shape[0]), dtype="float64")
        for child in self.children:
            J_i = child.J(self.x)
            J_dot_i = child.J_dot(self.x, self.x_dot)

            f_i = child.f
            M_i = child.M

            if child.f is not None and child.M is not None:
                f += np.dot(
                    J_i.T, (f_i - np.dot(np.dot(M_i, J_dot_i), self.x_dot))
                )  # 1-form
                M += np.dot(np.dot(J_i.T, M_i), J_i)  # 2-form

        self.f = f
        self.M = M


class RMPRoot(RMPNode):
    """
    The root node that lives in C-space
    """

    def __init__(self, name):
        # print("RMPRoot")
        super().__init__(name, None, None, None, None)

    def set_root_state(self, x, x_dot):
        """
        Set the state of the root node for pushforward
        """
        self.x = x
        self.x_dot = x_dot

    # def pushforward(self):
    #     """
    #     Apply pushforward recursively
    #     """
    #     if self.verbose:
    #         print("{}: pushforward".format(self.name))

    #     # recursion
    #     [child.pushforward() for child in self.children]

    def resolve(self):
        """
        Maps natural form [f,M]^M to canonical form (a,M)^M with a = M^+ f
        """
        if self.verbose:
            print("{}: resolve".format(self.name))

        self.a = np.dot(np.linalg.pinv(self.M), self.f)
        return self.a

    def solve(self, x, x_dot):
        """
        Given state of the root, solve for the controls
        """
        self.set_root_state(x, x_dot)
        self.pushforward()
        self.pushback()
        return self.resolve()


class RMPLeaf(RMPNode):
    """
    A leaf node
    """

    def __init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func):
        super().__init__(name, parent, psi, J, J_dot)
        self.RMP_func = RMP_func
        self.parent_param = parent_param

    def eval_leaf(self):
        """
        Compute natural form RMP given the state
        """
        self.f, self.M = self.RMP_func(self.x, self.x_dot)

    def pullback(self):
        """
        Pullback at leaf node is just evaluating the RMP
        """
        if self.verbose:
            print("{}: pullback".format(self.name))

        self.eval_leaf()

    def add_child(self):
        print("A leaf node should not be able to add a child node")
        pass

    def update_params(self):
        """
        Updating any relevant parameters that this leaf node needs
        """
        pass

    def update(self):
        """
        Updates to self
        """
        self.update_params()
        self.pushforward()
