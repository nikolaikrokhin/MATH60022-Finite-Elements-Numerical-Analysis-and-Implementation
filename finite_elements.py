# Cause division to always mean floating point division.
from __future__ import division
from re import T
import numpy as np
import math
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    if len(cell.vertices) == 2:
        z1,z2 = cell.vertices[0], cell.vertices[1]
        return np.linspace(z1,z2,degree+1)
    if len(cell.vertices) == 3:
        z1,z2,z3 = cell.vertices[0], cell.vertices[1], cell.vertices[2]
        points = []
        for i in range(degree+1):
            for j in range(degree - i + 1):
                points.append(z1+(z2-z1)*j/degree +(z3-z1)*i/degree)
        return np.array(points)
    

    


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    points = np.asarray(points)
    if len(points[0]) == 1:
        if degree == 0:
            if grad:
                return np.zeros((len(points),1,1))
            return np.ones((len(points),1))
        if not grad:
            V = np.ones((len(points),1))
        if grad:
            V =np.zeros((len(points),1,1))
        for i in range(degree):
            if not grad:
                V = np.concatenate((V,points**(i+1)),axis=1)
            if grad:
                gV = ((i+1)*points**i)
                dimgV = np.array([np.array([i for i in gV])])
                new = np.transpose(dimgV,(1,0,2))
                V = np.concatenate((V,new),axis=1)
                
        return V

    if len(points[0]) == 2:
        if degree == 0:
            if not grad:
                return np.ones((len(points),1))
            if grad:
                return np.zeros((len(points),1,2))
        V = vandermonde_matrix(cell, degree - 1, points, grad=grad)
        x = points[:, :1]
        y = points[:, 1:]
        for i in range(degree+1):
            if not grad:
                V = np.concatenate((V,x**(degree-i)*y**i),axis=1)
            if grad:
                gV = [(degree-i)*x**float(degree-i-1)*y**i,x**float(degree-i)*i*y**float(i-1)]
                dimgV = np.array([np.array([i for i in gV[0]]),np.array([i for i in gV[1]])])
                new = np.transpose(dimgV,(1,2,0))
                V = np.concatenate((V,new),axis=1)
        return np.nan_to_num(V,nan=0.0)




class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        if not grad:
            return vandermonde_matrix(self.cell,self.degree,points) @ self.basis_coefs
        if grad:
            return np.einsum("ijk,jl->ilk",vandermonde_matrix(self.cell,self.degree,points,grad=True),self.basis_coefs)

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        return fn(self.nodes)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell,degree)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        if len(nodes[0]) == 1:
            entity_nodes = {0:{0:[0],1:[degree]},1:{0:list(range(1,degree))}}
        if len(nodes[0]) == 2:
            if degree == 1:
                edge0 = []
                edge1 = []
                vertices = {0:[0],1:[1],2:[2]}
                face = {0:[]}
                edge2 = list(range(1,degree))
                edges = {0: edge0, 1: edge1, 2: edge2}
            else:
                edge0 = [2*degree]
                edge1 = [degree+1]
                for i in range(degree-2):
                    edge1.append(edge0[-1]+1)
                    edge0.append(edge0[-1]+degree-1-i)
                edge2 = list(range(1,degree))
                edges = {0: edge0, 1: edge1, 2: edge2}
                vertices = {0:[0],1:[degree],2:[edge0[-1]+1]}
                facevals = set(range(vertices[2][0]+1)).difference(*[edge0,edge1,edge2,[0,degree,edge0[-1]+1]])
                face = {0:list(facevals)}
            entity_nodes = {0:vertices,1:edges,2:face}
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)
