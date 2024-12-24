from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # vals_plus = list(vals)
    # vals_minus = list(vals)

    # Adjust the i-th argument by epsilon
    # vals_plus[arg] += epsilon
    # vals_minus[arg] -= epsilon

    # Compute the central difference approximation
    # return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)

    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    def is_leaf(self) -> bool: ...  # noqa: D102

    def is_constant(self) -> bool: ...  # noqa: D102

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa: D102

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    # visited = set()
    # topological_order = []

    # def dfs(v: Variable) -> None:
    #    if v.unique_id in visited or v.is_constant():
    #        return
    #    visited.add(v.unique_id)
    #    for parent in v.parents:
    #        dfs(parent)
    #    topological_order.append(v)

    # dfs(variable)
    # return reversed(topological_order)

    order: List[Variable] = []
    seen = set()

    def visit(v: Variable) -> None:
        if v.unique_id in seen or v.is_constant():
            return
        if not v.is_leaf():
            for parent in v.parents:
                if not parent.is_constant():
                    visit(parent)
        seen.add(v.unique_id)
        order.append(v)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaf nodes.

        variable (Variable): The right-most variable in the computation graph.
        deriv (Any): The derivative of the variable with respect to some scalar value.

    Returns
    -------
        None: This function does not return any value. It updates the derivative values of each leaf node through `accumulate_derivative`.

    """
    # Get the topological order of the variables
    # sorted_variables = topological_sort(variable)

    # Initialize a dictionary to store the derivative of each variable
    # gradients = {variable.unique_id: deriv}

    # Traverse the variables in reverse topological order (backpropagation)
    # for v in sorted_variables:
    # skip constant variables
    #    if v.is_constant():
    #        continue

    # Compute the derivative of the current variable
    #    current_gradients = gradients[v.unique_id]

    # If it's a leaf node, accumulate the derivative
    #    if v.is_leaf():
    #        v.accumulate_derivative(current_gradients)
    #    else:
    # Otherwise, backpropagate the derivative to the parents
    #        for parent, parent_deriv in v.chain_rule(current_gradients):
    #            if parent.unique_id not in gradients:
    #                gradients[parent.unique_id] = 0
    #            gradients[parent.unique_id] += parent_deriv

    queue = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in queue:
        derivative = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(derivative)
        else:
            for parent, parent_deriv in var.chain_rule(derivative):
                if not parent.is_constant():
                    derivatives[parent.unique_id] = (
                        derivatives.get(parent.unique_id, 0) + parent_deriv
                    )


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:  # noqa: D102
        return self.saved_values
