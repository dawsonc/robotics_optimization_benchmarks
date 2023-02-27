"""Define convenient types."""
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import PyTree


DecisionVariable = PyTree[Float[Array, "..."]]
PRNGKeyArray = Integer[Array, "2"]
