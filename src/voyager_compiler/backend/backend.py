"""What a target adds to the shared ``compile()``: a ``Backend``.

A config's ``backend()`` returns it.  ``compile()`` bufferizes with its
builders; the rest of the pipeline is the same for every target.
"""

from dataclasses import dataclass

from voyager_compiler.codegen.transform.bufferize import BuilderRule


@dataclass(frozen=True)
class Backend:
    """A target's hooks into ``compile()``.

    Attributes:
        builders: The rules that pick each compute node's builder, tried in
            order; a rule ahead of ``SHARED_BUILDERS`` replaces the shared
            builder for the nodes it matches.
        synchronizes: Whether ``compile()`` runs the sync pass, which wraps
            every dispatch in a ``commit`` and derives the unit-counter
            waits; off for a target whose own toolchain orders the work.
        store_lag: Steps a pipelined GEMM / conv nest's stores trail its
            compute (``PipelinedKernel``'s ``lag``).  At 1 a tile's store
            overlaps the next tile's compute, and a split reduction with a
            tail may rotate its partial sum over two scratch slots; at 0
            each step stores its own tile from one scratch slot.
    """

    builders: tuple[BuilderRule, ...]
    synchronizes: bool = True
    store_lag: int = 1
