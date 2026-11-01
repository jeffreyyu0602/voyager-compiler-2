#!/usr/bin/env python
"""Rewrites a Voyager ``model.txt`` into a form two compiles can be diffed by.

``model.txt`` is the text format of ``codegen/voyager_ir.proto``. The
compiler names each value after an FX node, and FX numbers its nodes with
global counters (``alloc_143``, ``cond_67``), so one added node renames every
later one. This tool renames those values so that a kernel that did not
change reads the same however much changed around it.

- Only strings that name values are renamed: the ``name`` of an Operation,
  PrimOp, Output, LoopIterArg or Tiling, the ``node`` of a TensorBox or
  ScalarValue, and a ForLoop's ``iv``. Op targets, kwargs keys, dtypes and
  ``str_value`` strings stay as written. Model inputs and parameters keep
  their names; every other value, model outputs included, is renamed.
- The top-level ops are cut into kernels. A kernel starts at a top-level
  ``voyager::alloc`` named ``<layer>_alloc``, ``<layer>_alloc_<N>``,
  ``<layer>_alloc_default`` or ``<layer>_alloc_default_<N>`` whose layer
  differs from the current kernel's; the ops before the first one form the
  kernel ``(start)``. A kernel is named by its layer, with ``#<k>`` added
  when the layer starts a kernel again later.
- A name a kernel defines (a ``name`` or ``iv`` anywhere in its ops) is that
  kernel's own: loop bodies are separate FX graphs, so two kernels may
  define the same name. Names and layers are read with every ``_<digits>``
  segment removed (``conv2d_3_scale`` -> ``conv2d_scale``). A name that then
  starts with the kernel's layer becomes ``<kernel>_<base>_<i>``, any other
  ``<kernel>/<base>_<i>``, where ``<base>`` is that counter-free name less
  the layer and ``<i>`` numbers the kernel's distinct names of that base by
  first appearance: ``layer1_0_conv1_alloc_143`` becomes
  ``layer1_0_conv1_alloc_0``, the scratch ``alloc_150`` in it
  ``layer1_0_conv1/alloc_0``, and an ``add_9`` in kernel ``add``
  ``add/add_0``. An op's output named ``<op>_<i>`` becomes the op's new
  name plus ``_<i>``. A name a kernel uses but does not define is the value
  a top-level op of another kernel defines, and takes that kernel's name for
  it; a name nothing defines belongs to each kernel that uses it. No two
  values get one name.
- Addresses are left out of the text, since one added buffer shifts every
  address after it. They are compared apart, per storage declaration (a
  TensorBox outside a reference), with each memory level's peak bytes.

Compared with a baseline (``canonicalize(text, reference)``), kernels are
matched rather than numbered by position. The two kernel lists are aligned
on a signature: the kernel's text with each other kernel's name reduced to
its layer's counter-free family (``mul_5`` -> ``mul``). Each kernel left
over then pairs with a left-over baseline kernel of the same family, the
most alike first (the share of signature lines they have in common), ties
going to the same layer, then the nearer position. A matched kernel takes
its baseline kernel's name; an unmatched one keeps its own, marked ``+``
while the baseline uses that name. A name nothing defines takes the
baseline's name for the same string, if the baseline defines it.

Limits: a changed kernel pairs by likeness within its family, so when a
change both adds and edits kernels of one family it can pick a different
partner than a person would; the ops between two kernels belong to the
earlier one; and a kernel reading a value no kernel defines any more shows
as changed when that value's name also changed.

Usage::

    python tools/canon_model.py MODEL.txt                # canonical text
    python tools/canon_model.py OLD.txt NEW.txt [--diff] # kernel summary
"""

import argparse
import difflib
import hashlib
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace

# Fields whose string value names a value.
NAME_KEYS = ("name", "node", "iv")
# Of those, the fields that define the value they name.
DEFINING_KEYS = ("name", "iv")
KERNEL_HEAD = re.compile(r"^(?P<layer>.+)_alloc(?:_default)?(?:_\d+)?$")
COUNTER = re.compile(r"_\d+$")
# Every ``_<digits>`` segment of a name, as FX counters are spelled.
COUNTERS = re.compile(r"_\d+(?=_|$)")
START = "(start)"
# Bits per element of the dtypes whose name carries no width.
DTYPE_BITS = {"bool": 8}
LEVEL_PREFIX = "MEMORY_LEVEL_"


def parse(text):
    """Parses protobuf text format into nested field lists.

    Args:
        text: The text, one field or closing brace per line, as
            ``text_format`` writes it.

    Returns:
        The top-level message: a list of ``(key, value)`` pairs, where
        ``value`` is the raw scalar text or, for a message, its own list.
    """
    root = []
    stack = [root]
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line == "}":
            stack.pop()
        elif line.endswith(" {"):
            child = []
            stack[-1].append((line[:-2], child))
            stack.append(child)
        else:
            key, sep, value = line.partition(": ")
            if not sep:
                raise ValueError(f"line {number}: cannot parse {line!r}")
            stack[-1].append((key, value))
    if len(stack) != 1:
        raise ValueError("unbalanced braces")
    return root


def _string(value):
    """The string a quoted scalar holds, or ``None`` for any other value."""
    if (
        isinstance(value, str)
        and len(value) > 1
        and value[0] == value[-1] == '"'
    ):
        return value[1:-1]
    return None


def _field(message, key, default=None):
    """The first value of ``key`` in ``message``."""
    return next((v for k, v in message if k == key), default)


def _names(message, found):
    """Appends ``(key, name)`` for each name field under ``message``."""
    for key, value in message:
        if isinstance(value, list):
            _names(value, found)
        elif key in NAME_KEYS and _string(value) is not None:
            found.append((key, _string(value)))
    return found


def _derived(message, found):
    """Finds the outputs named ``<op>_<i>``: ``{output: (op, i)}``."""
    op = _string(_field(message, "name"))
    for key, value in message:
        if not isinstance(value, list):
            continue
        output = _string(_field(value, "name"))
        if key == "outputs" and op and output and output != op:
            index = output.removeprefix(op + "_")
            if index != output and index.isdigit():
                found[output] = (op, index)
        _derived(value, found)
    return found


def render(message, rename, depth=0, lines=None):
    """Renders a message as text format, renamed and without addresses.

    Args:
        message: A field list from ``parse``.
        rename: Maps an original value name to the name written.
        depth: Indent level of ``message``'s fields.
        lines: The list rendered into.

    Returns:
        The rendered lines.
    """
    lines = [] if lines is None else lines
    pad = "  " * depth
    for key, value in message:
        if isinstance(value, list):
            lines.append(f"{pad}{key} {{")
            render(value, rename, depth + 1, lines)
            lines.append(f"{pad}}}")
        elif key == "address":
            continue
        elif key in NAME_KEYS and _string(value) is not None:
            lines.append(f'{pad}{key}: "{rename(_string(value))}"')
        else:
            lines.append(f"{pad}{key}: {value}")
    return lines


def dtype_bits(dtype):
    """Bits per element of a dtype name such as ``fp8_e4m3`` or ``int24``."""
    if dtype in DTYPE_BITS:
        return DTYPE_BITS[dtype]
    width = re.search(r"\d+", dtype)
    if width is None:
        raise ValueError(f"unknown dtype width: {dtype!r}")
    return int(width.group())


@dataclass(frozen=True)
class Box:
    """Where one storage declaration lives and how many bytes it spans."""

    level: str
    partition: int
    address: int
    size: int


def _box(message):
    """Reads a TensorBox's placement; proto3 text leaves zeros out."""
    memory = _field(message, "memory", [])
    numel = math.prod(int(v) for k, v in message if k == "shape")
    bank = math.ceil(numel * dtype_bits(_string(_field(message, "dtype"))) / 8)
    count = max(1, int(_field(message, "bank_count", 1)))
    stride = int(_field(message, "bank_stride_bytes", bank))
    level = _field(memory, "level", LEVEL_PREFIX + "IMMEDIATE")
    return Box(
        level=level.removeprefix(LEVEL_PREFIX),
        partition=int(_field(memory, "partition", 0)),
        address=int(_field(memory, "address", 0)),
        size=stride * (count - 1) + bank,
    )


def _declarations(message, key, found):
    """Appends ``(node, Box)`` for each TensorBox outside a reference."""
    if key != "box" and _field(message, "node") and _field(message, "dtype"):
        found.append((_string(_field(message, "node")), _box(message)))
    for child_key, value in message:
        if isinstance(value, list):
            _declarations(value, child_key, found)
    return found


def family(layer):
    """A layer's name less its FX counter: ``mul_5`` -> ``mul``."""
    return COUNTER.sub("", layer)


@dataclass
class Kernel:
    """A run of top-level ops starting at one layer's output allocation.

    Attributes:
        layer: The layer its first allocation is named after.
        ops: Its top-level ops.
        name: Its name in the canonical text.
        signature: A digest of its sketch, its text with every other
            kernel's name reduced to its layer's family, for matching
            against a baseline.
        lines: Its sketch's lines, counted.
        places: Each name it owns, but those in ``derived``, and its
            ``(named after the layer, base, i)``.
        derived: Its outputs named ``<op>_<i>``: ``{output: (op, i)}``.
        rename: Each name it owns and its canonical name.
    """

    layer: str
    ops: list = field(default_factory=list)
    name: str = ""
    signature: str = ""
    lines: Counter = field(default_factory=Counter)
    places: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    rename: dict = field(default_factory=dict)

    def number(self, found, local):
        """Numbers the names this kernel owns (see the module docstring).

        Args:
            found: ``(key, name)`` of each name field in its ops, in order.
            local: The names it owns: those it defines and those it uses
                that nothing else names.
        """
        for op in self.ops:
            _derived(op, self.derived)
        self.derived = {
            out: (op, i)
            for out, (op, i) in self.derived.items()
            if out in local and op in local
        }
        layer = COUNTERS.sub("", self.layer)
        counts = {}
        for _, value in found:
            if value not in local or self.owns(value):
                continue
            stem = COUNTERS.sub("", value)
            own = stem.startswith(layer + "_")
            base = stem[len(layer) + 1 :] if own else stem
            i = counts[own, base] = counts.get((own, base), -1) + 1
            self.places[value] = (own, base, i)

    def spell(self, value, prefix):
        """Spells a name this kernel owns under the kernel name ``prefix``."""
        if value in self.derived:
            op, i = self.derived[value]
            return f"{self.spell(op, prefix)}_{i}"
        own, base, i = self.places[value]
        return f"{prefix}{'_' if own else '/'}{base}_{i}"

    def owns(self, value):
        """Whether ``value`` is one of this kernel's own names."""
        return value in self.places or value in self.derived


def _split(ops):
    """Cuts the top-level ops into named kernels."""
    kernels = [Kernel(START)]
    for op in ops:
        head = KERNEL_HEAD.match(_string(_field(op, "name", '""')))
        if (
            head
            and _field(_field(op, "prim", []), "target") == '"voyager::alloc"'
            and head["layer"] != kernels[-1].layer
        ):
            kernels.append(Kernel(head["layer"]))
        kernels[-1].ops.append(op)
    if not kernels[0].ops:
        kernels.pop(0)
    seen = {}
    for kernel in kernels:
        k = seen[kernel.layer] = seen.get(kernel.layer, -1) + 1
        kernel.name = kernel.layer if k == 0 else f"{kernel.layer}#{k}"
    return kernels


def _pair(old, new):
    """Matches kernels to the baseline's (see the module docstring).

    Args:
        old: The baseline's kernels.
        new: This version's kernels.

    Returns:
        ``{new index: old index}`` for every matched kernel.
    """
    matcher = difflib.SequenceMatcher(
        None,
        [k.signature for k in old],
        [k.signature for k in new],
        autojunk=False,
    )
    pairs = {}
    for block in matcher.get_matching_blocks():
        pairs.update(
            zip(
                range(block.b, block.b + block.size),
                range(block.a, block.a + block.size),
            )
        )
    matched = set(pairs.values())
    olds = [i for i in range(len(old)) if i not in matched]
    news = [j for j in range(len(new)) if j not in pairs]
    candidates = sorted(
        (
            -_likeness(old[i], new[j]),
            old[i].layer != new[j].layer,
            abs(i - j),
            j,
            i,
        )
        for i in olds
        for j in news
        if family(old[i].layer) == family(new[j].layer)
    )
    taken = set()
    for *_, j, i in candidates:
        if j not in pairs and i not in taken:
            pairs[j] = i
            taken.add(i)
    return pairs


def _likeness(a, b):
    """The share of two kernels' sketch lines they have in common."""
    common = sum((a.lines & b.lines).values())
    return common / max(a.lines.total(), b.lines.total(), 1)


@dataclass
class Program:
    """One canonicalized ``model.txt``.

    Attributes:
        text: The whole canonical text, addresses left out.
        header: The canonical inputs, parameters and outputs.
        kernels: Each kernel's name and canonical text, in program order.
        boxes: Each storage declaration's canonical name and placement.
        globals: Each name a top-level op defines and its canonical name.
        others: Each name kept or used undefined and the name written.
        order: The kernels, for matching a later version against this one.
    """

    text: str
    header: str
    kernels: dict
    boxes: dict
    globals: dict
    others: dict
    order: list


def canonicalize(text, reference=None):
    """Renames a ``model.txt``'s values so two versions can be compared.

    Args:
        text: The ``model.txt`` text.
        reference: The baseline's ``Program``. Given, kernels are matched
            to the baseline's and take their names.

    Returns:
        The ``Program``.
    """
    tree = parse(text)
    header = [(k, v) for k, v in tree if k != "ops"]
    kept = {
        name
        for key, value in header
        if key in ("inputs", "parameters")
        for _, name in _names(value, [])
    }
    kernels = _split([v for k, v in tree if k == "ops"])
    hints = reference.globals if reference is not None else {}
    found = [[n for op in k.ops for n in _names(op, [])] for k in kernels]
    defined = [
        {n for key, n in f if key in DEFINING_KEYS} - kept for f in found
    ]
    top = {}
    for kernel, own in zip(kernels, defined):
        for op in kernel.ops:
            for message in (op, *[v for k, v in op if k == "outputs"]):
                name = _string(_field(message, "name"))
                if name in own:
                    top.setdefault(name, kernel)
    for kernel, names, own in zip(kernels, found, defined):
        unknown = {n for _, n in names} - kept - top.keys() - hints.keys()
        kernel.number(names, own | unknown)

    for kernel in kernels:

        def sketch(value):
            if kernel.owns(value):
                return kernel.spell(value, ".")
            if value in top:
                owner = top[value]
                return owner.spell(value, family(owner.layer))
            return value

        lines = [line for op in kernel.ops for line in render(op, sketch)]
        digest = hashlib.sha1("\n".join(lines).encode())
        kernel.signature = digest.hexdigest()
        kernel.lines = Counter(lines)

    if reference is not None:
        pairs = _pair(reference.order, kernels)
        used = {k.name for k in reference.order}
        for j, kernel in enumerate(kernels):
            if j in pairs:
                kernel.name = reference.order[pairs[j]].name
            else:
                while kernel.name in used:
                    kernel.name += "+"

    taken = set(kept)

    def fresh(name):
        while name in taken:
            name += "'"
        taken.add(name)
        return name

    for kernel in kernels:
        kernel.rename = {
            value: fresh(kernel.spell(value, kernel.name))
            for value in (*kernel.places, *kernel.derived)
        }
    globals_ = {value: owner.rename[value] for value, owner in top.items()}
    others = {name: name for name in kept}

    def outside(value):
        if value in globals_:
            return globals_[value]
        if value not in others:
            others[value] = fresh(hints.get(value, value))
        return others[value]

    texts = {}
    boxes = {}
    for kernel in kernels:

        def rename(value, kernel=kernel):
            return kernel.rename.get(value) or outside(value)

        wrapped = [("ops", op) for op in kernel.ops]
        texts[kernel.name] = "\n".join(render(wrapped, rename))
        for name, box in _declarations(wrapped, None, []):
            boxes.setdefault(rename(name), box)
    header_text = "\n".join(render(header, outside))
    for name, box in _declarations(header, None, []):
        boxes.setdefault(outside(name), box)
    return Program(
        text="\n".join([header_text, *texts.values()]) + "\n",
        header=header_text,
        kernels=texts,
        boxes=boxes,
        globals=globals_,
        others=others,
        order=kernels,
    )


def peaks(program):
    """Each memory level's highest byte in use: ``{level: bytes}``."""
    top = {}
    for box in program.boxes.values():
        top[box.level] = max(top.get(box.level, 0), box.address + box.size)
    return top


@dataclass
class Comparison:
    """How a version differs from its baseline.

    Attributes:
        same: Kernels whose text is unchanged.
        changed: Kernels in both whose text differs.
        added: Kernels only in the version.
        removed: Kernels only in the baseline.
        header_changed: The inputs, parameters or outputs differ.
        moved: Storage declarations in both whose memory level, partition
            or address differs. A size change shows in the text.
        peaks: ``(old, new)`` peak bytes per memory level.
    """

    same: list
    changed: list
    added: list
    removed: list
    header_changed: bool
    moved: list
    peaks: tuple

    @property
    def equal(self):
        """Whether the versions compare equal, addresses included."""
        return not (
            self.changed
            or self.added
            or self.removed
            or self.header_changed
            or self.moved
        )


def compare(old, new):
    """Compares a version with its baseline, both from ``canonicalize``."""
    return Comparison(
        same=[k for k in new.kernels if old.kernels.get(k) == new.kernels[k]],
        changed=[
            k
            for k in new.kernels
            if k in old.kernels and old.kernels[k] != new.kernels[k]
        ],
        added=[k for k in new.kernels if k not in old.kernels],
        removed=[k for k in old.kernels if k not in new.kernels],
        header_changed=old.header != new.header,
        moved=[
            name
            for name, box in new.boxes.items()
            if name in old.boxes
            and replace(old.boxes[name], size=box.size) != box
        ],
        peaks=(peaks(old), peaks(new)),
    )


def summary(result):
    """Describes a ``Comparison``, one finding per line."""
    lines = [
        f"kernels: same={len(result.same)} changed={len(result.changed)} "
        f"added={len(result.added)} removed={len(result.removed)}"
    ]
    for label in ("changed", "added", "removed"):
        if getattr(result, label):
            lines.append(f"  {label}: {' '.join(getattr(result, label))}")
    if result.header_changed:
        lines.append("inputs, parameters or outputs changed")
    old, new = result.peaks
    peak = ", ".join(
        f"{level} {old.get(level, 0)} -> {new.get(level, 0)} B"
        for level in ("DRAM", "SCRATCHPAD")
    )
    lines.append(f"addresses: {len(result.moved)} moved; peak {peak}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", help="A model.txt; the baseline, given two.")
    parser.add_argument("new", nargs="?", help="The model.txt compared.")
    parser.add_argument(
        "--diff", action="store_true", help="Also print the unified diff."
    )
    args = parser.parse_args()

    with open(args.model) as f:
        old = canonicalize(f.read())
    if args.new is None:
        sys.stdout.write(old.text)
        return 0
    with open(args.new) as f:
        new = canonicalize(f.read(), old)
    print("\n".join(summary(compare(old, new))))
    if args.diff:
        sys.stdout.writelines(
            difflib.unified_diff(
                old.text.splitlines(True),
                new.text.splitlines(True),
                args.model,
                args.new,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
