## B4.3 MeshConv operators vs shipped `icosphere_*.pkl`

**Disassembly** (`pickletools.genops`, never loaded): `icosphere_{0..5}.pkl` are
pickle protocol 2 (Python 2 era), 12,414 B (level 0) to 10,984,603 B (level 5).
Opcodes seen: `PROTO EMPTY_DICT MARK SETITEMS GLOBAL NEWOBJ REDUCE BUILD TUPLE*
BIN{PUT,GET} BININT* BINUNICODE NONE NEWTRUE NEWFALSE EMPTY_TUPLE STOP` — no
`STACK_GLOBAL`/`INST`/`OBJ`. GLOBALs, identical in all six files:

- `numpy.core.multiarray _reconstruct`
- `numpy ndarray`
- `numpy dtype`
- `_codecs encode` (used only as `encode(<str>, 'latin1')` for array buffers)
- `scipy.sparse.csc csc_matrix`
- `scipy.sparse.coo coo_matrix`

Loaded with a restricted `Unpickler` allowing exactly these.

**Keys**: `V F nv_prev nv_next G L N NS EW F2V M Seq Intp` (all levels).

**Levels**: 0..5 (no 6/7 pickles; `icosphere_5.vtk`, `icosphere_6.vtk` only).

**Max abs diff, shipped vs `build_operators(level)`** (limit 1e-6; sparse stored
patterns identical, `V F nv_prev nv_next Seq Intp` exactly 0):

| level | G | L | N | NS | EW | F2V | M |
|---|---|---|---|---|---|---|---|
| 0 | 2.2e-16 | 1.8e-15 | 0 | 2.2e-16 | 4.4e-16 | 0 | 2.2e-16 |
| 1 | 8.9e-16 | 8.9e-16 | 0 | 5.6e-16 | 5.6e-16 | 8.3e-17 | 5.6e-17 |
| 2 | 2.2e-15 | 1.3e-15 | 0 | 1.3e-15 | 1.0e-15 | 8.3e-17 | 2.8e-17 |
| 3 | 5.3e-15 | 1.8e-15 | 0 | 3.8e-15 | 3.7e-15 | 8.3e-17 | 6.9e-18 |
| 4 | 1.1e-14 | 1.8e-15 | 0 | 9.5e-15 | 9.5e-15 | 8.3e-17 | 1.7e-18 |
| 5 | 2.5e-14 | 2.2e-15 | 0 | 2.1e-14 | 2.1e-14 | 1.1e-16 | 6.5e-19 |

`.vtk` meshes: level 5 and 6 faces equal; vertices differ by <= 5.0e-7 (the
file prints 6 decimals as `float`).

**Unmatched fields**: none. dtypes: V/G/L/N/NS/EW/F2V/M/Intp float64; F, Seq int64.
