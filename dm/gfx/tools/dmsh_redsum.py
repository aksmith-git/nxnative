"""Float atomic adds whose old value is not used, summed across the SIMD group first (red_sum, a step of dmsh_build.py for compute shaders; profile key shader_redsum).

The translator writes NVN's float atomic add (RED.F32.FTZ.RN, ATOM / ATOMS .F32) as a compare-and-swap loop on the word (InstEmitMemory.cs EmitAtomicAddF32: neither SPIR-V
as the chain uses it nor Metal's texture atomics have a float add), and spirv-cross writes it as:

    bool _A;
    do
    {
        [uint _i = ...;]                                    the index, when the translator computes it inside the loop (Ocarina of Time's 21d9f83e6027c565)
        float _O = as_type<float>(BUF[I]);
        float _N = _O + V;
        [uint _j = ...;]                                    the same index again, for the exchange
        uint _E = as_type<uint>(_O);
        uint _C;
        do
        {
            _C = _E;
        } while (!atomic_compare_exchange_weak_explicit((device atomic_uint*)&BUF[J], &_C, as_type<uint>(_N), memory_order_relaxed, memory_order_relaxed) && _C == _E);
        uint _R = _C;
        _A = int(_R) != as_type<int>(_O);
    } while (_A);

When every thread of a dispatch adds into the same few words (Ocarina of Time's exposure and cubemap-to-SH reductions: 256 threads, or 96 groups of 256, into 1-4 words), the
loop serialises the dispatch on one cache line and retries under contention: 1bba09ec4bc8ff67 took 361 us and 6e28359aa3558f61 289 us on an M5 for trivial maths
(ZEL_build/agents/disp2.md section 4). When none of the loop's values is used after it (a pure accumulate, RED's form), the SIMD group sums its values first and one lane
adds the sum: 32x fewer atomics. In device memory the add is Metal's native float atomic (atomic_float, MSL 3.0; every Apple GPU of the M-series and A14+), with no
retry loop: 1bba 361 -> 9 us, 6e28 289 -> 70 us, 0.69 ms of the frame over the six shaders that have the loop. Threadgroup memory keeps the compare-and-swap, made by
the one lane (Metal guarantees atomic_float in device memory only; the elected lane alone is most of the gain anyway).

simd_sum adds the active lanes only, so a loop inside divergent control flow keeps its meaning. The sum's rounding differs from a serial order, as the order of the
atomics already does from run to run. The index:
- uniform in the dispatch (literals, and constant-buffer words at literal offsets, through the loop's own index statements): one sum, from the first active lane;
- anything else: lanes are grouped by index, one group per round (simd_broadcast_first picks the next index, the lanes that have it sum and leave), so lanes with
  different words never mix. A uniform index takes one round of this form too; the first form only saves the broadcast.
A loop is left as it is when any of its values is used after it in its function, when the loaded and the exchanged word differ (base or resolved index), or when the added value
reads the old one.
"""
import re

_IND = r'(?P<ind>[ \t]+)'
_DEF = r'(?:(?P=ind)    (?:u?int) _\d+ = [^;\n]+;\n)*'   # an index statement inside the loop
# the buffer word: a bound storage buffer's ((*spvDescriptorSet0.cp_s0)._m0[...]) or a threadgroup array's (_123[...])
_WORD = r'(?P<{b}>\(\*spvDescriptorSet0\.\w+\)\._m0|\w+)\[(?P<{i}>[^\]\n]+)\]'
LOOP = re.compile(
    _IND + r'bool (?P<A>_\d+);\n(?P=ind)do\n(?P=ind)\{\n'
    r'(?P<pre>' + _DEF + r')'
    r'(?P=ind)    float (?P<O>_\d+) = as_type<float>\(' + _WORD.format(b='base', i='idx') + r'\);\n'
    r'(?P=ind)    float (?P<N>_\d+) = (?P=O) \+ (?P<val>[^;\n]+);\n'
    r'(?P<pre2>' + _DEF + r')'
    r'(?P=ind)    uint (?P<E>_\d+) = as_type<uint>\((?P=O)\);\n'
    r'(?P=ind)    uint (?P<C>_\d+);\n(?P=ind)    do\n(?P=ind)    \{\n(?P=ind)        (?P=C) = (?P=E);\n'
    r'(?P=ind)    \} while \(!atomic_compare_exchange_weak_explicit\(\((?P<space>device|threadgroup) atomic_uint\*\)&' + _WORD.format(b='base2', i='idx2') +
    r', &(?P=C), as_type<uint>\((?P=N)\), memory_order_relaxed, memory_order_relaxed\) && (?P=C) == (?P=E)\);\n'
    r'(?P=ind)    uint (?P<R>_\d+) = (?P=C);\n'
    r'(?P=ind)    (?P=A) = int\((?P=R)\) != as_type<int>\((?P=O)\);\n'
    r'(?P=ind)\} while \((?P=A)\);\n')

_CONST_WORD = re.compile(r'\(\*spvDescriptorSet0\.[a-z]p_c\d+\)\._m0\[\d+u?\](?:\.[xyzw])?')   # a constant buffer's word at a literal offset: the same in every thread
_UNIFORM_LEFT = re.compile(r'(?:as_type<(?:u?int|float)>|\bu?int\b|\b\d+u?\b|K|[\s()+\-*/%<>&|^~])*')


def _resolve(expr, defs):   # the expression with the loop's own index statements substituted, to the bottom
    while True:
        nx = re.sub(r'\b(_\d+)\b', lambda k: '(' + defs[k.group(1)] + ')' if k.group(1) in defs else k.group(1), expr)
        if nx == expr: return expr
        expr = nx


def uniform(expr):
    """Whether a resolved index expression has the same value in every thread: integer literals, casts and arithmetic over constant-buffer words at literal offsets"""
    return _UNIFORM_LEFT.fullmatch(_CONST_WORD.sub('K', expr)) is not None


def red_sum(msl):
    """→ (the shader with each dead-result float atomic add loop as a SIMD-group sum and one add, [(word, 'uniform' / 'grouped', address space)] of the rewrites,
    [reasons of the loops left alone])"""
    if 'atomic_compare_exchange_weak_explicit' not in msl: return msl, [], []
    done, kept, out, at = [], [], [], 0
    for m in LOOP.finditer(msl):
        end = msl.find('\n}', m.end())   # the end of the function (spirv-cross closes one at column 0): a pixel shader's other entry points repeat main0's names
        tail = msl[m.end():end if end >= 0 else len(msl)]
        used = next((v for v in (m.group(k) for k in ('O', 'N', 'E', 'C', 'R', 'A')) if re.search(r'\b' + v + r'\b', tail)), None)
        if used: kept.append(f'{m.group("base2")}: old value {used} used after the loop'); continue
        if re.search(r'\b(?:%s|%s)\b' % (m.group('O'), m.group('N')), m.group('val')): kept.append(f'{m.group("base2")}: the added value reads the old one'); continue
        defs = dict(re.findall(r'u?int (_\d+) = ([^;\n]+);', m.group('pre') + m.group('pre2')))
        idx = _resolve(m.group('idx2'), defs)
        if m.group('base') != m.group('base2') or _resolve(m.group('idx'), defs) != idx:
            kept.append(f'{m.group("base2")}: the loaded and the exchanged word differ ({m.group("idx")} / {m.group("idx2")})'); continue
        ind, space, word = m.group('ind'), m.group('space'), m.group('base2') + '[%s]'
        keep = re.sub('^' + ind + '    ', ind, m.group('pre') + m.group('pre2'), flags=re.M)   # the index statements, now before the add
        if space == 'device':
            add = 'atomic_fetch_add_explicit((device atomic_float*)&%s, dm_s, memory_order_relaxed);'
        else:   # one compare-and-swap loop for the SIMD group's sum
            add = ('{ threadgroup atomic_uint* dm_p = (threadgroup atomic_uint*)&%s; uint dm_o = atomic_load_explicit(dm_p, memory_order_relaxed); '
                   'while (!atomic_compare_exchange_weak_explicit(dm_p, &dm_o, as_type<uint>(as_type<float>(dm_o) + dm_s), memory_order_relaxed, memory_order_relaxed)) {} }')
        if uniform(idx):
            body = '%s{ float dm_s = simd_sum(%s); if (simd_is_first()) %s }\n' % (ind, m.group('val'), add % (word % m.group('idx2')))
            form = 'uniform'
        else:
            body = ('%s{ int dm_r = int(%s); float dm_v = %s; for (;;) { int dm_f = simd_broadcast_first(dm_r); if (dm_r == dm_f) { float dm_s = simd_sum(dm_v); '
                    'if (simd_is_first()) %s break; } } }\n') % (ind, m.group('idx2'), m.group('val'), add % (word % 'dm_f'))
            form = 'grouped'
        out.append(msl[at:m.start()] + keep + body); at = m.end()
        done.append((word % m.group('idx2'), form, space))
    out.append(msl[at:])
    return ''.join(out), done, kept
