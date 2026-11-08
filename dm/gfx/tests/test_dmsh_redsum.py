"""Float atomic adds whose old value is not used, summed per SIMD group (dmsh_redsum.py red_sum): which loops are rewritten and into which form, which are left alone, and the compile."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dmsh_redsum import red_sum, uniform   # noqa: E402

HEAD = '''#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

struct _16
{
    uint4 _m0[4096];
};

struct _25
{
    uint _m0[1];
};

struct spvDescriptorSetBuffer0
{
    constant _16* cp_c1 [[id(0)]];
    device _25* cp_s0 [[id(1)]];
};

'''


def loop(a, word_load, word_cas, val, pre='', pre2='', space='device', ind='    '):
    """The translator's float atomic add as spirv-cross writes it; a is the first of its value numbers"""
    O, N, E, C, R = (f'_{a + k}' for k in range(1, 6))
    return (f'{ind}bool _{a};\n{ind}do\n{ind}{{\n{pre}'
            f'{ind}    float {O} = as_type<float>({word_load});\n'
            f'{ind}    float {N} = {O} + {val};\n{pre2}'
            f'{ind}    uint {E} = as_type<uint>({O});\n'
            f'{ind}    uint {C};\n{ind}    do\n{ind}    {{\n{ind}        {C} = {E};\n'
            f'{ind}    }} while (!atomic_compare_exchange_weak_explicit(({space} atomic_uint*)&{word_cas}, &{C}, as_type<uint>({N}), memory_order_relaxed, memory_order_relaxed) && {C} == {E});\n'
            f'{ind}    uint {R} = {C};\n'
            f'{ind}    _{a} = int({R}) != as_type<int>({O});\n'
            f'{ind}}} while (_{a});\n')


def kernel(body, extra=''):
    return (HEAD + 'kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])\n{\n'
            + extra + '    float _100 = float(gl_LocalInvocationID.x) * 0.5;\n' + body + '}\n\n')


S0 = '(*spvDescriptorSet0.cp_s0)._m0'
# Ocarina of Time's 278f48371bcf0d58 / 6e28359aa3558f61: every thread into one word at a literal index
LITERAL = kernel(loop(200, S0 + '[0]', S0 + '[0]', '_100') + loop(210, S0 + '[3]', S0 + '[3]', '_100', ind='        ').join(('    if (_100 > 1.0)\n    {\n', '    }\n')))
# 21d9f83e6027c565: the index computed inside the loop from a constant buffer word, twice (load, exchange)
CB = '(*spvDescriptorSet0.cp_c1)._m0[0].x'
IN_LOOP = kernel(loop(300, S0 + '[int(_320)]', S0 + '[int(_322)]', '_100',
                      pre=f'        int _319 = as_type<int>({CB}) + 4;\n        uint _320 = uint(int(uint(_319) >> uint(2)));\n',
                      pre2=f'        int _321 = as_type<int>({CB}) + 4;\n        uint _322 = uint(int(uint(_321) >> uint(2)));\n'))
# a word per thread group of 4 lanes: lanes with different words must not be summed together
PER_THREAD = kernel(loop(400, S0 + '[int(_420)]', S0 + '[int(_420)]', '_100'), '    uint _420 = gl_LocalInvocationID.x >> 2u;\n')
# threadgroup memory: the sum, then a compare-and-swap by the one lane
SHARED = kernel(loop(500, '_60[0]', '_60[0]', '_100', space='threadgroup') + '    threadgroup_barrier(mem_flags::mem_threadgroup);\n    (*spvDescriptorSet0.cp_s0)._m0[1] = _60[0];\n',
                '    threadgroup uint _60[4];\n')


class Rewrite(unittest.TestCase):
    def test_literal_index(self):
        out, done, kept = red_sum(LITERAL)
        self.assertEqual(done, [(S0 + '[0]', 'uniform', 'device'), (S0 + '[3]', 'uniform', 'device')])
        self.assertEqual(kept, [])
        self.assertIn(f'    {{ float dm_s = simd_sum(_100); if (simd_is_first()) atomic_fetch_add_explicit((device atomic_float*)&{S0}[0], dm_s, memory_order_relaxed); }}\n', out)
        self.assertIn(f'        {{ float dm_s = simd_sum(_100); if (simd_is_first()) atomic_fetch_add_explicit((device atomic_float*)&{S0}[3], dm_s, memory_order_relaxed); }}\n', out)
        self.assertNotIn('atomic_compare_exchange', out)
        self.assertIn('if (_100 > 1.0)', out)   # the divergent block stays: simd_sum takes the active lanes

    def test_index_inside_the_loop(self):
        out, done, kept = red_sum(IN_LOOP)
        self.assertEqual([d[1] for d in done], ['uniform'])
        for v in ('_319', '_320', '_321', '_322'): self.assertRegex(out, r'\n    u?int %s = ' % v)   # the index statements, now before the add, one level out
        self.assertIn(f'atomic_fetch_add_explicit((device atomic_float*)&{S0}[int(_322)], dm_s', out)

    def test_per_thread_index_groups_lanes(self):
        out, done, kept = red_sum(PER_THREAD)
        self.assertEqual(done, [(S0 + '[int(_420)]', 'grouped', 'device')])
        self.assertIn('int dm_r = int(int(_420)); float dm_v = _100; for (;;) { int dm_f = simd_broadcast_first(dm_r); if (dm_r == dm_f) { float dm_s = simd_sum(dm_v); '
                      f'if (simd_is_first()) atomic_fetch_add_explicit((device atomic_float*)&{S0}[dm_f], dm_s, memory_order_relaxed); break; }} }}', out)

    def test_threadgroup_keeps_a_compare_and_swap(self):
        out, done, kept = red_sum(SHARED)
        self.assertEqual(done, [('_60[0]', 'uniform', 'threadgroup')])
        self.assertNotIn('atomic_float', out)
        self.assertEqual(out.count('atomic_compare_exchange_weak_explicit'), 1)
        self.assertIn('if (simd_is_first()) { threadgroup atomic_uint* dm_p = (threadgroup atomic_uint*)&_60[0];', out)

    def test_old_value_used(self):
        src = LITERAL.replace('    }\n}\n', '    }\n    (*spvDescriptorSet0.cp_s0)._m0[7] = as_type<uint>(_201);\n}\n')
        out, done, kept = red_sum(src)
        self.assertEqual([d[0] for d in done], [S0 + '[3]'])
        self.assertEqual(kept, [S0 + ': old value _201 used after the loop'])
        self.assertIn('as_type<float>((*spvDescriptorSet0.cp_s0)._m0[0])', out)

    def test_names_of_another_function_do_not_count(self):   # a pixel shader's other entry points repeat main0's value numbers
        src = LITERAL + 'static inline float other(float _201) { return _201; }\n'
        self.assertEqual(len(red_sum(src)[1]), 2)

    def test_different_words(self):
        out, done, kept = red_sum(kernel(loop(600, S0 + '[0]', S0 + '[1]', '_100')))
        self.assertEqual(done, [])
        self.assertEqual(len(kept), 1)
        self.assertIn('differ', kept[0])
        src = IN_LOOP.replace(f'int _321 = as_type<int>({CB}) + 4;', f'int _321 = as_type<int>({CB}) + 8;')
        self.assertEqual(red_sum(src)[1], [])

    def test_value_reading_the_old_one(self):
        self.assertEqual(red_sum(kernel(loop(700, S0 + '[0]', S0 + '[0]', '_701')))[1], [])

    def test_nothing_to_do(self):
        src = kernel('')
        self.assertEqual(red_sum(src), (src, [], []))

    def test_uniform(self):
        for e in ('0', '32', 'int(uint(int(uint((as_type<int>((*spvDescriptorSet0.cp_c1)._m0[0].x) + 4)) >> uint(2))))', '(*spvDescriptorSet0.cp_c1)._m0[3u].y * 2'):
            with self.subTest(e): self.assertTrue(uniform(e))
        for e in ('int(_420)', 'gl_LocalInvocationID.x', '(*spvDescriptorSet0.cp_c1)._m0[_5].x', '(*spvDescriptorSet0.cp_s0)._m0[0]', 'f(0)'):
            with self.subTest(e): self.assertFalse(uniform(e))

    @unittest.skipUnless(shutil.which('xcrun') and sys.platform == 'darwin', 'Metal compiler')
    def test_compiles(self):   # the build's two compiles (dmsh_build.py MAC_METAL, ios_metal.METAL)
        cmds = {'macosx': ['xcrun', '-sdk', 'macosx', 'metal', '-c', '-O1', '-fpreserve-invariance'],
                'iphoneos': ['xcrun', '-sdk', 'iphoneos', 'metal', '-c', '-O1', '-mios-version-min=18.0', '-fmetal-math-mode=relaxed', '-fpreserve-invariance']}
        with tempfile.TemporaryDirectory() as d:
            for name, src in (('literal', LITERAL), ('in_loop', IN_LOOP), ('per_thread', PER_THREAD), ('shared', SHARED)):
                for sdk, cmd in cmds.items():
                    for stage, text in (('before', src), ('after', red_sum(src)[0])):
                        with self.subTest(name=name, sdk=sdk, stage=stage):
                            m = os.path.join(d, f'{name}_{stage}.metal'); Path(m).write_text(text)
                            r = subprocess.run([*cmd, m, '-o', os.path.join(d, f'{name}_{stage}_{sdk}.air')], capture_output=True, text=True)
                            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
