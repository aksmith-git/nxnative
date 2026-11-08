"""Execute Ocarina of Time's emitted frame-ratio stub (patches.py frame_step) without game files: the host's step with a target, the measured frame rate without one (fps_real)."""

from importlib import import_module
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

PROFILE = TOOLS.parent / "games/0400486024DEC000/patches.py"
SPEC = importlib.util.spec_from_file_location("oot_patches", PROFILE)
assert SPEC is not None and SPEC.loader is not None
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)


class FrameRatio(unittest.TestCase):
    mem, func, stub, state, stack = 0x10000000, 0x6e6be0, 0x5100190, 0x4ea4000, 0x18001000

    def setUp(self):
        try:
            uc = import_module("unicorn")
            self.arm = import_module("unicorn.arm64_const")
        except ImportError:
            self.skipTest("optional ARM64 execution dependency: unicorn")
        self.cpu = uc.Uc(uc.UC_ARCH_ARM64, uc.UC_MODE_ARM)
        for address in (self.mem + (self.func & ~0xfff), self.mem + (self.stub & ~0xfff), self.mem + self.state, self.stack):
            self.cpu.mem_map(address, 4096)
        self.cursor, self.sites, self.cuts = self.stub, [], []
        self.base, self.others, self.capabilities = self.func, {}, set()   # no dynamic resolution site: that stub is not emitted
        PATCH.frame_step(self)
        self.site_at = self.func + 0xe4
        self.assertIn((self.site_at, "2818201e"), self.sites)
        self.assertIn(self.site_at, self.cuts)
        self.end = self.mem + self.site_at + 4
        self.cpu.hook_add(uc.UC_HOOK_CODE, lambda cpu, *_: cpu.emu_stop(), begin=self.end, end=self.end)

    def here(self): return self.cursor
    def site(self, offset, original): self.sites.append((offset, original))
    def emit(self, word):
        self.cpu.mem_write(self.mem + self.cursor, struct.pack("<I", word)); self.cursor += 4
    def put(self, offset, words, _reason):
        self.cuts.append(offset); self.cpu.mem_write(self.mem + offset, struct.pack("<" + "I" * len(words), *words))
    def u64(self, off): return struct.unpack("<Q", self.cpu.mem_read(self.mem + self.state + off, 8))[0]

    def ratio(self, target, step, game_fps=60, measured=0, base=60.0):
        arm = self.arm
        self.cpu.mem_write(self.mem + self.state, struct.pack("<f", target))
        self.cpu.mem_write(self.mem + self.state + 0x20, struct.pack("<I", step))
        self.cpu.mem_write(self.mem + self.state + 0x2c, struct.pack("<I", measured))
        kept = {arm.UC_ARM64_REG_X8: 0x123456789, arm.UC_ARM64_REG_X15: 0x987654321, arm.UC_ARM64_REG_X21: 0x5555, arm.UC_ARM64_REG_SP: self.stack + 0x800,
                arm.UC_ARM64_REG_NZCV: 0xa0000000}
        for r, v in kept.items(): self.cpu.reg_write(r, v)
        self.cpu.reg_write(arm.UC_ARM64_REG_S0, struct.unpack("<I", struct.pack("<f", float(game_fps)))[0])   # the game's scvtf of [x21+0x2c8]
        self.cpu.reg_write(arm.UC_ARM64_REG_S1, struct.unpack("<I", struct.pack("<f", base))[0])
        self.cpu.emu_start(self.mem + self.site_at, self.end, count=64)
        self.assertEqual(self.cpu.reg_read(arm.UC_ARM64_REG_PC), self.end)
        for r, v in kept.items(): self.assertEqual(self.cpu.reg_read(r), v)
        return round(struct.unpack("<f", struct.pack("<I", self.cpu.reg_read(arm.UC_ARM64_REG_S8) & 0xffffffff))[0], 4)

    def test_target_follows_the_host_step(self):
        self.assertEqual(self.ratio(60, 1), 1.0)
        self.assertEqual(self.ratio(60, 2), 2.0)
        self.assertEqual(self.ratio(60, 2, base=30.0), 1.0)
        self.assertEqual(self.u64(0x30), 5)   # 1 + 2 + 2 vsyncs stepped
        self.assertEqual(self.u64(0x38), 0)

    def test_target_before_a_step_runs_the_division(self):
        self.assertEqual(self.ratio(60, 0, game_fps=30), 2.0)
        self.assertEqual(self.u64(0x30), 0)

    def test_target_zero_divides_by_the_measured_frame_rate(self):
        self.assertEqual(self.ratio(0, 2, measured=45), round(60 / 45, 4))   # a step left in +0x20 does not matter without a target
        self.assertEqual(self.u64(0x38), 1333)
        self.assertEqual(self.ratio(0, 0, measured=30), 2.0)
        self.assertEqual(self.u64(0x38), 1333 + 2000)
        self.assertEqual(self.ratio(0, 0, measured=60, base=30.0), 0.5)
        self.assertEqual(self.u64(0x38), 1333 + 2000 + 1000)
        self.assertEqual(self.u64(0x30), 0)

    def test_target_zero_without_a_measurement_keeps_the_division(self):
        self.assertEqual(self.ratio(0, 1, game_fps=60, measured=0), 1.0)
        self.assertEqual(self.ratio(0, 1, game_fps=20, measured=0), 3.0)
        self.assertEqual(self.u64(0x38), 0)


if __name__ == "__main__":
    unittest.main()
