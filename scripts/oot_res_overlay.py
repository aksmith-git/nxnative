#!/usr/bin/env python3
# oot_res_overlay.py <RomFS.img> <overlay folder> <top scale> [<LoadThreshold> | -] [flat]
# The Legend of Zelda: Ocarina of Time (Switch 2, 0400486024DEC000): its render resolution, through the game's own dynamic-resolution table. Writes a modified copy of
# Pack/Bootup.Nin_Ounce_NVN2.pack.zs into <overlay folder>/Pack/, for DM_ROMFS_OVERLAY (the IPA's romfs_overlay folder, the Mac app's --romfs-overlay); the RomFS image itself is not changed.
# In System/GameGraphics/ModelSceneExtensionParam the game has one base resolution per mode (ResolutionParam: 1920x1080 handheld, 2560x1440 docked, of which docked shows 2048x1152) and dynamic-resolution levels
# (DynamicResolutionParam: ResolutionReduceScale 0.75 down to 0.333, LoadThreshold 95) that multiply that base. Every ResolutionReduceScale is multiplied by <top scale> / 0.75, so the top
# level renders at <top scale> x the base and the lower levels keep their order; with "flat" every level is <top scale>, one fixed resolution (the game then never makes a new set of render
# targets). The optional fourth argument sets every LoadThreshold (GPU load in percent over which the game steps down; '-' keeps them). Recompressed with the dictionary the original used.
# Examples (docs/games/kid-with-sword-remake.md): into <romfs_overlay>/handheld, 0.666667 - flat = 1280x720; into <romfs_overlay>/docked, 0.9375 - flat = 1951x1098 (docked sizes come in
# steps of about 45 pixels; 1920x1080 is not one). The host serves the subfolder for the console mode (hle_fs.c rom_overlay_set).
# Needs the zstandard package (pip install zstandard). The game's files are read from your own dump; nothing of the game is in this repository.
import sys, os, struct, zstandard
img, out, top = sys.argv[1], sys.argv[2], float(sys.argv[3]); thr = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '-' else None   # optional: every LoadThreshold set to this ('-' keeps them)
flat = len(sys.argv) > 5 and sys.argv[5] == 'flat'   # every level at <top scale>: one fixed resolution, so the dynamic resolution never makes the game build a new set of render targets
sys.path.insert(0, os.environ.get('DMT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dm', 'tools')); import romfs_image as r
idx = r.read_index(img); f = open(img, 'rb')
def read(p): _, o, n = idx[p]; f.seek(o); return f.read(n)
def sarc_files(b):
    hl, bom, size, doff = struct.unpack_from('<HHII', b, 4); o = hl; sh, cnt, mult = struct.unpack_from('<HHI', b, o + 4); o += sh
    nodes = [struct.unpack_from('<IIII', b, o + 16 * i) for i in range(cnt)]; o += 16 * cnt + 8
    return {b[o + (a & 0xffffff) * 4:].split(b'\0', 1)[0].decode(): (doff + s, doff + e) for h, a, s, e in nodes}
dz = zstandard.ZstdDecompressor()
dicts = sarc_files(dz.decompress(read('Pack/ZsDic.pack.zs'), max_output_size=1 << 28)); zd = dz.decompress(read('Pack/ZsDic.pack.zs'), max_output_size=1 << 28)
PACK = 'Pack/Bootup.Nin_Ounce_NVN2.pack.zs'; raw = read(PACK); used = None
for name, (s, e) in dicts.items():
    try: data = bytearray(zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(zd[s:e])).decompress(raw, max_output_size=1 << 30)); used = name; break
    except Exception: pass
assert used, 'no dictionary decompressed the pack'
fs = sarc_files(bytes(data)); PARAM = 'System/GameGraphics/ModelSceneExtensionParam/Default.game__gfx__ModelSceneExtensionParam.bgyml'
s, e = fs[PARAM]; y = data[s:e]
# the BYML: find every float (type 0xD2) value whose dict key is ResolutionReduceScale
ver, ko, so, ro = struct.unpack_from('<HIII', y, 2); n = struct.unpack_from('<I', y, ko)[0] >> 8; offs = struct.unpack_from('<%dI' % (n + 1), y, ko + 4)
keys = [y[ko + offs[i]: y.index(b'\0', ko + offs[i])].decode() for i in range(n)]; K = keys.index('ResolutionReduceScale')
done = set()
for o in range(0, len(y) - 8, 4):   # dict entries are 8 bytes: u24 key | u8 type, u32 value; scan for (K, 0xD2)
    k = struct.unpack_from('<I', y, o)[0]
    if k == (K | 0xD2 << 24) and o + 4 not in done:
        v = struct.unpack_from('<f', y, o + 4)[0]
        if 0.05 < v <= 1.0: nv = top if flat else v * top / 0.75; struct.pack_into('<f', y, o + 4, nv); done.add(o + 4); print('ResolutionReduceScale %.5f -> %.5f (@%#x)' % (v, nv, o + 4))
assert done, 'no ResolutionReduceScale found'
if thr is not None:
    L = keys.index('LoadThreshold')
    for o in range(0, len(y) - 8, 4):
        if struct.unpack_from('<I', y, o)[0] == (L | 0xD2 << 24): v = struct.unpack_from('<f', y, o + 4)[0]; struct.pack_into('<f', y, o + 4, thr); print('LoadThreshold %g -> %g (@%#x)' % (v, thr, o + 4))
data[s:e] = y
cz = zstandard.ZstdCompressor(level=19, dict_data=zstandard.ZstdCompressionDict(zd[dicts[used][0]:dicts[used][1]]))
os.makedirs(os.path.join(out, 'Pack'), exist_ok=True); open(os.path.join(out, PACK), 'wb').write(cz.compress(bytes(data)))
print('wrote', os.path.join(out, PACK), 'dictionary', used)
