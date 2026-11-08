# The Legend of Zelda: Ocarina of Time (0400486024DEC000): the rewriter's patches of this game's own code. dm/tools/patch_nso.py loads every profile's patches.py;
# MATCH_FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
from nso_asm import ADRP, B, CBZW

# The game's frame ratio. Once a frame the function at main+0x6e6be0 (v131072) divides its base rate (60.0, or 30.0 while bit 0 of [x21+0x24f] is set) by the int frame rate at
# [x21+0x2c8] (fdiv s8,s1,s0 at +0xe4) and hands the result to the time object (main+0x3b4e300: ratio at +0x10, 1 / ratio at +0x14, round(60 / ratio) at +0x1c), which the game
# steps by; the frame's own time never enters it, so a frame that takes two vsyncs at rate 60 plays at half speed. The stub gives the ratio the host's step instead: step × base / 60
# (the division's value when the host keeps frames 60 / rate vsyncs), and counts the vsyncs stepped. Until the host has decided a step, the division runs.
# At fps_target 0 (the game's own pacing, the iPad's best: no frame is held) the game's frame rate stays 60 whatever the frames take, so below 60 fps it ran in slow motion. There the
# divisor is the whole frame rate the host measured from its presents (state +0x2c, 15..60, hle_nvn.c freal_tick, profile fps_real / DM_FPS_REAL), so the game runs at real speed at any
# frame rate; each such frame adds 60000 / fps (units of 1/60000 s) at +0x38 for the host's "real speed" line. 0 there (not measured yet, or fps_real off) runs the division as it is.
# s0 and x16 / x17 are free there (s0 is written again on every path before it is read; the function uses neither x16 nor x17); s1 keeps the base rate.
# Lichii (a GPL emulator; only the fact was taken) found the division (its notes: main+0x6E6CC4, 60 / [x21+0x2C8])
def frame_step(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF, base = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state, ctx.base
    site = base + 0xe4
    fs_site(site, '2818201e')   # fdiv s8,s1,s0
    words = [
        'adrp16',
        0xb9400211,          # ldr w17,[x16]          target fps (0 = the game's own pacing)
        ('cbz', 17, 'real'),
        0xb9402211,          # ldr w17,[x16,#0x20]    step in vsyncs
        ('cbz', 17, 'orig'), # not decided yet
        0x1e230220,          # ucvtf s0,w17
        0xf9401a10,          # ldr x16,[x16,#0x30]
        0x8b110210,          # add x16,x16,x17
        'adrp17',
        0xf9001a30,          # str x16,[x17,#0x30]    vsyncs stepped
        0x1e210808,          # fmul s8,s0,s1          step × base rate
        0x52a84e10,          # mov w16,#0x42700000    60.0
        0x1e270200,          # fmov s0,w16
        0x1e201908,          # fdiv s8,s8,s0          / 60
        'back',
        'real:',
        0xb9402e11,          # ldr w17,[x16,#0x2c]    the frame rate the host measured (0 = none yet, or fps_real off)
        ('cbz', 17, 'orig'),
        0x1e230220,          # ucvtf s0,w17           the divisor: the measured frame rate
        0x529d4c10,          # mov w16,#60000
        0x1ad10a11,          # udiv w17,w16,w17       this frame's game time in 1/60000 s
        'adrp16',
        0xf9401e10,          # ldr x16,[x16,#0x38]
        0x8b110210,          # add x16,x16,x17
        'adrp17',
        0xf9001e30,          # str x16,[x17,#0x38]    game time fed from the measured rate
        'orig:',
        0x1e201828,          # orig: fdiv s8,s1,s0
        'back',
    ]
    start = here(); labels, at = {}, start   # label: its address (labels take no space)
    for w in words:
        if isinstance(w, str) and w.endswith(':'): labels[w[:-1]] = at
        else: at += 4
    for w in words:
        if isinstance(w, str) and w.endswith(':'): continue
        pc = here()
        if w in ('adrp16', 'adrp17'): w = ADRP(int(w[4:]), pc, FS_STATE_OFF)
        elif w == 'back': w = B(pc, site + 4)
        elif isinstance(w, tuple): w = CBZW(w[1], pc, labels[w[2]])
        emit(w)
    fs_put(site, [B(site, start)], 'frame ratio: the host\'s step × base rate / 60 instead of base rate / the game\'s frame rate (state page target fps); at target 0 base rate / the frame rate the host measured (state +0x2c), or the game\'s ratio')
    dynamic_resolution(ctx)

# The game's dynamic resolution (game__gfx__DynamicResolutionParam: LoadThreshold, ReduceCounterMax, IncreaseCounterMax and a list of levels, each a ResolutionReduceScale and a
# Priority). Its controller's update (main+0x805aa0, v131072) reads the frame's GPU time, subtracts what its cost model predicts at the current size and keeps the residual's trend;
# the level function (main+0x8065e0) adds that residual to the model's cost at each level and returns the best level that stays under the threshold, 0 (the full size, 1536x864)
# when the current size does; the update moves to a lower level after ReduceCounterMax frames that ask for it and back up after IncreaseCounterMax. Its scale divides the 3D views.
# On the Mac the GPU holds 1536x864 within a 60 fps frame, yet the controller sat at 683x384 for minutes after a loading screen (game note, Dynamic resolution). With
# DM_DYNAMIC_RESOLUTION 0 (the default; state +0x40 0) the stub makes the level function return 0, so the update climbs to full resolution and stays; 1 runs the game's function.
# The function is a leaf: x16 / x17 are free at its entry, and w0 is its only result (the original also stores the predicted cost at +0x18, which only this function and the update's
# own decision read)
def dynamic_resolution(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state
    site = ctx.others.get('dynamic resolution')
    if site is None: return   # not found in this build (a red line from the locator): the game's own dynamic resolution
    fs_site(site, '017c40bd')   # ldr s1,[x0,#0x7c]
    start = here(); emit(ADRP(16, here(), FS_STATE_OFF))
    for w in (0xb9404210,   # ldr w16,[x16,#0x40]    dynamic resolution on
              0x35000070,   # cbnz w16,orig (+3)
              0x2a1f03e0,   # mov w0,wzr             level 0: full resolution
              0xd65f03c0,   # ret
              0xbd407c01):  # orig: ldr s1,[x0,#0x7c]
        emit(w)
    emit(B(here(), site + 4))
    fs_put(site, [B(site, start)], 'dynamic resolution level: 0 (full resolution) unless state +0x40 (DM_DYNAMIC_RESOLUTION) runs the game\'s choice')
    ctx.capabilities.add('DYNAMIC_RESOLUTION')

MATCH_FRAME_STEP = {'main': [
    {'name': 'frame ratio from the host\'s step', 'size': 0x9b4, 'fingerprints': ['6ae2aff5344d14568e4dbaf93455e2d8278384bf892ed6c6eca87fef98fbcd1d'], 'stub': frame_step,   # v131072 main+0x6e6be0
     'others': {'dynamic resolution': {'size': 0x294, 'fingerprints': ['7af6ceb0aedf8e0c17dc08ce17fcd6a7cba6d394a0d09e3b504c6d668dc418e7']}}},   # v131072 main+0x8065e0
]}

BINDING = [
    {'what': 'frame ratio from the host\'s step', 'now': 'game', 'goal': 'game', 'purpose': 'frame rate',
     'why': "the game's own frame-ratio division, found by its function's fingerprint on any build that keeps the function; without a target the frame rate the host measured (fps_real)",
     'checked': 'v131072 main: one function with this fingerprint, the site and its free registers read by disassembly (2026-11-08)'},
    {'what': 'dynamic resolution held at full size', 'now': 'game', 'goal': 'game', 'purpose': 'rendering',
     'why': "the game's own dynamic-resolution level function, found by its function's fingerprint on any build that keeps the function; switchable (DM_DYNAMIC_RESOLUTION)",
     'checked': 'v131072 main: one function with this fingerprint (a leaf, x16 / x17 free), its callers and the controller read by disassembly (2026-11-08)'},
]
