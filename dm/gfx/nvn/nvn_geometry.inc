// General geometry replay: native VS capture -> GS compute -> ordered index
// compaction -> ordinary rendering. Included after the resource/pipeline helpers.
static NvProg *g_geo_program;
static uint64_t g_geo_indices, g_geo_index_bytes, g_geo_args;
static const p_draw *g_geo_vrange;
static uint32_t g_geo_count;
static uint64_t g_geo_hoisted, g_geo_hoist_runs;
static int geo_hoist_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_GEO_HOIST"); on = !e || strcmp(e, "0"); } return on; }
static id<MTLComputePipelineState> g_geo_lean_k[3];
static void draw(const p_draw *dw, const p_draw_idx *di);
static uint32_t draw_color_writes(NvProg *pr);
static void lc_land(void);
static void sub_cut(void);

static int geo_cmp_index(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
enum { GEO_CAPTURE_VERTICES = (256u << 20) / (144 * 4) };
enum { GEO_COMPACT_MIN = 1u << 20, GEO_COMPACT_RUNS = 64 };   // compact captures (nvn_geometry.h): what one must save, and how many capture draws it may take
// The capture span is already bounded by its scratch allocation. A bit per
// vertex gives the same ascending unique list without sorting every reference.
// Keep an in-place sort fallback for allocation failure or a wider caller.
// order may be changed; a short destination is never partially written.
static size_t geo_capture_unique(uint32_t *order, size_t count, int restart, uint32_t restart_index,
                                 uint32_t lo, uint64_t span, uint32_t *dst, size_t capacity) {
    if (!span || span > (uint64_t)UINT32_MAX + 1 - lo) return SIZE_MAX;
    for (size_t i = 0; i < count; i++) if (!(restart && order[i] == restart_index) &&
        (order[i] < lo || (uint64_t)order[i] - lo >= span)) return SIZE_MAX;
    size_t nw = (size_t)((span + 63) / 64), n = 0;
    uint64_t *bits = span <= GEO_CAPTURE_VERTICES ? calloc(nw, sizeof *bits) : NULL;
    if (bits) {
        for (size_t i = 0; i < count; i++) if (!(restart && order[i] == restart_index)) {
            uint32_t v = order[i] - lo; bits[v >> 6] |= 1ull << (v & 63);
        }
        for (size_t i = 0; i < nw; i++) n += (size_t)__builtin_popcountll(bits[i]);
        if (n <= capacity) for (size_t i = 0, k = 0; i < nw; i++) {
            uint64_t word = bits[i];
            while (word) { dst[k++] = lo + (uint32_t)(i * 64 + __builtin_ctzll(word)); word &= word - 1; }
        }
        free(bits);
    } else {
        size_t references = 0;
        for (size_t i = 0; i < count; i++) if (!(restart && order[i] == restart_index)) order[references++] = order[i];
        qsort(order, references, sizeof *order, geo_cmp_index);
        for (size_t i = 0; i < references; i++) if (!i || order[i] != order[i - 1]) n++;
        if (n <= capacity) for (size_t i = 0, k = 0; i < references; i++)
            if (!i || order[i] != order[i - 1]) dst[k++] = order[i];
    }
    return n <= capacity ? n : SIZE_MAX;
}
// Only CPU reads need a completed snapshot. Native vertex capture stays on the
// GPU and follows earlier writes through normal encoder ordering. Unknown or
// overflowing write sets always retain the conservative drain.
static int geo_cpu_hazard(uint64_t lo, uint64_t n) {
    if (!n) return 0;
    if (!lo || n > UINT64_MAX - lo) return 1;
    if (drw_valid()) return drw_hits(lo, n) || nvn_sparse_alias(lo, lo + n);   // drained in this submission: only later writes count
    if (g_sbw_over) return 1;
    return pend_hits(lo, n) || sbw_hits(lo, lo + n) || gw_live(lo, lo + n);
}
static int geo_conversion_hazard(NvShader *vs) {
    for (int v = 0; v < vs->rf.nv; v++) {
        int a = vs->rf.v[v].nvn;
        if (a >= S.nattr || !vconv_wanted(&vs->rf, v, NULL, 0)) continue;
        int st = S.attr[a].stream;
        if (st < 16 && S.vbuf[st].address && geo_cpu_hazard(S.vbuf[st].address, S.vbuf[st].size)) return 1;
    }
    return 0;
}
// GS output -> counts -> offsets/indirect args -> packed indices stay in one
// compute encoder. Each transition needs dispatch ordering and device-memory
// visibility; full queue barriers remain at the encoder's boundaries.
static void geo_pack_encode(id<MTL4ComputeCommandEncoder> ce, id<MTLComputePipelineState> __strong *pack, uint64_t job, uint32_t groups) {
    enc_c_args(ce, job, 0);
    for (int i = 0; i < 3; i++) {
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_pso(ce, pack[i]);
        NSUInteger threads = i == 1 ? MIN((NSUInteger)256, pack[i].maxTotalThreadsPerThreadgroup) : 64;
        enc_c_dispatch(ce, i == 1 ? MTLSizeMake(1, 1, 1) : MTLSizeMake((groups + 63) / 64, 1, 1), MTLSizeMake(threads, 1, 1));
    }
}

// CPU primitive assembly may read static indices without draining unrelated GPU
// work. Unknown writes and both current/prior submission hazards retain the wait.
static int geo_index_wait(uint64_t address, uint64_t bytes) {
    static int always = -1;
    if (always < 0) { const char *e = getenv("DM_NVN_GEOMETRY_WAIT"); always = e && !strcmp(e, "always"); }
    return always || !address || !bytes || bytes > (16ull << 20) || bytes > UINT64_MAX - address ||
           g_pend_over || nvn_sparse_alias(address, address + bytes) || geo_cpu_hazard(address, bytes);
}

static id<MTLRenderPipelineState> geo_capture_pso(NvShader *vs) {
    pso_key k; static const uint32_t cf[8]; pso_key_fill(&k, vs, nil, cf, 0, 0, 1, 0); k.fsfn = GEO_MARK;
    id<MTLRenderPipelineState> ps = hm_get_id(&g_pso, &k);
    if (ps || hm_get_id(&g_pso_fail, &k)) return ps;
    NSError *e = nil;
    ps = [g_dev newRenderPipelineStateWithDescriptor:pso_desc(&k, vs, nil, nil, 0) error:&e];
    if (ps) hm_put_id(&g_pso, &k, ps);
    else { hm_put_id(&g_pso_fail, &k, NSNull.null); MLOG("[RED] geometry vertex capture %s: %s", vs->key, e.localizedDescription.UTF8String); }
    return ps;
}
// One diagnostic snapshot through the same native vertex fetch as geometry
// capture. Refuse shaders with writes: observing them must not run their side
// effects twice. Records retain the documented 144-word geometry capture ABI.
static void vertex_dump(NvShader *vs, const p_draw *dw, const p_draw_idx *di) {
    static const char *key, *path; static int configured, captured, limit; static uint64_t frame;
    if (!configured) { key = getenv("DM_NVN_VSDUMP_KEY"); path = getenv("DM_NVN_VSDUMP");
        const char *f = getenv("DM_NVN_VSDUMP_FROM"), *n = getenv("DM_NVN_VSDUMP_MAX");
        frame = f ? strtoull(f,NULL,0) : 0; limit = n ? atoi(n) : key && !strcmp(key,"*") ? 256 : 1; configured = 1; }
    int all = key && !strcmp(key,"*");
    if (captured >= limit || !key || !path || atomic_load(&g_cur_frame) < frame || !vs || (!all && strncmp(vs->key, key, 16)) || !vs->geo_fn ||
        (dw ? dw->ind || dw->mdi_cnt : di->ind || di->mdi_cnt) || S.rast_discard) return;
    for (int i = 0; i < vs->rf.nb; i++) if (vs->rf.b[i].rw) return;
    int count = dw ? dw->count : di->count, instances = dw ? dw->instances : di->instances;
    if (count <= 0 || instances > 1 || (di && (di->itype < 0 || di->itype > 2))) return;
    unsigned number = captured++; lc_land(); rp_flush(); mdi_wait();
    uint32_t lo = dw ? (uint32_t)dw->first : UINT32_MAX, hi = dw ? lo + count - 1 : 0;
    if (di) {
        int size = 1 << di->itype;
        if (!nvn_host_readable(di->ibuf, (uint64_t)count * size)) return;
        for (int i = 0; i < count; i++) { uint32_t n = 0; memcpy(&n, (void *)(uintptr_t)(di->ibuf + (uint64_t)i * size), size);
            if (S.restart.on && n == (S.restart.index & (size == 1 ? 255u : size == 2 ? 65535u : UINT32_MAX))) continue;
            lo = MIN(lo, n); hi = MAX(hi, n); }
        if (lo == UINT32_MAX || (int64_t)lo + di->base_vertex < 0 || (int64_t)hi + di->base_vertex > UINT32_MAX) return;
        lo += di->base_vertex; hi += di->base_vertex;
    }
    uint64_t span = (uint64_t)hi - lo + 1, bytes = span * 144 * 4;
    if (!span || bytes > (64u << 20)) return;
    id<MTLRenderPipelineState> ps = geo_capture_pso(vs); if (!ps) return;
    id<MTLBuffer> output; size_t offset = arena_alloc(bytes + 512, &output);
    uint64_t gpu = output.gpuAddress + offset;
    uint64_t info[4] = {gpu + 512, 0, (uint64_t)lo | span << 32, (uint32_t)(dw ? dw->base_instance : di->base_instance) | 1ull << 32};
    nvn_geo_capture gc = {info[0], 0, {(uint32_t)lo, (uint32_t)span, (uint32_t)info[3], 1}, GEO_CAPTURE_WORDS, {0}};
    for (int k = 0; k < GEO_CAPTURE_WORDS; k++) gc.slot[k] = (int16_t)k;   // every word, in order
    memcpy((uint8_t *)output.contents + offset, &gc, sizeof gc);
    p_draw range = {.first = (int32_t)lo, .count = (int32_t)span, .instances = 1, .base_instance = (int32_t)(info[3] & UINT32_MAX)};
    vbuf_prepare(&vs->rf, &range, NULL);   // the vertices the capture reads: streams it reads past get zeros there (vbuf_need)
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new]; rd.renderTargetWidth = rd.renderTargetHeight = rd.defaultRasterSampleCount = 1;
    id<MTL4RenderCommandEncoder> en = enc_r_begin(rd); enc_r_pso(en, ps); bind_vbufs(en);
    vconv_bind(en, vs, 1, &range, NULL); args_dirty_all(); encode_args(en, vs, 0, 1); enc_r_vbuf(en, 13, gpu);
    enc_r_draw_instanced(en, MTLPrimitiveTypePoint, lo, span, 1, (uint32_t)info[3]); enc_r_end(en);
    mdi_wait();
    char filename[1200]; if (all) snprintf(filename,sizeof filename,"%s_%04u_%s.bin",path,number,vs->key); else strlcpy(filename,path,sizeof filename);
    FILE *f = fopen(filename, "wb"); if (f) { uint32_t header[4] = {lo, (uint32_t)span, all ? 4u : 144u, (uint32_t)info[3]}; fwrite(header, sizeof header, 1, f);
        if (all) for (uint64_t i=0;i<span;i++) fwrite((uint8_t *)output.contents + offset + 512 + i*144*4,1,16,f);
        else fwrite((uint8_t *)output.contents + offset + 512, 1, bytes, f); fclose(f); }
    MLOG("vertex dump %s: first %u, count %llu, %s", vs->key, lo, (unsigned long long)span, filename);
    args_dirty_all(); g_enc_pso = nil; g_enc_dss = nil; g_ed.ok = 0;
}
static void geo_draw(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0, uint32_t primitive_base);
// A draw whose capture would need more scratch than the bound, drawn as smaller draws in the hardware's order: one instance at a time, then runs of a list topology's primitives
// (points, lines, triangles, lines and triangles with adjacency), each with its own index range. The capture lays the vertices out by vertex number from the lowest index to the highest for
// every instance, and the geometry shader's output takes room per primitive: OCTOPATH TRAVELER II draws 129,032 triangles of one mesh through a cube shadow map's geometry shader in its
// first battle, which asked for more than the bound and was skipped. The index list is read once and cut into runs of at most GEO_RUN_PRIMS primitives whose index range takes at most
// GEO_RUN_SPAN of capture; the runs skip the GPU wait their draw would take (the outer draw waited already, and nothing in between writes index or vertex data). Halving the list at every
// level instead, each part reading its indices again and waiting for the GPU, made that one frame take 805 ms. Strips, fans and primitive restart cannot be cut that way.
// Each child retains its draw-wide primitive base; only the guest PrimitiveID adds it, while capture and output addresses remain local to that child.
// Returns 0 when the draw cannot be split (the caller reports it)
enum { GEO_RUN_PRIMS = 4096 }; static const uint64_t GEO_RUN_SPAN = 64ull << 20;
static int g_geo_split_depth;   // > 0 while geo_split draws its runs: they skip the GPU wait
static int geo_sync_cpu(int *drained) {
    if (*drained) return 1;
    mdi_wait();
    if (g_mdi_ev.signaledValue < g_mdi_v) return 0;
    memset(g_vconv, 0, sizeof g_vconv); *drained = 1;
    return 1;
}
static int geo_split(NvProg *pr, const p_draw *dw, const p_draw_idx *di, int count, int instances, uint32_t first_instance, uint32_t primitive_base) {
    // geo_prep can pass a record in g_geo_run. Each recursive draw reuses that
    // job, so keep the original range independent of the child draw's record.
    const p_draw original_dw = dw ? *dw : (p_draw){0};
    const p_draw_idx original_di = di ? *di : (p_draw_idx){0};
    dw = dw ? &original_dw : NULL; di = di ? &original_di : NULL;
    int prim = dw ? dw->prim : di->prim, per = prim == 0 ? 1 : prim == 1 ? 2 : prim == 4 ? 3 : prim == 10 ? 4 : prim == 12 ? 6 : 0;
    if (!per || (di && S.restart.on) || count / per < 1) return 0;
    static uint64_t nsplit; if (nsplit++ < 4) MLOG("geometry draw above the scratch bound: drawn in parts (vertex %s, geometry %s, %d primitives, %d instances)",
                                                   pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[2] ? pr->sh[2]->key : "-", count / per, instances);
    p_draw a = dw ? *dw : (p_draw){0}; p_draw_idx b = di ? *di : (p_draw_idx){0};
    g_geo_split_depth++;
    if (instances > 1) {
        for (int i = 0; i < instances; i++) {
            if (dw) { a.instances = 1; a.base_instance = (int32_t)(first_instance + i); geo_draw(pr, &a, NULL, primitive_base); }
            else { b.instances = 1; b.base_instance = (int32_t)(first_instance + i); geo_draw(pr, NULL, &b, primitive_base); }
        }
        g_geo_split_depth--; return 1;
    }
    int np = count / per, isz = di ? 1 << di->itype : 0;
    for (int p0 = 0; p0 < np;) {
        int p1 = p0; uint32_t lo = UINT32_MAX, hi = 0;
        while (p1 < np && p1 - p0 < GEO_RUN_PRIMS) {   // extend the run while its index range fits
            uint32_t l = lo, h = hi;
            for (int k = 0; k < per; k++) {
                uint32_t v = (uint32_t)(p1 * per + k);
                if (di) { v = 0; memcpy(&v, (const uint8_t *)(uintptr_t)di->ibuf + (size_t)(p1 * per + k) * isz, isz); }
                l = MIN(l, v); h = MAX(h, v);
            }
            if (p1 > p0 && ((uint64_t)h - l + 1) * 144 * 4 > GEO_RUN_SPAN) break;
            lo = l; hi = h; p1++;
        }
        if (p0 == 0 && p1 == np) {   // one run is the whole draw: its own outputs are what exceed the bound, so halve it
            p1 = np / 2; if (!p1) { g_geo_split_depth--; return 0; }
        }
        if (dw) { a.first = dw->first + p0 * per; a.count = (p1 - p0) * per; geo_draw(pr, &a, NULL, primitive_base + (uint32_t)p0); }
        else { b.ibuf = di->ibuf + ((uint64_t)(p0 * per) << di->itype); b.count = (p1 - p0) * per; geo_draw(pr, NULL, &b, primitive_base + (uint32_t)p0); }
        p0 = p1;
    }
    g_geo_split_depth--;
    return 1;
}

static void geo_draw_(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0, uint32_t primitive_base);
static int g_geo_split_compute = -1, g_geo_full_colors = -1, g_geo_pack_legacy = -1;
static int geo_pack_legacy(void) { if (g_geo_pack_legacy < 0) { const char *e = getenv("DM_NVN_GEOMETRY_PACK"); g_geo_pack_legacy = e && !strcmp(e, "legacy"); } return g_geo_pack_legacy; }
static uint64_t g_geo_pack_small, g_geo_pack_large;
static uint64_t g_geo_color_omits;
// Keep CPU primitive maps and unique capture IDs, never vertex results or GPU
// output. Every reuse compares the original index bytes after the existing
// hazard/wait checks, so CPU edits, GPU writes and storage reuse all invalidate
// the result without depending on write-protection or a hash collision.
typedef struct { uint64_t address; int32_t count, type, prim, restart; uint32_t restart_index, pad; } geo_ix_key;
typedef struct geo_ix_ent {
    geo_ix_key key; uint32_t lo, hi;
    size_t nu, words, unique, raw_n, bytes;
    struct geo_ix_ent *prev, *next;
    uint8_t data[];   // original bytes, normalized primitive map, unique IDs
} geo_ix_ent;
static nvn_hmap g_geo_ix = {.klen = sizeof(geo_ix_key)};
static geo_ix_ent *g_geo_ix_first, *g_geo_ix_last;
static size_t g_geo_ix_bytes;
static uint64_t g_geo_ix_hits, g_geo_ix_misses;
static int g_geo_ix_on = -1;
enum { GEO_IX_BYTES = 32 << 20, GEO_IX_ENTRIES = 4096 };
static int geo_ix_on(void) { if (g_geo_ix_on < 0) { const char *e=getenv("DM_NVN_GEOMETRY_INDEX_CACHE");g_geo_ix_on=!e||strcmp(e,"0"); }return g_geo_ix_on; }
static void geo_ix_unlink(geo_ix_ent *e) {
    if(e->prev)e->prev->next=e->next;else g_geo_ix_first=e->next;
    if(e->next)e->next->prev=e->prev;else g_geo_ix_last=e->prev;
}
static void geo_ix_front(geo_ix_ent *e) {
    e->prev=NULL;e->next=g_geo_ix_first;if(e->next)e->next->prev=e;else g_geo_ix_last=e;g_geo_ix_first=e;
}
static void geo_ix_drop(geo_ix_ent *e) {geo_ix_unlink(e);nvn_hm_del(&g_geo_ix,&e->key);g_geo_ix_bytes-=e->bytes;free(e);}
static geo_ix_ent *geo_ix_get(const geo_ix_key *key,const void *raw,size_t raw_n) {
    if(!geo_ix_on())return NULL;
    geo_ix_ent *e=nvn_hm_get(&g_geo_ix,key);
    if(e&&e->raw_n==raw_n&&!memcmp(e->data,raw,raw_n)){geo_ix_unlink(e);geo_ix_front(e);g_geo_ix_hits++;return e;}
    if(e)geo_ix_drop(e);g_geo_ix_misses++;return NULL;
}
static void geo_ix_put(const geo_ix_key *key,const void *raw,size_t raw_n,uint32_t lo,uint32_t hi,size_t nu,const void *map,size_t words,const void *unique,size_t n_unique) {
    if(!geo_ix_on())return;
    size_t bytes=sizeof(geo_ix_ent)+raw_n+(words+n_unique)*4;
    if(bytes>GEO_IX_BYTES)return;
    geo_ix_ent *old=nvn_hm_get(&g_geo_ix,key);if(old)geo_ix_drop(old);
    while(g_geo_ix_last&&(g_geo_ix_bytes+bytes>GEO_IX_BYTES||g_geo_ix.n>=GEO_IX_ENTRIES))geo_ix_drop(g_geo_ix_last);
    geo_ix_ent *e=malloc(bytes);if(!e)return;
    *e=(geo_ix_ent){.key=*key,.lo=lo,.hi=hi,.nu=nu,.words=words,.unique=n_unique,.raw_n=raw_n,.bytes=bytes};
    memcpy(e->data,raw,raw_n);memcpy(e->data+raw_n,map,words*4);memcpy(e->data+raw_n+words*4,unique,n_unique*4);
    nvn_hm_put(&g_geo_ix,key,e);geo_ix_front(e);g_geo_ix_bytes+=bytes;
}
static int geo_color_omit_(void);
static int geo_color_omit(void) {
    if (g_geo_full_colors < 0) { const char *e = getenv("DM_NVN_GEOMETRY_COLOR"); g_geo_full_colors = e && !strcmp(e, "full"); }
    if (g_geo_full_colors || !R.has_rt || !R.rt.depth || R.clear_d || R.clear_s) return 0;
    // The answer of the last call with the same targets (each tex_t with its serial, so another texture at the same address does not match), views, clears and color
    // mask: resolving every attachment's Metal texture a draw was about 15 % of a mesh draw's time in Donkey Kong Bananza
    // A byte key: the target record's bytes as they lie (padding included: unequal padding of an equal record only costs a recompute), then the serials, mask and clears
    enum { KEY_RT = sizeof(p_set_rt), KEY_N = KEY_RT + 10 * 8 + 8 };
    uint8_t key[KEY_N]; uint64_t w[10];
    for (int i = 0; i < 8; i++) w[i] = R.rt.color[i] ? R.rt.color[i]->serial : 0;
    w[8] = R.rt.depth->serial; w[9] = (uint64_t)S.chmask.mask;
    memcpy(key, &R.rt, KEY_RT); memcpy(key + KEY_RT, w, sizeof w); memcpy(key + KEY_RT + sizeof w, R.clear_c, 8);
    static uint8_t last[KEY_N]; static int last_r = -1;
    if (last_r >= 0 && !memcmp(key, last, KEY_N)) return last_r;
    memcpy(last, key, KEY_N); last_r = geo_color_omit_(); return last_r;
}
static int geo_color_omit_(void) {
    // Removing an inactive color must not change raster bounds, sample count or
    // layer selection. Keep the general case on the original attachment path.
    id<MTLTexture> d = tex_mtl(R.rt.depth);
    if (!d || d.textureType != MTLTextureType2D || !mtl_fmt_depth(d.pixelFormat) || d.pixelFormat == MTLPixelFormatStencil8) return 0;
    NSUInteger dl = R.rt.has_dview && R.rt.dview.set_levels ? R.rt.dview.base_level : 0;
    NSUInteger w = mip_extent(d.width, dl), h = mip_extent(d.height, dl);
    int n = 0;
    for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) {
        if (R.clear_c[i] || ((S.chmask.mask >> (4 * i)) & 15)) return 0;
        id<MTLTexture> c = tex_mtl(R.rt.color[i]);
        NSUInteger l = R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0;
        if (!c || c.textureType != MTLTextureType2D || c.sampleCount != d.sampleCount || mip_extent(c.width, l) != w || mip_extent(c.height, l) != h) return 0;
        n++;
    }
    return n != 0;
}
// A run of geometry draws separated only by state changes: all vertex captures
// in one render pass, all GS and compaction dispatches in one compute pass,
// then the raster draws in the open render pass as replay reaches each one.
typedef struct { uint64_t lo, hi; int write; } geo_range;
typedef struct { geo_range r[64]; uint64_t writes; int n, unknown; } geo_access;
static void geo_access_add(geo_access *a, uint64_t lo, uint64_t size, int write) {
    if (!lo) return;
    if (!size || lo > UINT64_MAX-size || nvn_sparse_alias(lo, lo + size)) { a->unknown=1; return; }
    for (int i=0;i<a->n;i++) if (a->r[i].lo==lo && a->r[i].hi==lo+size) {
        a->r[i].write|=write; if(write)a->writes|=1ull<<i; return;
    }
    if (a->n==64) { a->unknown=1; return; }
    a->r[a->n]=(geo_range){lo,lo+size,write};
    if(write)a->writes|=1ull<<a->n; a->n++;
}
// A run moves every capture/GS ahead of every raster draw. Keep that ordering
// only when external buffer accesses cannot communicate between its jobs.
static geo_access geo_access_of(NvProg *pr, NvShader *gs) {
    geo_access a={0};
    for (int i=0;i<16;i++) if (S.vbuf[i].size) geo_access_add(&a,S.vbuf[i].address,S.vbuf[i].size,0);   // an empty stream reads no guest memory (an unknown range otherwise)
    for (int st=0;st<3;st++) {
        NvShader *sh=st==2?gs:pr->sh[st]; if (!sh) continue;
        if (sh->rf.drv_read[0]) a.unknown=1;
        for (int i=0;i<sh->rf.nb;i++) {
            const nvn_rbind *b=&sh->rf.b[i];
            if ((b->kind==NVN_RB_TEX || b->kind==NVN_RB_TEXU || b->kind==NVN_RB_TEXD || b->kind==NVN_RB_TEXA) && b->dtt==MTLTextureTypeTextureBuffer) a.unknown=1;
            if (b->kind==NVN_RB_CBUF || b->kind==NVN_RB_UBOP || b->kind==NVN_RB_SBUFP || b->kind==NVN_RB_TEXU || b->kind==NVN_RB_SAMPU) {
                if (b->reg>=16) { a.unknown=1; continue; }
                const p_bind_buf *u=b->kind==NVN_RB_SBUFP?&S.ssbo[st][b->reg]:&S.ubo[st][b->reg];
                int write=b->kind==NVN_RB_SBUFP && b->rw && !(sh->sb_ro>>b->reg&1);
                geo_access_add(&a,u->addr,b->kind==NVN_RB_CBUF?cb_len(sh,b->reg,u->size):u->size,write);
            } else if (b->kind==NVN_RB_SBUFU || b->kind==NVN_RB_IMG || b->kind==NVN_RB_IMGU || b->kind==NVN_RB_TBUF || b->kind==NVN_RB_TBUFU) a.unknown=1;
        }
    }
    return a;
}
// A passthrough geometry shader cannot change the vertices it passes on: with input of points, lines or triangles, its lean
// kernel (dmsh_geometry.lean) reads only the capture words it decides with, writes the viewport masks and its storage, and the
// program's own vertex and fragment stages draw the kept primitives. Instanced or multi-invocation draws, vertex shaders with
// storage writes (they would run twice) and transform feedback keep the full kernel and its reader. So does a passthrough shader that writes the layer (its reader has it, the
// vertex shader does not): Mario Tennis Fever's atmosphere table (d9c3a99828b4f39f, one 8x8 plane of a 64-plane 3D texture per draw) and its shadow map went to layer 0, and the
// other planes stayed zero: a blue wash over the whole scene. DM_NVN_GEO_LEAN=0 turns it off.
static NvShader *geo_lean_pick(NvProg *pr, NvShader *gs, int topology, int instances) {
    static int lean_on = -1; if (lean_on < 0) { const char *e = getenv("DM_NVN_GEO_LEAN"); lean_on = !(e && !strcmp(e, "0")); }
    NvShader *vs = pr->sh[0], *base = pr->sh[2];
    if (!lean_on || instances > 1 || (gs && gs->geo_invocations != 1) || base->geo_invocations != 1 || !vs || vs->discard_fn || S.tf_on || !pr->sh[1] || !(topology == 0 || topology == 1 || topology == 3)) return NULL;
    if (!(base->geo_bg_tried & 0x80)) { base->geo_bg_tried |= 0x80; base->geo_bgv_v = shader_get([[[NSString stringWithUTF8String:base->key] substringToIndex:16] stringByAppendingString:@"_bgv"], 0); }
    if (base->geo_bgv_v && base->geo_bgv_v->layer_out) return NULL;
    if (!(base->geo_lean_tried >> topology & 1)) {
        base->geo_lean_tried |= 1 << topology;
        NSString *key = [[NSString stringWithUTF8String:base->key] substringToIndex:16];
        base->geo_lean_v[topology] = shader_get([key stringByAppendingFormat:@"_bl%d", topology], 2);
    }
    NvShader *lean = base->geo_lean_v[topology];
    if (!lean || !lean->geo_slot || lean->geo_vertices != (uint32_t)nvn_geo_width(topology) || lean->geo_invocations != 1) return NULL;
    if (!lean->geo_pso && !lean->geo_pso_failed) { NSError *e = nil; lean->geo_pso = [g_dev newComputePipelineStateWithFunction:lean->fn error:&e];
        if (!lean->geo_pso || lean->geo_pso.maxTotalThreadsPerThreadgroup < 32) { lean->geo_pso = nil; lean->geo_pso_failed = 1; MLOG("[RED] lean geometry compute %s: %s; the full kernel draws", lean->key, e ? e.localizedDescription.UTF8String : "no 32-thread group"); } }
    if (!lean->geo_pso) return NULL;
    if (!g_geo_lean_k[2]) {
        static int failed;
        if (failed) return NULL;
        NSError *error = nil;
        id<MTLLibrary> lib = host_lib("geo_pack", &error);
        const char *names[] = {"geo_lean_cull", "geo_lean_index", "geo_lean_tri"};
        for (int i = 0; i < 3 && lib; i++) g_geo_lean_k[i] = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(names[i])] error:&error];
        if (!g_geo_lean_k[0] || !g_geo_lean_k[1] || !g_geo_lean_k[2]) {
            failed = 1; g_geo_lean_k[0] = g_geo_lean_k[1] = g_geo_lean_k[2] = nil;
            MLOG("[RED] lean geometry index kernels: %s", error.localizedDescription.UTF8String); return NULL;
        }
    }
    return lean;
}
// Passthrough geometry as mesh draws: a lean triangle-list draw whose pair has a mesh library, <vertex key>_<kernel key>_m3 (dm/gfx/tools/dmsh_mesh.py, built for the
// pairs in the profile's geo_mesh.tsv), is one mesh draw inside the render pass. A threadgroup per 32 triangles runs the pulling vertex shader per index and the kernel body
// per triangle, and culls what the kernel culls: no capture, no compute, no index compaction, no pass break. Its pipeline is built in the background; until it is ready, and
// for every other pair, the compute path draws. DM_NVN_GEO_MESH=0 turns it off
static int geo_mesh_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_NVN_GEO_MESH"); on = !(e && !strcmp(e, "0")); if (!on) MLOG("DM_NVN_GEO_MESH=0: passthrough geometry draws through the compute path"); }
    return on;
}
enum { GEO_MESH_PRIMS = 32, GEO_MESH_DRAW_SLOT = 12, GEO_MESH_GS_SLOT = 13 };   // = dmsh_mesh.py PRIMS, DRAW_SLOT, GS_SLOT
typedef struct { uint64_t ib; uint32_t itype, first, count; int32_t base_vertex; uint32_t base_instance, pad; } geo_mesh_args;   // dmsh_mesh.py dm_mesh_draw
static struct { int on, miss; uint32_t prims; __unsafe_unretained id<MTLFunction> fn; __unsafe_unretained NvShader *lean; __unsafe_unretained id<MTLRenderPipelineState> ps; geo_mesh_args a; } g_geo_mesh;
// Triangles a threadgroup of this mesh entry takes: GEO_MESH_PRIMS, or the count its name gives (dm_mesh_p16: dmsh_mesh.py prims_for, when the vertices of 32 exceed the mesh output limit)
static uint32_t geo_mesh_prims(id<MTLFunction> f) {
    static __unsafe_unretained id<MTLFunction> last; static uint32_t prims;
    if (f != last) { NSRange r = [f.name rangeOfString:@"_p" options:NSBackwardsSearch]; prims = r.location != NSNotFound ? (uint32_t)[f.name substringFromIndex:NSMaxRange(r)].intValue : 0;
        if (!prims || prims > GEO_MESH_PRIMS) prims = GEO_MESH_PRIMS; last = f; }
    return prims;
}
static struct { NvShader *vs, *lean; id<MTLFunction> f; } g_geo_mesh_last;
static NSMutableDictionary *g_geo_mesh_runfn;   // a mesh entry → its run twin (dm_mesh_run..., dmsh_mesh.py), or NSNull: none in the library, or its pipeline failed
// The mesh entry for this draw's program, or nil (the compute path draws it)
static id<MTLFunction> geo_mesh_fn(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    if (!geo_mesh_on()) return nil;
    int prim = dw ? dw->prim : di->prim, instances = dw ? dw->instances : di->instances, count = dw ? dw->count : di->count;
    NvShader *vs = pr->sh[0];
    if (prim != 4 || count < 3 || instances > 1 || (dw ? dw->ind : di->ind) || (di && (S.restart.on || di->itype < 0 || di->itype > 2)) || S.rast_discard || S.tf_on ||
        !vs || !vs->has_pull || vs->layer_out || vs->discard_fn || !vs->fn || pr->sh[3] || pr->sh[4]) return nil;
    NvShader *lean = geo_lean_pick(pr, NULL, 3, instances); if (!lean) return nil;
    // the last pair's answer, without a key string and a dictionary lookup a draw (strong references: a pair cannot be freed and another take its addresses)
    if (vs == g_geo_mesh_last.vs && lean == g_geo_mesh_last.lean) return g_geo_mesh_last.f;
    if (!vs->geo_mesh) vs->geo_mesh = [NSMutableDictionary dictionary];
    NSString *k = @(lean->key); id f = vs->geo_mesh[k];
    if (!f) {
        NSString *path = [NSString stringWithFormat:@"%@/%.16s_%.16s_m3.metallib", g_msl_dir, vs->key, lean->key];
        id<MTLLibrary> lib = lib_from_file(path, NULL);
        NSString *want = [vs->fn.name isEqualToString:@"main0_z"] ? @"dm_mesh_z" : @"dm_mesh";
        f = lib ? [lib newFunctionWithName:want] : nil;
        if (lib && !f) for (NSString *nm in lib.functionNames) if ([nm hasPrefix:[want stringByAppendingString:@"_p"]]) { f = [lib newFunctionWithName:nm]; break; }   // fewer triangles a group (dm_mesh_p16)
        if (f) { if (!g_geo_mesh_runfn) g_geo_mesh_runfn = [NSMutableDictionary dictionary];   // its run twin: dm_mesh_z_p16 → dm_mesh_run_z_p16
            g_geo_mesh_runfn[[NSValue valueWithPointer:(__bridge const void *)f]] = [lib newFunctionWithName:[@"dm_mesh_run" stringByAppendingString:[((id<MTLFunction>)f).name substringFromIndex:7]]] ?: NSNull.null; }
        static int n; if (f || n++ < 16) MLOG("geometry mesh: vertex %s kernel %s %s", vs->key, lean->key, f ? "has a mesh library" : "has no mesh library (geo_mesh.tsv): the compute path draws it");
        vs->geo_mesh[k] = f = f ?: NSNull.null;
    }
    g_geo_mesh_last.vs = vs; g_geo_mesh_last.lean = lean; g_geo_mesh_last.f = f == NSNull.null ? nil : f;
    return g_geo_mesh_last.f;
}
static nvn_hmap g_pso_mesh = {.klen = sizeof(pso_key)}; static os_unfair_lock g_pso_mesh_lock = OS_UNFAIR_LOCK_INIT;
static NSObject *g_pso_mesh_building;
// The draw's mesh pipeline for mesh function fn (the draw's own, or its run twin), or nil while it is built in the background (or when it failed)
static id<MTLRenderPipelineState> geo_mesh_pso_for(NvProg *pr, id<MTLFunction> fn, int run) {
    NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    pso_key k; pso_key_fill(&k, vs, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, 0);
    memset(k.attr, 0, sizeof k.attr); memset(k.attr_hi, 0, sizeof k.attr_hi); memset(k.stream, 0, sizeof k.stream);   // the pipeline pulls its streams: no vertex layout in it
    k.fsfn = (__bridge void *)fn;
    os_unfair_lock_lock(&g_pso_mesh_lock); id ps = hm_get_id(&g_pso_mesh, &k);
    if (!ps) { if (!g_pso_mesh_building) g_pso_mesh_building = [NSObject new]; hm_put_id(&g_pso_mesh, &k, g_pso_mesh_building); }
    os_unfair_lock_unlock(&g_pso_mesh_lock);
    if (!ps) {
        MTLRenderPipelineDescriptor *d = pso_desc(&k, vs, fs, nil, 0);
        MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
        m.meshFunction = fn; m.fragmentFunction = d.fragmentFunction;
        m.rasterSampleCount = d.rasterSampleCount; m.alphaToCoverageEnabled = d.alphaToCoverageEnabled;
        for (int i = 0; i < 8; i++) m.colorAttachments[i] = d.colorAttachments[i];
        m.depthAttachmentPixelFormat = d.depthAttachmentPixelFormat; m.stencilAttachmentPixelFormat = d.stencilAttachmentPixelFormat;
        const char *vk = vs->key, *fk = fs ? fs->key : "-", *gk = g_geo_mesh.lean->key; pso_key kk = k;
        dispatch_async(pp_bg_queue(), ^{
            NSError *e = nil; uint64_t t0 = bp_now();
            id<MTLRenderPipelineState> p = [g_dev newRenderPipelineStateWithMeshDescriptor:m options:MTLPipelineOptionNone reflection:nil error:&e];
            MLOG("%sgeometry mesh pipeline: vertex %s fragment %s kernel %s, %.1f ms in the background%s%s", p ? "" : "[RED] ", vk, fk, gk, (bp_now() - t0) / 1e6, p ? "" : ": ", p ? "" : e.localizedDescription.UTF8String);
            os_unfair_lock_lock(&g_pso_mesh_lock); hm_put_id(&g_pso_mesh, &kk, p ?: NSNull.null); os_unfair_lock_unlock(&g_pso_mesh_lock);
        });
        return nil;
    }
    // A failed pipeline sends the pair back to the compute path for good: with its mesh function still set, every draw of the pair left the geometry runs (geo_run_walk) and drew
    // alone (a mesh output over Metal's limit took Donkey Kong Bananza's translation from about 10 to 22-31 ms a frame)
    if (ps == NSNull.null && run) g_geo_mesh_runfn[[NSValue valueWithPointer:(__bridge const void *)g_geo_mesh.fn]] = NSNull.null;   // the draws of the pair stay single
    else if (ps == NSNull.null && g_geo_mesh.lean) { NvShader *vs = pr->sh[0]; vs->geo_mesh[@(g_geo_mesh.lean->key)] = NSNull.null; g_geo_mesh_last.vs = nil; }
    return ps == NSNull.null || ps == g_pso_mesh_building ? nil : ps;
}
static id<MTLRenderPipelineState> geo_mesh_pso(NvProg *pr) { return g_geo_mesh.ps = geo_mesh_pso_for(pr, g_geo_mesh.fn, 0); }
// Mesh draws gathered into one dispatch. A mesh draw costs the M1 about 2.7 us however few triangles it has: Donkey Kong Bananza's depth prepass draws 420-550 of them a frame
// (vertex f9c807930abd1234, culling kernel 5d537f7630ec3671), each a culling geometry shader over 15-350 triangles, 1.5-1.7 ms of GPU a frame where one draw of the same
// triangles takes 0.5-0.6. A draw that fits waits instead (geo_mesh_take): its record keeps its arguments and the addresses bound at the vertex table's slots for it (g_vt_addr,
// which the mesh stage shares), and the run is drawn as one dispatch of the run twin (dmsh_mesh.py dm_mesh_run...), whose threadgroups find their draw in the table, before:
// any operation but bindings, a program bind and draws (geo_mesh_keeps, from mtl_exec), a draw that is not a mesh draw of the same pair (draw_, geo_mesh_draw), and the end of
// the pass (rp_end). Bindings only change S and dirty flags, so nothing else reaches the encoder in between. The draws of a run then run in one dispatch, whose threadgroups the
// GPU may rasterize in any order, so a run takes only draws whose result does not depend on their order: no colour writes, no stencil writes, depth written only with a test that
// keeps the nearest (or the farthest), and a fragment shader with no resources (the fragment table is not in the record). Occlusion query counting may be on (it stays on for the rest
// of the game after the first samples-passed reset): a report ends the run (geo_mesh_keeps) and a pass keeps one slot, so a query's interval holds whole runs, and whether any of its
// samples passed is the same in any order (the nearest fragment of each pixel passes whatever the order), which is what occlusion tests and conditional rendering ask. Only the exact
// count can differ by fragments another order rejects. Donkey Kong Bananza resets that counter once and draws its depth prepass with counting on: refused, no draw ever joined a run.
// DM_NVN_GEO_MESH_RUN=0 turns it off
typedef struct { uint64_t slot[31]; geo_mesh_args d; uint32_t group_end, n; } geo_mesh_rec;   // = dmsh_mesh.py dm_mesh_rec (RUN_SLOTS)
enum { GEO_MESH_RUN = 1024 };
static struct { int n; uint32_t groups, prims; __unsafe_unretained id<MTL4RenderCommandEncoder> en; __unsafe_unretained id<MTLRenderPipelineState> ps, run_ps;
                __unsafe_unretained NvShader *vs, *fs, *lean; __unsafe_unretained id<MTLFunction> fn; geo_mesh_rec r[GEO_MESH_RUN]; } g_mrun;
static uint64_t g_mrun_draws, g_mrun_runs, g_mrun_ends[4];   // draws drawn in runs, runs; runs ended by an operation, another draw, the pass end, a full table
static int geo_mesh_run_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_NVN_GEO_MESH_RUN"); on = !(e && !strcmp(e, "0")); if (!on) MLOG("DM_NVN_GEO_MESH_RUN=0: every mesh draw is a dispatch of its own"); }
    return on;
}
// Draw the waiting run (why: index into g_mrun_ends)
static void geo_mesh_flush_(int why) {
    int n = g_mrun.n; if (!n) return;
    g_mrun.n = 0; g_mrun_ends[why]++;
    id<MTL4RenderCommandEncoder> en = g_mrun.en;
    if (!en || (__bridge void *)en != (__bridge void *)g_enc) { static int once; if (!once++) MLOG("[RED] geometry mesh run: its pass ended before the run was drawn; %d draws lost", n); return; }
    g_mrun.r[0].n = (uint32_t)n;
    size_t bytes = sizeof(geo_mesh_rec) * (size_t)n; id<MTLBuffer> b; size_t o = arena_alloc(bytes, &b);
    memcpy((uint8_t *)b.contents + o, g_mrun.r, bytes);
    [en setRenderPipelineState:g_mrun.run_ps];
    [en setArgumentTable:g_vt atStages:MTLRenderStageMesh];
    enc_r_vbuf(en, GEO_MESH_DRAW_SLOT, b.gpuAddress + o);
    [en drawMeshThreadgroups:MTLSizeMake(g_mrun.groups, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(3 * g_mrun.prims, 1, 1)];
    [en setRenderPipelineState:g_mrun.ps];   // as the draws left it
    g_mrun_draws += (uint64_t)n; g_mrun_runs++;
    static int once; if (!once++) MLOG("geometry mesh run: first run, %d draws of vertex %s kernel %s in one dispatch of %u threadgroups", n, g_mrun.vs->key, g_mrun.lean->key, g_mrun.groups);
}
static void geo_mesh_flush(void) { geo_mesh_flush_(0); }
static void geo_mesh_flush_draw(void) { geo_mesh_flush_(1); }
static void geo_mesh_flush_pass(void) { geo_mesh_flush_(2); }
// The operations a waiting run can wait across (mtl_exec): they change bindings, the program or nothing, and draws decide for themselves
static int geo_mesh_keeps(uint32_t op) {
    switch (op) {
        case OP_BIND_PROGRAM: case OP_BIND_UBO: case OP_BIND_UBO_SMALL: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP:
        case OP_BIND_VBUF: case OP_BIND_VBUF1: case OP_BIND_VSTREAM: case OP_BIND_VATTR: case OP_DRAW: case OP_DRAW_IDX: case OP_BEGIN: case OP_JUMP: case OP_CALL: return 1;
        default: return 0;
    }
}
// Whether the draw's result is the same in any order among the draws of a run (see above); the first refusal of each kind is logged
static int geo_mesh_run_fits(NvProg *pr) {
    static const char *const why[] = {"its fragment shader has resources", "it writes colour", "it writes stencil", "its depth test does not keep the nearest or the farthest"};
    static uint8_t logged;
    NvShader *fs = pr->sh[1]; int r = -1;
    uint32_t attached = 0; for (int i = 0; i < 8; i++) if (g_pass.cfmt[i]) attached |= 1u << i;
    if (fs && (fs->rf.nb || fs->rf.bindless || fs->rf.dyn)) r = 0;
    else if (draw_color_writes(pr) & attached) r = 1;
    else if (g_pass.hs && S.ds.st && (S.st_wmask[0] || S.st_wmask[1])) r = 2;
    else if (g_pass.hd && S.ds.dw && !(S.ds.dt && (S.ds.df == 1 || S.ds.df == 2 || S.ds.df == 4 || S.ds.df == 5 || S.ds.df == 7))) r = 3;   // NVN NEVER, LESS, LEQUAL, GREATER, GEQUAL
    if (r < 0) return 1;
    if (!(logged >> r & 1)) { logged |= 1 << r; MLOG("geometry mesh run: a draw of vertex %s draws alone: %s", pr->sh[0] ? pr->sh[0]->key : "-", why[r]); }
    return 0;
}
// The draw about to be dispatched (geo_mesh_encode, with its arguments bound): 1 when it joined the run instead
static int geo_mesh_take(id<MTL4RenderCommandEncoder> en) {
    if (!geo_mesh_run_on() || g_defer_replaying) return 0;
    NvProg *pr = g_geo_program;
    if (g_mrun.n && ((__bridge void *)en != (__bridge void *)g_mrun.en || g_geo_mesh.ps != g_mrun.ps || g_geo_mesh.fn != g_mrun.fn || pr->sh[0] != g_mrun.vs || pr->sh[1] != g_mrun.fs)) {
        static int once; if (!once++) MLOG("[RED] geometry mesh run: a draw reached the encoder changed from the run's (first case pass %llu); the run is drawn first", g_stat_pass);
        geo_mesh_flush_draw();
    }
    if (g_mrun.n == GEO_MESH_RUN) geo_mesh_flush_(3);
    if (!g_mrun.n) {
        NSValue *k = [NSValue valueWithPointer:(__bridge const void *)g_geo_mesh.fn]; id f = g_geo_mesh_runfn[k];
        if (!f || f == NSNull.null || !geo_mesh_run_fits(pr)) return 0;   // no twin (a library from before runs, or a pair whose vertex resources are not all constant references)
        id<MTLRenderPipelineState> rps = geo_mesh_pso_for(pr, f, 1); if (!rps) return 0;   // built in the background: single draws until it is ready
        g_mrun.en = en; g_mrun.ps = g_geo_mesh.ps; g_mrun.run_ps = rps; g_mrun.fn = g_geo_mesh.fn; g_mrun.vs = pr->sh[0]; g_mrun.fs = pr->sh[1]; g_mrun.lean = g_geo_mesh.lean;
        g_mrun.prims = g_geo_mesh.prims; g_mrun.groups = 0;
    }
    geo_mesh_rec *r = &g_mrun.r[g_mrun.n++];
    for (int i = 0; i < 31; i++) r->slot[i] = g_vt_valid >> i & 1 ? g_vt_addr[i] : 0;
    r->d = g_geo_mesh.a; r->n = 0;
    uint32_t tris = g_geo_mesh.a.count / 3; g_mrun.groups += (tris + g_mrun.prims - 1) / g_mrun.prims; r->group_end = g_mrun.groups;
    return 1;
}
// The draw itself (draw() calls it with its pipeline, the vertex and fragment arguments and the pulling layout bound): the kernel's arguments, with vb_info
// m0 = (0, 1 instance) and m1.x = the draw's triangles, and the draw's index source
static void geo_mesh_encode(id<MTLRenderCommandEncoder> en_) {
    id<MTL4RenderCommandEncoder> en = (id<MTL4RenderCommandEncoder>)en_;
    [en setArgumentTable:g_vt atStages:MTLRenderStageMesh];
    uint32_t tris = g_geo_mesh.a.count / 3;
    id<MTLBuffer> b; size_t o = arena_alloc(64, &b); uint8_t *p = (uint8_t *)b.contents + o;
    uint32_t info[8] = {0, 1, 0, 0, tris, 0, 0, 0}; memcpy(p, info, 32); memcpy(p + 32, &g_geo_mesh.a, sizeof g_geo_mesh.a);
    enc_r_vbuf(en, GEO_MESH_DRAW_SLOT, b.gpuAddress + o + 32);
    uint64_t save[5]; memcpy(save, g_geo_bind, sizeof save); memset(g_geo_bind, 0, sizeof g_geo_bind); g_geo_bind[3] = b.gpuAddress + o;
    g_args_dirty[2] = 1; g_args_mesh = GEO_MESH_GS_SLOT; encode_args(en, g_geo_mesh.lean, 2, 0); g_args_mesh = 0; g_args_dirty[2] = 1;
    memcpy(g_geo_bind, save, sizeof save);
    if (geo_mesh_take(en)) return;
    uint32_t pg = g_geo_mesh.prims;
    [en drawMeshThreadgroups:MTLSizeMake((tris + pg - 1) / pg, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(3 * pg, 1, 1)];
}
// ---------- NVN2 task and mesh shaders: nvnCommandBufferDrawMeshTasks ----------
// A program's task and mesh shaders are Metal object and mesh functions (dmsh_meshstage.py), kept in the tessellation stages' places (sh[3], sh[4]). The draw goes through draw() with its
// pixel stage alone, a program of its own as a geometry draw's raster stage is, so it takes the render pass and the pixel stage's state like any draw; mtk_pso gives its pipeline and
// mtk_encode binds the two stages and draws. The mesh stage shares the vertex stage's table, as geo_mesh_encode's does, and a task shader has the object stage's; both read the
// draw's first workgroup at buffer 29. Native mesh draws have no equivalent fallback: the demanded pipeline must finish compiling before its first draw
static struct { int on, miss; NvShader *ts, *ms; uint32_t first, count; uint64_t ind, ind_first; uint32_t mem_count, mem_first; } g_mtk;   // mem_*: an indirect draw's record as the CPU sees it when the draw is replayed (diagnostics: the GPU writes it earlier in the submission, so this is usually the previous frame's value)   // ind: an indirect draw's threadgroups, ind_first: where the GPU finds its first
static void draw_(const p_draw *dw, const p_draw_idx *di);
static uint64_t g_mtk_n, g_mtk_wait;
static nvn_hmap g_pso_mtk = {.klen = sizeof(pso_key)};   // only the submit thread creates and uses native mesh pipelines
// A mesh pipeline must have a fragment function, where a vertex one need not (validateWithDevice). A task / mesh program with no pixel shader draws depth only (Ocarina of Time's foliage shadows)
static id<MTLFunction> mtk_nop_fragment(void) {
    static id<MTLFunction> fn; static dispatch_once_t once;
    dispatch_once(&once, ^{ NSError *e = nil; id<MTLLibrary> l = [g_dev newLibraryWithSource:@"fragment void dm_mesh_nop() {}" options:nil error:&e];
        fn = [l newFunctionWithName:@"dm_mesh_nop"]; if (!fn) MLOG("[RED] the empty fragment function did not compile: %s", e.localizedDescription.UTF8String); });
    return fn;
}
// The task / mesh pipeline of a key (k: the mesh shader in the vertex place, the pixel shader, the render pass's formats and the draw's colour state; mtk_pso, and the prewarm's M rows)
// Measurement DM_NVN_MESH_PSO=<bits> (0 by default): 1 tells the compiler the mesh / object threadgroup sizes are multiples of the execution width (they are: the translated stages run one
// 32-thread warp; the optimizer may then drop the general barrier / partial-group paths), 2 caps the mesh grid an object threadgroup may launch at the wrapper's 1,024
// (set_threadgroups_per_grid(min(n, 1024)); "may improve scheduling"), 4 draws every task / mesh draw with the empty fragment function and no colour write (the picture is wrong: the
// mesh / object stages and the tiling run, the fragment work does not), for telling the two apart on a device without a stage profile. DM_NVN_MESH_PSO_AB=1 (with DM_AB) applies the bits
// in the odd phases only: the active mode is part of the pipeline key (attr_hi[15], which a mesh key does not use), so both pipelines of a draw are cached
static int mesh_pso_mode(void) {
    static int bits = -1, ab;
    if (bits < 0) { const char *e = getenv("DM_NVN_MESH_PSO"); bits = e ? atoi(e) & 7 : 0; ab = bits ? sw_mode("DM_NVN_MESH_PSO_AB", 0) : 0;
        if (bits) MLOG("DM_NVN_MESH_PSO=%d: task / mesh pipelines%s%s%s%s", bits, bits & 1 ? " threadgroup size a multiple of the execution width" : "", bits & 2 ? ", mesh grid capped at 1024" : "",
                       bits & 4 ? ", empty fragment function and no colour write (diagnostic; the picture is wrong)" : "", ab ? " (in the DM_AB phases the switch is on)" : ""); }
    return !bits ? 0 : !ab ? bits : sw_on(ab) ? bits : 0;
}
static inline int mtk_key_mode(const pso_key *k) { return ((const uint8_t *)&k->attr_hi[15])[0]; }
static inline void mtk_key_set_mode(pso_key *k, int mode) { ((uint8_t *)&k->attr_hi[15])[0] = (uint8_t)mode; }
static id<MTLRenderPipelineState> mtk_build(const pso_key *k, NvShader *ts, NvShader *ms, NvShader *fs, NSError **error) {
    MTLRenderPipelineDescriptor *d = pso_desc(k, ms, fs, nil, 0);
    if (!pso_ds_valid((uint32_t)d.depthAttachmentPixelFormat, (uint32_t)d.stencilAttachmentPixelFormat)) return nil;
    MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
    m.meshFunction = ms->fn; m.fragmentFunction = d.fragmentFunction ?: mtk_nop_fragment(); m.maxTotalThreadsPerMeshThreadgroup = ms->mesh_threads;
    if (ts) { m.objectFunction = ts->fn; m.payloadMemoryLength = (ts->mesh_payload + 15u) & ~15u; m.maxTotalThreadsPerObjectThreadgroup = ts->mesh_threads; }
    { int mp = mtk_key_mode(k);   // the mode this pipeline was asked with (mesh_pso_mode at mtk_pso)
      if (mp & 1) { m.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth = ms->mesh_threads % 32 == 0; m.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth = !ts || ts->mesh_threads % 32 == 0; }
      if (mp & 2) m.maxTotalThreadgroupsPerMeshGrid = 1024;
      if (mp & 4) { m.fragmentFunction = mtk_nop_fragment(); for (int i = 0; i < 8; i++) if (d.colorAttachments[i].pixelFormat) d.colorAttachments[i].writeMask = MTLColorWriteMaskNone; } }
    m.rasterSampleCount = d.rasterSampleCount; m.alphaToCoverageEnabled = d.alphaToCoverageEnabled;
    for (int i = 0; i < 8; i++) m.colorAttachments[i] = d.colorAttachments[i];
    m.depthAttachmentPixelFormat = d.depthAttachmentPixelFormat; m.stencilAttachmentPixelFormat = d.stencilAttachmentPixelFormat;
    return [g_dev newRenderPipelineStateWithMeshDescriptor:m options:MTLPipelineOptionNone reflection:nil error:error];
}
static void mtk_bind_key(pso_key *k, NvShader *ts, NvShader *ms, NvShader *fs) {
    k->vs = (__bridge void *)ms; k->fs = (__bridge void *)fs;
    k->fsfn = (__bridge void *)(ts ? ts->fn : ms->fn);   // with ms in the key's vertex place: the task shader is the third of the pipeline's functions
}
static id<MTLRenderPipelineState> mtk_pso(NvProg *raster) {
    NvShader *ts = g_mtk.ts, *ms = g_mtk.ms, *fs = raster->sh[1];
    pso_key k; pso_key_fill(&k, ms, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, 0);
    memset(k.attr, 0, sizeof k.attr); memset(k.attr_hi, 0, sizeof k.attr_hi); memset(k.stream, 0, sizeof k.stream);
    mtk_bind_key(&k, ts, ms, fs); mtk_key_set_mode(&k, mesh_pso_mode());
    id ps = hm_get_id(&g_pso_mtk, &k);
    if (ps) return ps == NSNull.null ? nil : ps;
    NSError *e = nil; uint64_t t0 = bp_now();
    id<MTLRenderPipelineState> p = mtk_build(&k, ts, ms, fs, &e);
    MLOG("%stask / mesh pipeline: task %s mesh %s fragment %s, %.1f ms%s%s", p ? "" : "[RED] ", ts ? ts->key : "-", ms->key, fs ? fs->key : "-", (bp_now() - t0) / 1e6, p ? "" : ": ", p ? "" : e ? e.localizedDescription.UTF8String : "invalid depth / stencil formats");
    hm_put_id(&g_pso_mtk, &k, p ?: NSNull.null);
    return p;
}
static uint64_t g_cpre_dbg_gpu;   // DM_NVN_MESH_CPRE_DBG=1|<dir>: the mesh path runs main0_dbg (the overlay's mesh wrapper that copies its finished stage array out) into this buffer (slot 30 of the mesh stage); a captured draw's own buffer (cpre_record, cpre_dbg_dump)
static int cpre_dbg(void) { static int m = -1; if (m < 0) { m = getenv("DM_NVN_MESH_CPRE_DBG") != NULL; if (m) MLOG("DM_NVN_MESH_CPRE_DBG: mesh-only pre-pass draws of profiled frames captured both ways (the kernels' slices and the mesh path's stage arrays, main0_dbg), compared and written to the directory a few frames later ([mtk-dbg])"); } return m; }
// A captured draw, a few frames later (the GPU is done with both halves): its pre-pass buffer (header, indices, the kernels' stage-array slices, the argument words) and the mesh path's
// stage arrays of its first 64 threadgroups (main0_dbg), written as <dir>/cpre_<mesh key>_F<frame>_{c,m}.bin for ZEL_build/agents/grass/dbgcmp.py; a line with the counts of differing words
static void cpre_dbg_dump(NvShader *ms, int force) {
    if (!ms->dbg_cb || (!force && (uint32_t)atomic_load(&g_cur_frame) < ms->dbg_frame + 8)) return;
    const char *dir = getenv("DM_NVN_MESH_CPRE_DBG"); char home[1024];
    if (dir && dir[0] == '~' && dir[1] == '/' && getenv("HOME")) { snprintf(home, sizeof home, "%s%s", getenv("HOME"), dir + 1); dir = home; }   // ~/Documents/... on iOS: the app container's
    if (!dir || dir[0] != '/') dir = getenv("DM_TEMP"); if (!dir) dir = ".";
    mkdir(dir, 0755);
    const uint32_t *c = (const uint32_t *)ms->dbg_cb.contents, *m = (const uint32_t *)ms->dbg_mb.contents; uint32_t sw = ms->cpre_sw, n;
    const uint32_t *a = (const uint32_t *)((const uint8_t *)ms->dbg_cb.contents + ms->dbg_cb_len - 64); n = a[0];   // the threadgroups the pre-pass dispatched
    char p[1024]; snprintf(p, sizeof p, "%s/cpre_%s_F%u_c.bin", dir, ms->key, ms->dbg_frame); FILE *f = fopen(p, "wb"); if (f) { fwrite(c, 1, ms->dbg_cb_len, f); fclose(f); }
    snprintf(p, sizeof p, "%s/cpre_%s_F%u_m.bin", dir, ms->key, ms->dbg_frame); f = fopen(p, "wb"); if (f) { fwrite(m, 4, (size_t)64 * sw, f); fclose(f); }
    uint32_t k = n < 64 ? n : 64, diff = 0, tgd = 0, first = ~0u, nz = 0;
    for (uint32_t t = 0; t < k; t++) { const uint32_t *x = c + c[3] + (size_t)t * sw, *y = m + (size_t)t * sw; int d = 0; for (uint32_t i = 0; i < sw; i++) { if (y[i]) nz++; if (x[i] != y[i]) { diff++; d = 1; if (first == ~0u) first = t << 16 | i; } } tgd += d; }
    MLOG("[mtk-dbg] mesh %s frame %u: %u threadgroups dispatched (record %u), %u compared: %u words differ in %u threadgroups (first tg %u word %u), mesh path nonzero words %u; %s", ms->key, ms->dbg_frame, n, ms->dbg_n, k, diff, tgd,
         first == ~0u ? 0 : first >> 16, first == ~0u ? 0 : first & 0xffff, nz, p);
    ms->dbg_cb = nil; ms->dbg_mb = nil; ms->dbg_caps++;
}
static int g_cpre_dbg_cap;   // cpre_record took a DBG capture: mesh_tasks_draw runs the mesh path too
static void mtk_encode(id<MTLRenderCommandEncoder> en_) {
    id<MTL4RenderCommandEncoder> en = (id<MTL4RenderCommandEncoder>)en_; NvShader *ts = g_mtk.ts, *ms = g_mtk.ms;
    [en setArgumentTable:g_vt atStages:MTLRenderStageMesh];
    uint64_t first_gpu = g_mtk.ind_first;
    if (!g_mtk.ind) { id<MTLBuffer> b; size_t o = arena_alloc(16, &b); memcpy((uint8_t *)b.contents + o, &g_mtk.first, 4); first_gpu = b.gpuAddress + o; }
    if (ts) { [en setArgumentTable:g_ot atStages:MTLRenderStageObject]; g_args_dirty[3] = 1; encode_args(en, ts, 3, 0);
              if (rec_on()) rec_table_op(RO_T_ADDR, g_ot, first_gpu, DMSH_MESH_FIRST_SLOT); else [g_ot setAddress:first_gpu atIndex:DMSH_MESH_FIRST_SLOT]; }
    g_args_dirty[4] = 1; encode_args(en, ms, 4, 1);
    enc_r_vbuf(en, DMSH_MESH_FIRST_SLOT, first_gpu);
    if (g_cpre_dbg_gpu) enc_r_vbuf(en, 30, g_cpre_dbg_gpu);   // DM_NVN_MESH_CPRE_DBG: main0_dbg copies its stage array here
    if (g_mtk.ind) [en drawMeshThreadgroupsWithIndirectBuffer:g_mtk.ind threadsPerObjectThreadgroup:MTLSizeMake(ts ? ts->mesh_threads : 1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(ms->mesh_threads, 1, 1)];
    else [en drawMeshThreadgroups:MTLSizeMake(g_mtk.count, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(ts ? ts->mesh_threads : 1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(ms->mesh_threads, 1, 1)];
}
// ---------- The compute pre-pass for the indirect task / mesh grass draws (DM_NVN_MESH_CPRE=1|ab; Ocarina of Time, agents cp2 / grass) ----------
// On Apple GPUs the heavy grass mesh shaders cost ~4 us a task threadgroup of vertex-phase time whatever the draw's form (cp2.md section 5): a 32-thread threadgroup with a 13 KB
// "shared memory" stage array (plus the mesh output) holds the mesh stage to about one threadgroup a core, and the 5-9 k instruction straight-line body makes each one ~75 us long,
// so the ~1,300 mesh threadgroups a frame run nearly serially per core. The library's compute pre-pass entries (an overlay made by ZEL_build/agents/grass/mkcpre2.py from the library's
// sources; later the chain's dmsh_meshstage.py) run the same bodies as compute kernels whose stage array is a per-threadgroup slice of a device buffer (no threadgroup memory: the
// occupancy is the GPU's), and a pass-through vertex function main0_pt reads the vertices from that slice in the stage layout. main0_c is the mesh body (mesh-only: slice [0] = first + tg;
// task-fed: the task's list entry names its task threadgroup and mesh index, the payload is read from the task's slice), main0_tc in the task shader's library is the task body: it
// counts its mesh threadgroups (the ballot the game's culling makes) on an atomic and appends (task << 10 | mesh) entries to a list, so the mesh kernel's grid is the exact fan-out.
// dm_cpre2_<vertices>_<primitives>_<stage words>_<task stage words> names the sizes. For such a draw (indirect, a colour target bound, the entries present) the host records the draw
// with its state; before the next render pass begins (cpre_prepass from rp_begin_ex_) one compute encoder runs, for all recorded draws: the argument kernels (the record's count
// clamped to the caps → the task / mesh dispatch and the indexed draw's arguments), the task kernels, the mesh-count argument kernels, the mesh kernels, so the draws' bodies overlap
// each other; the draws are then encoded first thing in the pass (cpre_flush from draw_) as one indirect indexed draw each with the pass-through pipeline, the pixel shader's table as it
// was at the draw, and the draw's depth / stencil state. Buffer header (words): [0] index region offset, [1] mesh cap, [2] indices a mesh threadgroup, [3] mesh scratch offset, [4] task cap,
// [6] list offset, [7] mesh counter (read back a frame later: the fan-out, for the cap and the [mtk] line), [8] task scratch offset. DM_NVN_MESH_CPRE_CAP=<n>: the mesh threadgroups the
// buffer holds (else from the counts seen, 64..4096); the grass draws are consecutive, so the pass is still split once (by mdx_run), not per draw.
// The colour pass draws the grass with depth EQUAL against the depth pre-pass's mesh shaders, so the kernels' positions must be bit-identical to their [[invariant]] ones: the entries'
// sources carry "#pragma clang fp contract(off) reassociate(off)" (mkcpre2.py), the flags -fpreserve-invariance gives an invariant output's chain. Without it the compiler fused other
// multiply-adds in the kernel's copy of the body, positions came out 1 ulp off and those blades failed the EQUAL test, leaving white specks (grass.md 4.4)
enum { CPRE_SLOT = 30, CPRE_MAX = 32, CPRE_HDR = 16, CPRE_MAGIC = 0x43505245 };
static struct { int n, ready, done; struct { NvShader *ts, *ms, *fs; NvProg *pr; typeof(S) st; p_set_rt rt; uint64_t rec, cg, ag, tgpu, mgpu, fgpu; NSUInteger tai, mai, fai; size_t ioff, ilen; uint32_t cap, tcap; } d[CPRE_MAX];   // tgpu / mgpu / fgpu: the task, mesh and pixel stages' argument tables as written at the draw (the uniform ring blocks of that moment)
                uint64_t draws, flushes, lost, splits, dispatches; } g_cpre;
static nvn_hmap g_pso_cpre = {.klen = sizeof(pso_key)};
static int cpre_mode(void) { static int m = -1; if (m < 0) { m = sw_mode("DM_NVN_MESH_CPRE", 0); if (m == 1) MLOG("DM_NVN_MESH_CPRE: indirect task / mesh draws of colour passes expanded by a compute pre-pass and drawn as indexed triangles (where the library has main0_c / main0_pt / main0_tc)"); } return m; }
static uint32_t cpre_clamp(uint32_t n, uint32_t lo, uint32_t hi) { return n < lo ? lo : n > hi ? hi : n; }
static uint32_t cpre_cap_env(void) { static long c = -1; if (c < 0) { const char *e = getenv("DM_NVN_MESH_CPRE_CAP"); c = e ? atol(e) : 0; if (c < 0) c = 0; } return (uint32_t)c; }
static int cpre_fns(NvShader *ms, NvShader *ts) {
    if (ms->cpre_state < 0) return 0;
    if (!ms->cpre_state) {
        ms->cpre_state = -1;
        static int said; const char *why = NULL;
        if (!ms->lib || !ms->ventry) why = "no library handle or entry name";
        if (!why && ![ms->ventry isEqualToString:@"main0"]) why = "an entry name other than main0";
        unsigned maxv = 0, maxp = 0, sw = 0, tsw = 0;
        if (!why) for (NSString *n in ms->lib.functionNames) if (sscanf(n.UTF8String, "dm_cpre2_%u_%u_%u_%u", &maxv, &maxp, &sw, &tsw) == 4) break;
        if (!why && (!maxv || !maxp || !sw)) why = "no dm_cpre2_<vertices>_<primitives>_<stage words>_<task stage words> kernel in its library";
        if (!why) { ms->fn_c = [ms->lib newFunctionWithName:@"main0_c"]; ms->fn_pt = [ms->lib newFunctionWithName:@"main0_pt"]; if (!ms->fn_c || !ms->fn_pt) why = "main0_c / main0_pt missing"; }
        if (why) { if (said++ < 4) MLOG("compute pre-pass: mesh %s (entry %s) stays on the mesh path: %s", ms->key, ms->ventry ? ms->ventry.UTF8String : "-", why); return 0; }
        NSError *e = nil; ms->cpre_ps = cps_new(ms->fn_c, &e);
        if (!ms->cpre_ps) { MLOG("[RED] compute pre-pass: no compute pipeline for mesh %s: %s", ms->key, e.localizedDescription.UTF8String); return 0; }
        ms->cpre_maxv = (uint16_t)maxv; ms->cpre_maxp = (uint16_t)maxp; ms->cpre_sw = (uint16_t)sw; ms->cpre_tsw = (uint16_t)tsw; ms->cpre_state = 1;
        MLOG("compute pre-pass: mesh %s: %u vertices, %u primitives, %u stage words a threadgroup%s%u", ms->key, maxv, maxp, sw, tsw ? ", task-fed, task stage words " : ", mesh-only ", tsw);
    }
    if ((ms->cpre_tsw != 0) != (ts != NULL)) { static int said; if (said++ < 4) MLOG("compute pre-pass: mesh %s stays on the mesh path: its entries are %s but the draw %s a task shader", ms->key, ms->cpre_tsw ? "task-fed" : "mesh-only", ts ? "has" : "has not"); return 0; }
    if (!ts) return 1;
    if (ts->cpre_state < 0) return 0;
    if (!ts->cpre_state) {
        ts->cpre_state = -1;
        if (!ts->lib) { MLOG("compute pre-pass: task %s: no library handle", ts->key); return 0; }
        ts->fn_c = [ts->lib newFunctionWithName:@"main0_tc"];
        if (!ts->fn_c) { MLOG("compute pre-pass: task %s has no main0_tc: mesh %s stays on the mesh path", ts->key, ms->key); return 0; }
        NSError *e = nil; ts->cpre_ps = cps_new(ts->fn_c, &e);
        if (!ts->cpre_ps) { MLOG("[RED] compute pre-pass: no compute pipeline for task %s: %s", ts->key, e.localizedDescription.UTF8String); return 0; }
        ts->cpre_state = 1; MLOG("compute pre-pass: task %s: main0_tc built", ts->key);
    }
    return 1;
}
static id<MTLComputePipelineState> g_cpre_args_ps[3];   // the argument kernels: mesh-only (record → mesh dispatch + draw), task (record → task dispatch, counter reset), mesh count (counter → mesh dispatch + draw)
static int cpre_args_ps(void) {
    static dispatch_once_t once; static int ok;
    dispatch_once(&once, ^{ NSError *e = nil;
        id<MTLLibrary> l = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void dm_cpre_args(device const uint* rec [[buffer(0)]], device uint* hdr [[buffer(1)]], device uint* args [[buffer(2)]]) { uint n = min(rec[0], hdr[1]); args[0] = n; args[1] = 1u; args[2] = 1u; args[3] = 0u; args[4] = n * hdr[2]; args[5] = 1u; args[6] = 0u; args[7] = 0u; args[8] = 0u; }\n"
            "kernel void dm_cpre_targs(device const uint* rec [[buffer(0)]], device uint* hdr [[buffer(1)]], device uint* args [[buffer(2)]]) { uint n = min(rec[0], hdr[4]); args[0] = n; args[1] = 1u; args[2] = 1u; args[3] = 0u; args[4] = 0u; args[5] = 1u; args[6] = 0u; args[7] = 0u; args[8] = 0u; hdr[7] = 0u; }\n"
            "kernel void dm_cpre_margs(device const uint* rec [[buffer(0)]], device uint* hdr [[buffer(1)]], device uint* args [[buffer(2)]]) { uint n = min(hdr[7], hdr[1]); args[12] = n; args[13] = 1u; args[14] = 1u; args[4] = n * hdr[2]; }\n" options:nil error:&e];
        const char *names[3] = {"dm_cpre_args", "dm_cpre_targs", "dm_cpre_margs"}; ok = l != nil;
        for (int i = 0; ok && i < 3; i++) { id<MTLFunction> f = [l newFunctionWithName:@(names[i])]; g_cpre_args_ps[i] = f ? cps_new(f, &e) : nil; if (!g_cpre_args_ps[i]) ok = 0; }
        if (!ok) MLOG("[RED] compute pre-pass: the argument kernels did not build: %s", e.localizedDescription.UTF8String); });
    return ok;
}
static int cpre_draw_mesh(void) { static int m = -1; if (m < 0) { const char *e = getenv("DM_NVN_MESH_CPRE_DRAW"); m = e && !strcmp(e, "mesh"); if (m) MLOG("DM_NVN_MESH_CPRE_DRAW=mesh: the pre-pass's draws go through a pass-through mesh pipeline (main0_ptm over the slice) instead of an indexed draw"); } return m; }
static id<MTLRenderPipelineState> cpre_pso_mesh(NvShader *ms, NvShader *fs) {   // the pass-through mesh pipeline: main0_ptm reads the slice as the mesh wrapper would its stage array
    if (!ms->fn_ptm) { ms->fn_ptm = [ms->lib newFunctionWithName:@"main0_ptm"]; if (!ms->fn_ptm) { static int said; if (said++ < 2) MLOG("[RED] compute pre-pass: mesh %s has no main0_ptm", ms->key); return nil; } }
    pso_key k; pso_key_fill(&k, ms, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, 0);
    memset(k.attr, 0, sizeof k.attr); memset(k.attr_hi, 0, sizeof k.attr_hi); memset(k.stream, 0, sizeof k.stream);
    k.vs = (__bridge void *)ms; k.fs = (__bridge void *)fs; k.fsfn = (__bridge void *)ms->fn_ptm;
    id ps = hm_get_id(&g_pso_cpre, &k);
    if (ps) return ps == NSNull.null ? nil : ps;
    MTLRenderPipelineDescriptor *d = pso_desc(&k, ms, fs, nil, 0);
    MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
    m.meshFunction = ms->fn_ptm; m.fragmentFunction = d.fragmentFunction ?: mtk_nop_fragment(); m.maxTotalThreadsPerMeshThreadgroup = 32;
    m.rasterSampleCount = d.rasterSampleCount; m.alphaToCoverageEnabled = d.alphaToCoverageEnabled;
    for (int i = 0; i < 8; i++) m.colorAttachments[i] = d.colorAttachments[i];
    m.depthAttachmentPixelFormat = d.depthAttachmentPixelFormat; m.stencilAttachmentPixelFormat = d.stencilAttachmentPixelFormat;
    NSError *e = nil; uint64_t t0 = bp_now(); ps = [g_dev newRenderPipelineStateWithMeshDescriptor:m options:MTLPipelineOptionNone reflection:nil error:&e];
    MLOG("%scompute pre-pass pipeline: mesh %s (pass-through mesh function) fragment %s, %.1f ms%s%s", ps ? "" : "[RED] ", ms->key, fs ? fs->key : "-", (bp_now() - t0) / 1e6, ps ? "" : ": ", ps ? "" : e.localizedDescription.UTF8String);
    hm_put_id(&g_pso_cpre, &k, ps ?: NSNull.null);
    return ps;
}
static id<MTLRenderPipelineState> cpre_pso(NvShader *ms, NvShader *fs) {
    pso_key k; pso_key_fill(&k, ms, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, 0);
    memset(k.attr, 0, sizeof k.attr); memset(k.attr_hi, 0, sizeof k.attr_hi); memset(k.stream, 0, sizeof k.stream);
    k.vs = (__bridge void *)ms; k.fs = (__bridge void *)fs; k.fsfn = (__bridge void *)ms->fn_pt;
    id ps = hm_get_id(&g_pso_cpre, &k);
    if (ps) return ps == NSNull.null ? nil : ps;
    MTLRenderPipelineDescriptor *d = pso_desc(&k, ms, fs, nil, 0);
    d.vertexFunction = ms->fn_pt; d.vertexDescriptor = nil;   // the chain's dm_vertex_t outputs are float4 (as the pixel shaders' inputs): the pass-through returns it as it is
    NSError *e = nil; uint64_t t0 = bp_now(); ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
    MLOG("%scompute pre-pass pipeline: mesh %s (pass-through vertex) fragment %s, %.1f ms%s%s", ps ? "" : "[RED] ", ms->key, fs ? fs->key : "-", (bp_now() - t0) / 1e6, ps ? "" : ": ", ps ? "" : e.localizedDescription.UTF8String);
    hm_put_id(&g_pso_cpre, &k, ps ?: NSNull.null);
    return ps;
}
// The draw is recorded with its state and buffers; its compute work runs in cpre_dispatch before the next render pass, the draw itself in cpre_flush inside it. rec_gpu: the
// DrawMeshTasksIndirect record {count, first}. A frame later the header's mesh counter says what the task fan-out was (ms->cpre_fan: the mesh cap of the next frames)
static void cpre_dispatch(void);
static int cpre_record(NvProg *pr, NvShader *ts, NvShader *ms, NvShader *fs, uint64_t rec_gpu) {
    int mode = cpre_mode(); if (!mode || !sw_on(mode)) return 0;
    if (!fs) { static int said; if (said++ < 2) MLOG("compute pre-pass: mesh %s stays on the mesh path: no pixel shader", ms->key); return 0; }
    { int col = 0; for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) col = 1; if (!col) return 0; }   // colour passes only: the depth / shadow passes draw the grass with light variants
    { static int task = -1; if (task < 0) { const char *e = getenv("DM_NVN_MESH_CPRE_TASK"); task = e ? atoi(e) : 1; if (!task) MLOG("DM_NVN_MESH_CPRE_TASK=0: the task-fed grass draws stay on the mesh path"); } if (ts && !task) return 0; }
    if (!cpre_fns(ms, ts) || !cpre_args_ps()) return 0;
    if (g_cpre.n >= CPRE_MAX) { g_cpre.lost++; return 0; }
    if (g_cpre.n && g_enc) { g_cpre.lost += (uint64_t)g_cpre.n; g_cpre.n = 0; g_cpre.ready = 0; g_cpre.done = 0; }   // dispatched but never flushed (no draw followed in a pass): dropped
    image_prepare(pr, 1u << 1);   // the pixel stage's images (uploads, views), as draw_ would have done for the mesh path's raster program
    uint32_t seen = g_mtk.mem_count, fan = 0, ran = 0, ranm = 0;   // last frame's buffers as the GPU left them: the mesh counter (task draws), the dispatch counts
    const uint32_t *rh = ms->cpre_rb ? (const uint32_t *)((const uint8_t *)ms->cpre_rb.contents + ms->cpre_rb_off) : NULL; if (rh && (rh[9] != CPRE_MAGIC || rh[10] != ms->cpre_rb_frame)) rh = NULL;   // recycled meanwhile
    const uint32_t *ra = ms->cpre_ra ? (const uint32_t *)((const uint8_t *)ms->cpre_ra.contents + ms->cpre_ra_off) : NULL; if (ra && (ra[10] != CPRE_MAGIC || ra[11] != ms->cpre_rb_frame)) ra = NULL;
    if (rh) { fan = rh[7]; if (fan > 65536) fan = 0; ms->cpre_fan = fan; }
    if (ra) { ran = ra[0]; ranm = ra[12]; }
    if (gp_active() && rh) {   // diagnostic: last frame's slice 0 as the pass-through reads it: vertex 0's words (hex = float), the packed id (a0) of vertices 0-7, the first indices
        const uint32_t *h = rh; const uint32_t *sl = h + h[3]; const uint32_t *ix = h + h[0]; if (h[3] < CPRE_HDR + (size_t)h[1] * h[2] + 1 && h[0] == CPRE_HDR) {
        char buf[640]; int o = 0; for (int k = 0; k < 20 && o < 400; k++) { uint32_t w = sl[36 + 32 * k]; float f; memcpy(&f, &w, 4); o += snprintf(buf + o, sizeof buf - o, " %08x=%.4g", w, (double)f); }
        for (int n = 0; n < 8 && o < 600; n++) o += snprintf(buf + o, sizeof buf - o, "%s%08x", n ? " " : "; a0 of vertices 0-7:", sl[36 + (n / 32) * 800 + n % 32 + 128]);
        MLOG("[mtk-rb] mesh %s: v0 words%s; indices %u %u %u %u %u %u", ms->key, buf, ix[0], ix[1], ix[2], ix[3], ix[4], ix[5]); } }
    uint32_t tcap = ts ? cpre_clamp(seen * 2 + 32, 32, 2048) : 0;
    uint32_t cap = cpre_cap_env() ? cpre_cap_env() : ts ? cpre_clamp((fan > seen * 4 ? fan + fan / 2 : seen * 4) + 64, 64, 4096) : cpre_clamp(seen * 2 + 64, 64, 4096);
    size_t iw = (size_t)cap * ms->cpre_maxp * 3, sw = (size_t)cap * ms->cpre_sw, tw = (size_t)tcap * ms->cpre_tsw, lw = ts ? cap : 0;
    size_t total = CPRE_HDR + iw + sw + tw + lw;
    id<MTLBuffer> cb; size_t co;
    int cap_dbg = cpre_dbg() && ms->fn_dbg && !ts && !ms->dbg_cb && ms->dbg_caps < 4 && gp_active();   // DM_NVN_MESH_CPRE_DBG: capture this draw both ways (own buffers, compared a few frames later)
    if (cap_dbg) { cb = [g_dev newBufferWithLength:total * 4 + 64 options:MTLResourceStorageModeShared]; co = 0; res_add(cb); res_commit(); }
    else co = arena_alloc(total * 4, &cb);
    uint32_t *h = (uint32_t *)((uint8_t *)cb.contents + co); memset(h, 0, CPRE_HDR * 4);
    uint32_t fr = (uint32_t)atomic_load(&g_cur_frame);
    h[0] = CPRE_HDR; h[1] = cap; h[2] = ms->cpre_maxp * 3u; h[3] = (uint32_t)(CPRE_HDR + iw); h[4] = tcap; h[6] = (uint32_t)(CPRE_HDR + iw + sw + tw); h[8] = (uint32_t)(CPRE_HDR + iw + sw);
    h[9] = CPRE_MAGIC; h[10] = fr;   // the readbacks a frame later trust the buffer only with these (the arena may have been recycled by then)
    id<MTLBuffer> ab; size_t ao; if (cap_dbg) { ab = cb; ao = total * 4; } else ao = arena_alloc(64, &ab); memset((uint8_t *)ab.contents + ao, 0, 64);
    if (cap_dbg) { size_t ml = (size_t)64 * ms->cpre_sw * 4; ms->dbg_mb = [g_dev newBufferWithLength:ml options:MTLResourceStorageModeShared]; res_add(ms->dbg_mb); res_commit(); memset(ms->dbg_mb.contents, 0, ml);
                   ms->dbg_cb = cb; ms->dbg_cb_len = total * 4 + 64; ms->dbg_frame = fr; ms->dbg_n = seen; g_cpre_dbg_gpu = ms->dbg_mb.gpuAddress; g_cpre_dbg_cap = 1; }
    { uint32_t *a = (uint32_t *)((uint8_t *)ab.contents + ao); a[10] = CPRE_MAGIC; a[11] = fr; }
    ms->cpre_rb_frame = fr;
    typeof(g_cpre.d[0]) *d = &g_cpre.d[g_cpre.n++];
    d->ts = ts; d->ms = ms; d->fs = fs; d->pr = pr; d->st = S; d->rt = R.rt; d->rec = rec_gpu;
    d->cg = cb.gpuAddress + co; d->ag = ab.gpuAddress + ao; d->ioff = CPRE_HDR * 4; d->ilen = iw * 4; d->cap = cap; d->tcap = tcap;
    // The stages' argument tables are written now, as the mesh path would have at this draw (uniform slots resolve to the ring blocks of this moment; later updates for the following
    // draws must not reach them), and only bound at the dispatch / flush
    g_cpre_args = 2;
    if (ts) { g_args_dirty[3] = 1; g_cpre_rec_gpu = 0; encode_args((id)nil, ts, 3, 0); d->tgpu = g_cpre_rec_gpu; d->tai = g_cpre_rec_ai; }
    g_args_dirty[4] = 1; g_cpre_rec_gpu = 0; encode_args((id)nil, ms, 4, 0); d->mgpu = g_cpre_rec_gpu; d->mai = g_cpre_rec_ai;
    g_args_dirty[1] = 1; g_cpre_rec_gpu = 0; encode_args((id)nil, fs, 1, 0); d->fgpu = g_cpre_rec_gpu; d->fai = g_cpre_rec_ai;
    g_cpre_args = 0; args_dirty_all();
    ms->cpre_rb = cb; ms->cpre_rb_off = co; ms->cpre_ra = ab; ms->cpre_ra_off = ao;
    g_cpre.draws++;
    { static int imm = -1; if (imm < 0) imm = getenv("DM_NVN_MESH_CPRE_IMM") != NULL; if (imm || cap_dbg) cpre_dispatch(); }   // a DBG capture: now, so the mesh path's draw that follows flushes it   // DM_NVN_MESH_CPRE_IMM=1 (test): now, not at the next pass: an encoder begun later lands the uniform updates the game makes for the draws after this one (ubd_land_compute), and the kernels would read them
    if (gp_active()) MLOG("[mtk] F%llu pass %llu task %s mesh %s fs %s: compute pre-pass (the record as the CPU sees it) threadgroups %u first %u, task cap %u mesh cap %u, %zu KB; last frame as the GPU left it: %u task / %u mesh threadgroups dispatched, fan-out %u", (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)g_stat_pass, ts ? ts->key : "-", ms->key, fs->key, seen, g_mtk.mem_first,
                         tcap, cap, total * 4 / 1024, ts ? ran : 0, ts ? ranm : ran, fan);
    static int once; if (!once++) MLOG("compute pre-pass: first draw, task %s mesh %s fragment %s, %u threadgroups seen, %zu KB", ts ? ts->key : "-", ms->key, fs->key, seen, total * 4 / 1024);
    return 1;
}
// The recorded draws' compute work, in one encoder, before the render pass that holds them begins (rp_begin_ex_ → cpre_prepass): all argument kernels, the task kernels, the mesh-count
// kernels, the mesh kernels, a dispatch barrier between the groups and none inside them, so the draws' threadgroups fill the GPU together
static void cpre_dispatch(void) {
    if (g_cpre.done >= g_cpre.n) return;
    if (g_enc) { g_cpre.splits++; rp_end(); }
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL); int i0 = g_cpre.done, n = g_cpre.n, tasks = 0;
    typeof(S) s0 = S;   // each kernel's table is encoded under its draw's state (encode_args_ reads the stage's uniform buffers from S), then the current state comes back
    for (int i = i0; i < n; i++) { typeof(g_cpre.d[0]) *d = &g_cpre.d[i]; tasks += d->ts != nil;
        enc_c_pso(ce, g_cpre_args_ps[d->ts ? 1 : 0]); enc_c_args(ce, d->rec, 0); enc_c_args(ce, d->cg, 1); enc_c_args(ce, d->ag, 2); enc_c_dispatch(ce, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1)); }
    [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:resource_vis()];
    if (tasks) {
        for (int i = i0; i < n; i++) { typeof(g_cpre.d[0]) *d = &g_cpre.d[i]; if (!d->ts) continue;
            enc_c_pso(ce, d->ts->cpre_ps); S = d->st;
            if (d->tgpu) enc_c_args(ce, d->tgpu, d->tai);
            enc_c_args(ce, d->rec + 4, DMSH_MESH_FIRST_SLOT); enc_c_args(ce, d->cg, CPRE_SLOT);
            enc_c_dispatch_indirect(ce, d->ag, MTLSizeMake(d->ts->mesh_threads, 1, 1)); }
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:resource_vis()];
        for (int i = i0; i < n; i++) { typeof(g_cpre.d[0]) *d = &g_cpre.d[i]; if (!d->ts) continue;
            enc_c_pso(ce, g_cpre_args_ps[2]); enc_c_args(ce, d->rec, 0); enc_c_args(ce, d->cg, 1); enc_c_args(ce, d->ag, 2); enc_c_dispatch(ce, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1)); }
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:resource_vis()];
    }
    for (int i = i0; i < n; i++) { typeof(g_cpre.d[0]) *d = &g_cpre.d[i];
        enc_c_pso(ce, d->ms->cpre_ps); S = d->st;
        if (d->mgpu) enc_c_args(ce, d->mgpu, d->mai);
        if (!d->ts) enc_c_args(ce, d->rec + 4, DMSH_MESH_FIRST_SLOT);
        enc_c_args(ce, d->cg, CPRE_SLOT);
        enc_c_dispatch_indirect(ce, d->ag + (d->ts ? 48 : 0), MTLSizeMake(d->ms->mesh_threads, 1, 1)); }
    S = s0;
    enc_c_end(ce); args_dirty_all(); g_cpre.done = n; g_cpre.ready = 1; g_cpre.dispatches++;
    static int once; if (!once++) MLOG("compute pre-pass: first dispatch, %d draws (%d task-fed) in one compute encoder", n, tasks);
}
static void cpre_prepass(void) { if (g_cpre.done < g_cpre.n) cpre_dispatch(); }   // rp_begin_ex_, before the pass's encoder is made
static void cpre_sub_end(void) { if (g_cpre.n) { g_cpre.lost += (uint64_t)g_cpre.n; g_cpre.n = 0; g_cpre.ready = 0; g_cpre.done = 0; } }   // the submission ends with draws recorded and no pass to flush them into
// The rasterizer state of a kept draw, from its saved S as draw_ sets it for its own draw (the same g_ed cache, so the draw that follows re-applies only what differs); the grass
// pixel shaders read front_facing, so the winding matters, and a new encoder starts with Metal's defaults
static void cpre_raster(id<MTL4RenderCommandEncoder> en) {
    { uint32_t sr = S.st_ref[0]; if (!(g_ed.ok & ED_SREF) || g_ed.sref != sr) { enc_r_stencil_ref(en, sr, sr); g_ed.sref = sr; g_ed.ok |= ED_SREF; } }
    NSUInteger tw = g_pass.tw, th = g_pass.th;
    { MTLViewport v = S.has_vp ? (MTLViewport){S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3], S.dr[0], S.dr[1]} : (MTLViewport){0, 0, tw, th, 0, 1};
      if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { v.originY += v.height; v.height = -v.height; }
      { uint32_t sw; memcpy(&sw, S.vp_sw[0], 4); if (sw == 0x100) { v.originY += v.height; v.height = -v.height; } }
      if (!(g_ed.ok & ED_VP) || memcmp(&g_ed.vp, &v, sizeof v)) { enc_r_viewport(en, v); g_ed.vp = v; g_ed.ok |= ED_VP; } }
    { MTLScissorRect sc = {0, 0, tw, th};
      if (R.has_sc && !S.rast_discard) { long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y <= 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h;
        if (x1 > (long)tw) x1 = tw; if (y1 > (long)th) y1 = th; if (x1 > x0 && y1 > y0) sc = (MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)}; }
      if (!(g_ed.ok & ED_SC) || memcmp(&g_ed.sc, &sc, sizeof sc)) { enc_r_scissor(en, sc); g_ed.sc = sc; g_ed.ok |= ED_SC; } }
    { int cu = S.poly.cull == NVN_FACE_FRONT ? MTLCullModeFront : S.poly.cull == NVN_FACE_BACK ? MTLCullModeBack : MTLCullModeNone;
      int wi = (S.poly.front == NVN_FRONT_FACE_CCW) != g_queue_origin_ll ? MTLWindingCounterClockwise : MTLWindingClockwise;
      { static int flip = -1; if (flip < 0) flip = getenv("DM_NVN_MESH_CPRE_WIND") != NULL; if (flip) wi = wi == MTLWindingClockwise ? MTLWindingCounterClockwise : MTLWindingClockwise; }   // DM_NVN_MESH_CPRE_WIND=1: the winding flipped (a test)
      int fi = S.poly.mode == NVN_POLYGON_MODE_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill, cl = !S.depth_clamp;
      float bi[3] = {0, 0, 0}; if (S.poly.offset_en & 4) { bi[0] = S.po.units; bi[1] = S.po.factor; bi[2] = S.po.clamp; }
      if (!(g_ed.ok & ED_CULL) || g_ed.cull != cu) { enc_r_cull(en, (MTLCullMode)cu); g_ed.cull = cu; g_ed.ok |= ED_CULL; }
      if (!(g_ed.ok & ED_WIND) || g_ed.wind != wi) { enc_r_winding(en, (MTLWinding)wi); g_ed.wind = wi; g_ed.ok |= ED_WIND; }
      if (!(g_ed.ok & ED_FILL) || g_ed.fill != fi) { enc_r_fill(en, (MTLTriangleFillMode)fi); g_ed.fill = fi; g_ed.ok |= ED_FILL; }
      if (!(g_ed.ok & ED_BIAS) || memcmp(g_ed.bias, bi, sizeof bi)) { enc_r_depth_bias(en, bi[0], bi[1], bi[2]); memcpy(g_ed.bias, bi, sizeof bi); g_ed.ok |= ED_BIAS; }
      if (!(g_ed.ok & ED_CLIP) || g_ed.clip != cl) { enc_r_depth_clip(en, cl); g_ed.clip = cl; g_ed.ok |= ED_CLIP; }
      if (!(g_ed.ok & ED_BLEND) || memcmp(g_ed.blend, S.blend_color, sizeof g_ed.blend)) { enc_r_blend_color(en, S.blend_color[0], S.blend_color[1], S.blend_color[2], S.blend_color[3]); memcpy(g_ed.blend, S.blend_color, sizeof g_ed.blend); g_ed.ok |= ED_BLEND; } }
}
// The kept draws, into the pass just opened (draw_, before its own draw)
static void cpre_flush(id en_) {
    id<MTL4RenderCommandEncoder> en = (id<MTL4RenderCommandEncoder>)en_;
    int n = g_cpre.n, ready = g_cpre.ready && g_cpre.done == n; g_cpre.n = 0; g_cpre.ready = 0; g_cpre.done = 0; if (!n) return;
    if (!en || !ready) { g_cpre.lost += (uint64_t)n; return; }
    [en setArgumentTable:g_vt atStages:MTLRenderStageVertex]; [en setArgumentTable:g_ft atStages:MTLRenderStageFragment];
    typeof(S) s0 = S;   // each kept draw is encoded under its own state, then the current draw's comes back
    for (int i = 0; i < n; i++) { typeof(g_cpre.d[0]) *d = &g_cpre.d[i];
        if (!rt_same(&R.rt, &d->rt)) { g_cpre.lost++; static int said; if (said++ < 3) MLOG("compute pre-pass: a recorded draw's pass never came (the next pass has other targets): dropped"); continue; }
        S = d->st;
        // DM_NVN_MESH_CPRE_DEQ=<units> (default 8; 0 = the exact EQUAL test): the colour pass draws the grass with depth EQUAL against the depth pass's mesh shaders' [[invariant]] positions; the kernels'
        // positions match them bit for bit only where the compiler computes the kernel's chain as it computes an invariant one (mkcpre2.py's fp pragma; true on macOS). With the switch an
        // EQUAL test without depth writes becomes LEQUAL with a constant depth bias of -<units> (toward the camera), so ulp differences pass. Needed on the iPad (iOS's compiler: every blade failed EQUAL, ipad_v19g3_deq.png); harmless on the Mac
        { static float deq = -1; if (deq < 0) { const char *e = getenv("DM_NVN_MESH_CPRE_DEQ"); deq = e ? (float)atof(e) : 8; if (deq < 0) deq = 0; if (deq > 0) MLOG("DM_NVN_MESH_CPRE_DEQ=%g: the pre-pass's depth EQUAL draws use LEQUAL with a depth bias of -%g", (double)deq, (double)deq); }
          if (deq > 0 && S.ds.dt && S.ds.df == NVN_DEPTH_FUNC_EQUAL && !S.ds.dw) { S.ds.df = NVN_DEPTH_FUNC_LEQUAL; if (!(S.poly.offset_en & 4)) { S.po.units = 0; S.po.factor = 0; S.po.clamp = 0; } S.poly.offset_en |= 4; S.po.units -= deq; } }
        int mesh = cpre_draw_mesh();
        id<MTLRenderPipelineState> ps = mesh ? cpre_pso_mesh(d->ms, d->fs) : cpre_pso(d->ms, d->fs);
        if (!ps) {   // no pass-through pipeline (it did not link with this pixel shader): this draw is lost, the mesh shader's later draws go back to the mesh path (else its grass would be missing every frame)
            g_cpre.lost++; if (d->ms->cpre_state > 0) { d->ms->cpre_state = -1; MLOG("[RED] compute pre-pass: mesh %s back on the mesh path: no pass-through pipeline with fragment %s", d->ms->key, d->fs->key); }
            continue; }
        id<MTLDepthStencilState> dss = dss_get(g_pass.hd, g_pass.hs);
        if (ps != g_enc_pso) { enc_r_pso(en, ps); g_enc_pso = ps; }
        if (mesh) [en setArgumentTable:g_vt atStages:MTLRenderStageMesh];
        if (dss != g_enc_dss) { enc_r_dss(en, dss); g_enc_dss = dss; }
        cpre_raster(en);
        if (d->fs->has_alpha && atest_on()) at_bind(en);   // the pixel stage: the alpha test and logic operation from the saved state, the table as written at the draw
        if (d->fs->has_logic && lop_on()) lo_bind(en);
        if (d->fgpu) bind_args(en, 1, 0, nil, d->fgpu, 0, d->fs->rf.abi);
        enc_r_vbuf(en, CPRE_SLOT, d->cg);
        if (mesh) [en drawMeshThreadgroupsWithIndirectBuffer:d->ag + (d->ts ? 48 : 0) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(32, 1, 1)];
        else enc_r_draw_indexed_indirect(en, MTLPrimitiveTypeTriangle, MTLIndexTypeUInt32, d->cg + d->ioff, d->ilen, d->ag + 16);
        g_stat_draw++;
    }
    for (int i = 0; i < n; i++) { g_cpre.d[i].ts = nil; g_cpre.d[i].ms = nil; g_cpre.d[i].fs = nil; g_cpre.d[i].pr = nil; }
    S = s0;
    g_pso_dirty = 1; g_dss_dirty = 1; args_dirty_all(); g_cpre.flushes++;
    static int once; if (!once++) MLOG("compute pre-pass: first flush, %d draws into pass %llu", n, (unsigned long long)g_stat_pass);
}
// DrawMeshTasksIndirect: {count, first}, which the game's culling writes earlier in the submission, while Metal's indirect mesh draw takes {x, y, z}. mdx_run copies the count to a {count, 1, 1} on the
// GPU, before the render pass and for every such draw up to the next write, and the shader reads first from the record. Reading the record on the CPU instead took a wait for the GPU at each hazard:
// 73 % of the submit thread in Ocarina of Time's Kokiri Forest, frames of 90 ms and more against 25 ms of GPU work
static void mesh_tasks_draw(const p_draw *dw) {
    if (!S.prog || (!dw->ind && dw->count <= 0)) return;
    // Diagnostic DM_NVN_SKIP_MESH=1 (off by default): drops every task / mesh draw, as DM_NVN_SKIP_FS drops draws by fragment shader, to measure what they cost in all (the picture is wrong)
    { static int skip = -1; if (skip < 0) { const char *e = getenv("DM_NVN_SKIP_MESH"); skip = e && atoi(e); if (skip) MLOG("DM_NVN_SKIP_MESH: task / mesh draws dropped (diagnostic; the picture is wrong)"); } if (skip) return; }
    __unsafe_unretained NvProg *pr = prog_get(S.prog);
    NvShader *ts = pr ? pr->sh[3] : nil, *ms = pr ? pr->sh[4] : nil;
    if (ts && ts->mesh_kind != 1) ts = nil;
    if (!ms || ms->mesh_kind != 2 || !ms->fn || (ts ? !ts->fn : !ms->mesh_notask)) {
        RED_ONCE(150, "DrawMeshTasks without its task and mesh shaders (task %s mesh %s): draw skipped", ts ? ts->key : "-", ms ? ms->key : "-"); return; }
    image_prepare(pr, 1u << 3 | 1u << 4);
    g_cpre_dbg_gpu = 0;
    if (cpre_dbg() && ms->lib) {
        if (!ms->fn_dbg) { ms->fn_dbg = [ms->lib newFunctionWithName:@"main0_dbg"]; if (ms->fn_dbg) { ms->fn = ms->fn_dbg; MLOG("compute pre-pass debug: mesh %s runs main0_dbg", ms->key); } }
        cpre_dbg_dump(ms, 0);
        if (ms->fn_dbg) { unsigned maxv = 0, maxp = 0, sw = 0, tsw = 0; for (NSString *n in ms->lib.functionNames) if (sscanf(n.UTF8String, "dm_cpre2_%u_%u_%u_%u", &maxv, &maxp, &sw, &tsw) == 4) break;
            if (sw) { id<MTLBuffer> b; size_t o = arena_alloc((size_t)64 * sw * 4, &b); g_cpre_dbg_gpu = b.gpuAddress + o; } } }   // main0_dbg always writes somewhere; a captured draw's own buffer replaces it (cpre_record)
    if (g_nubd && (g_enc || ubd_lazy()) && ((ts && ubd_conflict_sh(ts, 3)) || ubd_conflict_sh(ms, 4))) { g_ubd_conflicts++; rp_end(); ubd_land(); }   // as draw_ does for its stages
    NvProg *raster = pr->geo_lean_raster;   // the pixel stage alone
    if (!raster || raster->sh[1] != pr->sh[1] || raster->sh[0]) { raster = [NvProg new]; raster->sh[1] = pr->sh[1]; pr->geo_lean_raster = raster; }
    uint64_t ind = 0, ind_first = 0; uint32_t count = (uint32_t)dw->count, first = (uint32_t)dw->first;
    // Measurement DM_NVN_MESH_DIRECT=1|ab: an indirect task / mesh draw is encoded as a direct one with the record as the CPU sees it at replay (usually the previous frame's culling result:
    // the GPU writes it earlier in this submission), so no mdx_run before the pass (no pass split) and no indirect mesh draw on the GPU. Measures what the indirect form itself costs on a
    // device; a stale count draws last frame's number of task threadgroups (each culls its own patch), so the picture can lag the camera by a frame
    static int direct = -1; if (direct < 0) { direct = sw_mode("DM_NVN_MESH_DIRECT", 0); if (direct == 1) MLOG("DM_NVN_MESH_DIRECT: indirect task / mesh draws encoded as direct draws with the record as the CPU sees it"); }
    if (dw->ind) { void *mb; uint64_t mo; uint64_t gpu;
        if (!mtl_gpu_addr(dw->ind, 8, &mb, &mo, &gpu)) { RED_ONCE(154, "DrawMeshTasksIndirect record at %#llx is in no memory pool: draw skipped", (unsigned long long)dw->ind); return; }
        const uint32_t *rec = (const uint32_t *)((const uint8_t *)((__bridge id<MTLBuffer>)mb).contents + mo); g_mtk.mem_count = rec[0]; g_mtk.mem_first = rec[1];
        if (direct && sw_on(direct)) { count = rec[0]; first = rec[1]; if (!count) return; }
        else {
            if (!mdx_find(dw->ind, dw->ind, &ind)) { mdx_run(dw, NULL); if (!mdx_find(dw->ind, dw->ind, &ind)) ind = 0; }
            if (!ind) { RED_ONCE(155, "DrawMeshTasksIndirect record at %#llx: no converted arguments: draw skipped", (unsigned long long)dw->ind); return; }
            ind_first = gpu + 4;
            if (cpre_record(pr, ts, ms, pr->sh[1], gpu)) { swrite_note(pr); g_wr_epoch++; if (!g_cpre_dbg_cap) return; g_cpre_dbg_cap = 0; } } }   // a DBG capture also runs the mesh path (main0_dbg) for the same draw
    else { g_mtk.mem_count = count; g_mtk.mem_first = first; }
    g_mtk.on = 1; g_mtk.miss = 0; g_mtk.ts = ts; g_mtk.ms = ms; g_mtk.first = first; g_mtk.count = count;
    g_mtk.ind = ind; g_mtk.ind_first = ind_first;
    g_geo_program = raster;
    p_draw triangle = {.prim = 4, .count = 3, .instances = 1};
    draw_(&triangle, NULL);
    g_geo_program = nil; g_mtk.on = 0; args_dirty_all();
    if (g_mtk.miss) { g_mtk_wait++; if (ms->dbg_cb && ms->dbg_frame == (uint32_t)atomic_load(&g_cur_frame)) { ms->dbg_cb = nil; ms->dbg_mb = nil; } return; }   // a capture without its mesh-path half: dropped
    // One line per task / mesh draw of a profiled frame (DM_NVN_GPUPROF): what the GPU is asked to run, beside the [gpuprof] pass lines. The task's threadgroups (count) each launch up to
    // 1,024 mesh threadgroups (the wrapper's set_threadgroups_per_grid), so the draw's geometry is not known here; an indirect draw's count is the record as the CPU sees it (mem_count)
    if (gp_active()) MLOG("[mtk] F%llu pass %llu task %s mesh %s fs %s: %s threadgroups %u first %u, task threads %u mesh threads %u", (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)g_stat_pass, ts ? ts->key : "-", ms->key, pr->sh[1] ? pr->sh[1]->key : "-",
                         ind ? "indirect (the record as the CPU sees it)" : "direct", g_mtk.mem_count, g_mtk.mem_first, ts ? ts->mesh_threads : 1, ms->mesh_threads);
    swrite_note(pr); g_wr_epoch++;
    if (!g_mtk_n++) MLOG("task / mesh: first draw, task %s mesh %s fragment %s, %u workgroups from %u%s", ts ? ts->key : "-", ms->key, pr->sh[1] ? pr->sh[1]->key : "-", g_mtk.count, g_mtk.first, ind ? " (indirect)" : "");
}
static uint64_t g_geo_mesh_n, g_geo_mesh_wait;   // mesh draws; draws the compute path drew while their mesh pipeline was built
static int geo_mesh_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    id<MTLFunction> fn = geo_mesh_fn(pr, dw, di); if (!fn) return 0;
    uint64_t ib = 0; void *mb; uint64_t mo;
    if (di && !mtl_gpu_addr(di->ibuf, 1, &mb, &mo, &ib)) return 0;
    uint32_t bi = (uint32_t)(dw ? dw->base_instance : di->base_instance);
    NvProg *raster = pr->geo_lean_raster;
    if (!raster || raster->sh[0] != pr->sh[0] || raster->sh[1] != pr->sh[1]) { raster = [NvProg new]; raster->sh[0] = pr->sh[0]; raster->sh[1] = pr->sh[1]; pr->geo_lean_raster = raster; }
    if (g_mrun.n && (pr->sh[0] != g_mrun.vs || pr->sh[1] != g_mrun.fs || fn != g_mrun.fn || (__bridge void *)g_enc != (__bridge void *)g_mrun.en)) geo_mesh_flush_draw();
    g_geo_mesh.on = 1; g_geo_mesh.miss = 0; g_geo_mesh.fn = fn; g_geo_mesh.prims = geo_mesh_prims(fn); g_geo_mesh.lean = pr->sh[2]->geo_lean_v[3];
    g_geo_mesh.a = (geo_mesh_args){ib, di ? 1u << di->itype : 0, dw ? (uint32_t)dw->first : 0, (uint32_t)(dw ? dw->count : di->count), di ? di->base_vertex : 0, bi, 0};
    g_geo_program = raster;
    p_draw synthetic = {.prim = 4, .count = 3, .instances = 1, .base_instance = (int32_t)bi};
    g_geo_omit_colors = geo_color_omit();
    draw(&synthetic, NULL);
    g_geo_omit_colors = 0;
    g_geo_program = nil; g_geo_mesh.on = 0; args_dirty_all();
    if (g_geo_mesh.miss) { g_geo_mesh_wait++; return 0; }
    // the kernel's storage writes, noted as its arguments were encoded (always: geo_mesh_encode), so a drain's list of later writes holds them (g_wr_geo, drw_valid)
    swrite_note(pr); udc_note_draw(pr, 0); g_wr_epoch++; g_wr_geo++; g_mdx_epoch++;
    if (!g_geo_mesh_n++) MLOG("geometry mesh: first mesh draw, vertex %s kernel %s, %u triangles", pr->sh[0]->key, g_geo_mesh.lean->key, g_geo_mesh.a.count / 3);
    return 1;
}
typedef struct {
    const uint8_t *op;
    NvProg *pr; NvShader *vs, *gs, *reader;
    id<MTLRenderPipelineState> capture;
    p_draw dw, vertex_range; p_draw_idx di; int has_di;
    uint32_t first_instance, groups, emits;
    uint64_t np; size_t unique_count, nu;
    uint64_t input, output, raw, packed, packed_size, cnt, map, capture_ix, info, capture_info, args, job;
    uint64_t compact_map;   // a compact capture (nvn_geometry.h): the positions the geometry stage reads through, 0 = map
    const uint32_t *runs; size_t nruns;   // and its runs of IDs, {position, length} pairs in the scratch, each with a capture_info of its own
    geo_access access;
    uint64_t lean_job;
    uint32_t capture_words, direct_count;
    int lean;
} geo_job;
enum { GEO_RUN = 64 };
static geo_job g_geo_run[GEO_RUN];
// A run's accesses so far, as two sorted sets of disjoint half-open intervals: the union of its jobs' writes and of their
// reads. A job conflicts with some earlier job of the run exactly when one of its ranges meets the writes or one of its writes
// meets the reads (the pairwise rule: read/read pairs cannot communicate), and overlap with a union is overlap with one of its members,
// so merging touching or overlapping intervals loses nothing. Checking each job against every earlier one cost O(n^2) range
// scans a run: about 8 % of the submit thread in Donkey Kong Bananza (58-draw runs, hundreds of geometry draws a frame).
typedef struct { uint64_t lo[GEO_RUN * 64], hi[GEO_RUN * 64]; int n; } geo_span_set;
static int geo_span_first(const geo_span_set *s, uint64_t lo) {   // the first interval that ends after lo (branch-free halving)
    const uint64_t *h = s->hi; int base = 0, n = s->n;
    if (!n) return 0;
    while (n > 1) { int half = n / 2; base = h[base + half - 1] <= lo ? base + half : base; n -= half; }
    return base + (h[base] <= lo);
}
static int geo_span_meets(const geo_span_set *s, uint64_t lo, uint64_t hi) { int i = geo_span_first(s, lo); return i < s->n && s->lo[i] < hi; }
static void geo_span_add(geo_span_set *s, uint64_t lo, uint64_t hi) {
    int i = geo_span_first(s, lo), j = i;
    if (i < s->n && s->lo[i] <= lo && hi <= s->hi[i]) return;   // already covered: most of a run's shared reads
    if (i > 0 && s->hi[i - 1] == lo) i--;   // touching on the left
    while (j < s->n && s->lo[j] <= hi) { lo = MIN(lo, s->lo[j]); hi = MAX(hi, s->hi[j]); j++; }
    if (i < j) { lo = MIN(lo, s->lo[i]); hi = MAX(hi, s->hi[i]); }
    // replace [i, j) by the one merged interval
    int d = 1 - (j - i);
    if (d) { memmove(s->lo + j + d, s->lo + j, (size_t)(s->n - j) * 8); memmove(s->hi + j + d, s->hi + j, (size_t)(s->n - j) * 8); s->n += d; }
    s->lo[i] = lo; s->hi[i] = hi;
}
typedef struct { geo_span_set w, r; int unknown; } geo_run_access;
static int geo_run_conflict(const geo_run_access *u, const geo_access *a) {
    if (u->unknown || a->unknown) return 1;
    for (int k = 0; k < a->n; k++) {
        if (geo_span_meets(&u->w, a->r[k].lo, a->r[k].hi)) return 1;
        if ((a->writes >> k & 1) && geo_span_meets(&u->r, a->r[k].lo, a->r[k].hi)) return 1;
    }
    return 0;
}
static void geo_run_note(geo_run_access *u, const geo_access *a) {
    u->unknown |= a->unknown;
    for (int k = 0; k < a->n; k++) geo_span_add(a->writes >> k & 1 ? &u->w : &u->r, a->r[k].lo, a->r[k].hi);
}
static int g_geo_nrun;
static uint64_t g_geo_dependency_splits;
static int g_geo_query_run = -1;
static int g_geo_cut_on = -1;
static int geo_cut_on(void) { if(g_geo_cut_on<0){const char *e=getenv("DM_NVN_GEOMETRY_CUT");g_geo_cut_on=!e||atoi(e);}return g_geo_cut_on; }
static int geo_query_run(void) {
    if (g_geo_query_run < 0) { const char *e = getenv("DM_NVN_GEOMETRY_QUERY_RUN"); g_geo_query_run = !e || strcmp(e, "0"); }
    return g_geo_query_run;
}
// 1: j is ready; 0: this draw draws nothing; -1: cannot join an open run.
static int geo_prep(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0, geo_job *j, int may_wait, uint32_t primitive_base) {
    *j = (geo_job){.pr = pr, .vs = pr ? pr->sh[0] : nil, .gs = pr ? pr->sh[2] : nil,
                   .dw = dw0 ? *dw0 : (p_draw){0}, .di = di0 ? *di0 : (p_draw_idx){0}, .has_di = di0 != NULL};
    NvShader *vs = j->vs, *base = j->gs;
    if (!vs || !vs->geo_fn || !base || !base->geo_vertices || pr->sh[3] || pr->sh[4]) {
        if (!may_wait) return -1;
        static int red; if (red++ < 8) MLOG("[RED] geometry draw cannot capture this pipeline (vertex %s, geometry %s, tessellation %d, feedback %d)", vs ? vs->key : "-", base ? base->key : "-", !!(pr->sh[3] || pr->sh[4]), S.tf_on); return 0;
    }
    if (di0 && (j->di.itype < 0 || j->di.itype > 2)) { if (may_wait) MLOG("[RED] geometry index type %d is invalid", j->di.itype); return 0; }
    uint64_t ind = dw0 ? j->dw.ind : j->di.ind;
    // A run may not read indirect records or prepare an unaligned stream ahead
    // of intervening draws. The single-draw path keeps main's ordered copies.
    if (!may_wait) {
        if (ind) return -1;
        // An empty stream has nothing to prepare, a CPU fill at most (vbuf_unaligned): FINAL FANTASY XII THE ZODIAC AGE leaves one bound at address 1, size 0, which kept every one of its geometry draws out of a run
        for (int i = 0; i < 16; i++) if (S.vbuf[i].size && (S.vbuf[i].address & 3)) return -1;
    }
    int drained = g_geo_split_depth != 0;
    if (geo_conversion_hazard(vs) || (ind && geo_cpu_hazard(ind, dw0 ? 16 : 20))) {
        if (!may_wait) return -1;
        if (!geo_sync_cpu(&drained)) return 0;
    }
    if (ind) {
        uint32_t a[5] = {0}; size_t n = dw0 ? 16 : 20;
        if (!nvn_host_readable(ind, n)) { MLOG("[RED] geometry indirect arguments are not readable"); return 0; }
        memcpy(a, (void *)(uintptr_t)ind, n);
        if (!a[0] || !a[1]) return 0;
        if (a[0] > INT32_MAX || a[1] > INT32_MAX) { MLOG("[RED] geometry indirect draw exceeds the host count range"); return 0; }
        if (dw0) { j->dw.count = a[0]; j->dw.instances = a[1]; j->dw.first = a[2]; j->dw.base_instance = a[3]; j->dw.ind = j->dw.mdi_cnt = 0; }
        else { j->di.count = a[0]; j->di.instances = a[1]; j->di.ibuf += (uint64_t)a[2] << j->di.itype; j->di.base_vertex = (int32_t)a[3]; j->di.base_instance = a[4]; j->di.ind = j->di.mdi_cnt = 0; }
    }
    int prim = dw0 ? j->dw.prim : j->di.prim, topology = nvn_geo_topology(prim);
    int count = dw0 ? j->dw.count : j->di.count, instances = dw0 ? j->dw.instances : j->di.instances;
    uint32_t first_instance = (uint32_t)(dw0 ? j->dw.base_instance : j->di.base_instance);
    g_draw_binst = first_instance;
    if (count <= 0 || instances < 0) return 0;
    if (!instances) instances = 1;
    if (topology < 0 || count > (1 << 22)) { if (may_wait) MLOG("[RED] geometry primitive %d or count %d exceeds the assembly path", prim, count); return 0; }
    if (!drained && di0 && geo_index_wait(j->di.ibuf, (uint64_t)count << j->di.itype)) {
        if (!may_wait) return -1;
        if (!geo_sync_cpu(&drained)) return 0;
    }
    if (cp_on()) { if (drained) g_cp_geo_waited++; else g_cp_geo_safe++; }
    // The kernel for this topology and the reader, looked up by name once per geometry shader: building the names on every draw took about 40 % of geo_prep's time
    // in Donkey Kong Bananza (an NSString format, a substring and two dictionary lookups a draw, at hundreds of geometry draws a frame)
    if (!(base->geo_bg_tried >> topology & 1) || !(base->geo_bg_tried & 0x80)) {
        NSString *key = [[NSString stringWithUTF8String:base->key] substringToIndex:16];
        if (!(base->geo_bg_tried >> topology & 1)) { base->geo_bg_tried |= 1 << topology; base->geo_bg_v[topology] = shader_get([key stringByAppendingFormat:@"_bg%d", topology], 2); }
        if (!(base->geo_bg_tried & 0x80)) { base->geo_bg_tried |= 0x80; base->geo_bgv_v = shader_get([key stringByAppendingString:@"_bgv"], 0); }
    }
    NvShader *gs = base->geo_bg_v[topology], *reader = base->geo_bgv_v;
    if (!gs || !reader) return 0;
    j->access=geo_access_of(pr,gs);
    NvShader *lean = geo_lean_pick(pr, gs, topology, instances);
    if (lean) gs = lean;
    uint32_t record = lean ? gs->geo_words : GEO_CAPTURE_WORDS;
    id<MTLRenderPipelineState> capture = geo_capture_pso(vs); if (!capture) return 0;
    if (!gs->geo_pso) { NSError *e = nil; gs->geo_pso = [g_dev newComputePipelineStateWithFunction:gs->fn error:&e];
        if (!gs->geo_pso) { MLOG("[RED] geometry compute %s: %s", gs->key, e.localizedDescription.UTF8String); return 0; } }
    if (gs->geo_pso.maxTotalThreadsPerThreadgroup < 32) { MLOG("[RED] geometry kernel cannot run a 32-thread group"); return 0; }

    uint32_t lo = UINT32_MAX, hi = 0; size_t nu = 0;
    int restart = di0 && S.restart.on;
    int isz = di0 ? 1 << j->di.itype : 0;
    uint32_t restart_index = S.restart.index & (isz == 1 ? UINT8_MAX : isz == 2 ? UINT16_MAX : UINT32_MAX);
    if (di0 && (j->di.itype < 0 || j->di.itype > 2 || !nvn_host_readable(j->di.ibuf, (uint64_t)count * isz))) {
        if (may_wait) MLOG("[RED] geometry index buffer is not readable"); return 0;
    }
    size_t raw_n=(size_t)count*isz;
    geo_ix_key ixkey={.address=j->di.ibuf,.count=count,.type=j->di.itype,.prim=prim,.restart=restart,.restart_index=restart?restart_index:0};
    geo_ix_ent *cached=di0?geo_ix_get(&ixkey,(const void *)(uintptr_t)j->di.ibuf,raw_n):NULL;
    uint32_t *order=NULL;const uint8_t *source=(const uint8_t *)(uintptr_t)j->di.ibuf;
    int keep=di0&&geo_ix_on();
    if(cached){lo=cached->lo;hi=cached->hi;nu=cached->nu;}
    else {
      order=malloc((size_t)count*4+(keep?raw_n:0));
      if(!order){MLOG("[RED] geometry index allocation failed");return 0;}
      if(keep){uint8_t *snapshot=(uint8_t *)(order+count);memcpy(snapshot,source,raw_n);source=snapshot;}
      for (int i = 0; i < count; i++) {
        uint32_t v = (uint32_t)i;
        if (di0) { v = 0; memcpy(&v, source + (size_t)i * isz, isz); }
        order[i] = v;
        if (restart && v == restart_index) continue;
        lo = MIN(lo, v); hi = MAX(hi, v); nu++;
      }
    }
    if (!nu) { free(order); return 0; }
    int64_t first = di0 ? (int64_t)lo + j->di.base_vertex : j->dw.first;
    uint64_t span = (uint64_t)hi - lo + 1;
    if (first < 0 || first + span > (uint64_t)UINT32_MAX + 1) { free(order); if (may_wait) MLOG("[RED] geometry vertex range is outside 32-bit vertex IDs"); return 0; }
    // Normalize only after assembly, so a restart value cannot collide with a
    // normalized vertex number. Indexed captures use a sorted unique index list.
    size_t words = cached?cached->words:nvn_geo_assemble(prim, order, count, restart, restart_index, NULL, 0);
    int width = nvn_geo_width(topology);
    uint64_t np = words == SIZE_MAX ? 0 : words / width;
    if (!np) { free(order); return 0; }
    // Vertex IDs far apart: a compact capture when it saves real memory and takes few draws (nvn_geometry.h). nu counts references, so span > nu means holes for certain
    uint32_t *ids = NULL; size_t id_n = 0, id_runs = 0; uint64_t cspan = span;
    if (di0 && record && span > nu && (span - nu) * instances * record * 4 >= GEO_COMPACT_MIN) {
        uint32_t *copy = NULL; ids = malloc(nu * 4);
        if (ids && cached) { id_n = cached->unique; memcpy(ids, cached->data + raw_n + words * 4, id_n * 4); }
        else if (ids && (copy = malloc((size_t)count * 4))) { memcpy(copy, order, (size_t)count * 4); id_n = geo_capture_unique(copy, count, restart, restart_index, lo, span, ids, nu); if (id_n == SIZE_MAX) id_n = 0; }
        free(copy);
        id_runs = nvn_geo_capture_runs(ids, id_n);
        if (id_n && id_runs <= GEO_COMPACT_RUNS) cspan = id_n; else { free(ids); ids = NULL; id_runs = 0; }
    }
    uint64_t groups = np * (uint64_t)instances * gs->geo_invocations;
    if (groups > UINT32_MAX || cspan * (uint64_t)instances > (256ull << 20) / (MAX(record, 1u) * 4) ||
        (uint64_t)first_instance + instances > (uint64_t)UINT32_MAX + 1) {
        free(order); free(ids);
        if (may_wait && !geo_split(pr, dw0 ? &j->dw : NULL, dw0 ? NULL : &j->di, count, instances, first_instance, primitive_base))
            MLOG("[RED] geometry draw exceeds the capture or invocation bound");
        return may_wait ? 0 : -1;
    }
    uint32_t emits = gs->geo_index_stride ? gs->geo_vertices : 0;   // 0: the GS writes storage buffers only
    uint64_t input_size = MAX(cspan * instances * record * 4, 4ull);
    uint64_t output_size = emits ? groups * emits * gs->geo_stride * 4 : 4;
    uint64_t raw_size = emits ? groups * gs->geo_index_stride * 4 : 4;
    uint64_t packed_size = emits ? groups * emits * (gs->geo_output == 7 ? 3 : gs->geo_output == 6 ? 2 : 1) * 4 : 4;
    uint64_t count_size = emits ? (groups + (groups + 63) / 64) * 4 : 4;
    uint64_t total = input_size + output_size + raw_size + packed_size + count_size + words * 4 + nu * 4 + sizeof(nvn_geo_capture) + sizeof(nvn_geo_pack_job) + sizeof(nvn_geo_lean_job) + 32 + 20 + 15 * 255
        + (ids ? words * 4 + id_runs * (sizeof(nvn_geo_capture) + 8) : 0);
    if (groups > UINT32_MAX || total > (256ull << 20) || np > UINT32_MAX) {
        free(order); free(ids);
        if (may_wait && !geo_split(pr, dw0 ? &j->dw : NULL, dw0 ? NULL : &j->di, count, instances, first_instance, primitive_base))
            MLOG("[RED] geometry draw needs %llu MiB of scratch, above the 256 MiB bound", (unsigned long long)(total >> 20));
        return may_wait ? 0 : -1;
    }
    id<MTLBuffer> scratch; size_t off = arena_alloc((size_t)total, &scratch);
    uint8_t *cpu = (uint8_t *)scratch.contents + off; uint64_t gpu = scratch.gpuAddress + off;
    size_t used = 0;
    #define GEO_ALLOC(N, G, P) do { used = (used + 255) & ~(size_t)255; (G) = gpu + used; (P) = cpu + used; used += (size_t)(N); } while (0)
    void *pinput, *poutput, *praw, *ppacked, *pcnt, *pmap, *pcapture_ix, *pinfo, *pcapture_info, *pargs, *pjob, *plean_job;
    GEO_ALLOC(input_size, j->input, pinput); GEO_ALLOC(output_size, j->output, poutput); GEO_ALLOC(raw_size, j->raw, praw); GEO_ALLOC(packed_size, j->packed, ppacked); GEO_ALLOC(count_size, j->cnt, pcnt);
    GEO_ALLOC(words * 4, j->map, pmap); GEO_ALLOC(nu * 4, j->capture_ix, pcapture_ix); GEO_ALLOC(32, j->info, pinfo); GEO_ALLOC(sizeof(nvn_geo_capture), j->capture_info, pcapture_info);
    GEO_ALLOC(20, j->args, pargs); GEO_ALLOC(sizeof(nvn_geo_pack_job), j->job, pjob);
    GEO_ALLOC(sizeof(nvn_geo_lean_job), j->lean_job, plean_job);
    void *pcompact = NULL, *pruns = NULL; uint64_t run_gpu = 0; j->compact_map = 0; j->runs = NULL; j->nruns = 0;
    if (ids) { GEO_ALLOC(words * 4, j->compact_map, pcompact); GEO_ALLOC(id_runs * 8, run_gpu, pruns); GEO_ALLOC(id_runs * sizeof(nvn_geo_capture), j->capture_info, pcapture_info); }
    #undef GEO_ALLOC
    (void)pinput; (void)poutput; (void)praw; (void)ppacked; (void)pcnt;
    if (used > total) MDIE("geometry scratch size accounting");
    size_t unique_count = 0;
    if(cached){memcpy(pmap,cached->data+raw_n,words*4);unique_count=cached->unique;memcpy(pcapture_ix,cached->data+raw_n+words*4,unique_count*4);}
    else {
      nvn_geo_assemble(prim, order, count, restart, restart_index, pmap, words);
      for (size_t i = 0; i < words; i++) ((uint32_t *)pmap)[i] -= lo;
      if (di0) {
        unique_count = geo_capture_unique(order, count, restart, restart_index, lo, span, pcapture_ix, nu);
        if (unique_count == SIZE_MAX) { free(order); free(ids); MLOG("[RED] geometry capture indices exceed their range or allocation"); return 0; }
      }
      if(keep)geo_ix_put(&ixkey,source,raw_n,lo,hi,nu,pmap,words,pcapture_ix,unique_count);
    }
    free(order);
    // geometry_counts.y is the guest primitive base, independent of local maps.
    uint32_t vi[8] = {(uint32_t)cspan, (uint32_t)instances, 0, 0, (uint32_t)np, primitive_base, 0, 0}; memcpy(pinfo, vi, 32);
    nvn_geo_capture capture_info = {j->input, 0, {(uint32_t)first, (uint32_t)cspan, first_instance, (uint32_t)instances}, MAX(record, 1u), {0}};
    for (int k = 0; k < GEO_CAPTURE_WORDS; k++) capture_info.slot[k] = lean ? gs->geo_slot[k] : (int16_t)k;
    if (ids) {   // a capture_info per run of IDs: a record's address is (vertex ID - info[0]) words on, so info[0] = the run's first ID less its position (in vertex IDs, with the base vertex)
        if (!nvn_geo_compact_map(pmap, words, lo, ids, id_n, pcompact)) MDIE("geometry capture: a mapped vertex is not in the unique list");
        uint32_t *rp = pruns; nvn_geo_capture *cp = pcapture_info; size_t r = 0;
        for (size_t i = 0; i < id_n; r++) { size_t e = i + 1; while (e < id_n && ids[e] == ids[e - 1] + 1) e++;
            rp[2 * r] = (uint32_t)i; rp[2 * r + 1] = (uint32_t)(e - i); capture_info.info[0] = (uint32_t)((int64_t)first + (ids[i] - lo) - (int64_t)i); cp[r] = capture_info; i = e; }
        j->runs = rp; j->nruns = r;
        static int nl; if (nl++ < 4) MLOG("geometry capture of %zu vertices in %zu runs over a range of %llu: %llu KB instead of %llu MB", id_n, r, (unsigned long long)span, (unsigned long long)(input_size >> 10), (unsigned long long)((span * instances * record * 4) >> 20));
        free(ids); ids = NULL;
    } else
    memcpy(pcapture_info, &capture_info, sizeof capture_info);
    memset(pargs, 0, 20);
    nvn_geo_pack_job pj = {j->raw, j->packed, j->args, (uint32_t)groups, gs->geo_index_stride, gs->geo_output, (uint32_t)(packed_size / 4), j->cnt};
    memcpy(pjob, &pj, sizeof pj);
    nvn_geo_lean_job lj = {j->raw, j->output, j->map, j->packed, j->args, (uint32_t)groups, gs->geo_index_stride,
                          (uint32_t)first, first_instance, (uint32_t)(packed_size / 4), (uint32_t)first};
    memcpy(plean_job, &lj, sizeof lj);
    j->lean = lean != nil; j->capture_words = record;
    j->direct_count = lean && topology == 3 ? (uint32_t)(np * 3) : 0;
    j->gs = gs; j->reader = lean ? vs : reader; j->capture = capture; j->dw.instances = instances; j->di.instances = instances;
    j->vertex_range = (p_draw){.first = (int32_t)first, .count = (int32_t)span, .instances = instances, .base_instance = (int32_t)first_instance};
    j->first_instance = first_instance; j->groups = (uint32_t)groups; j->emits = emits; j->np = np; j->unique_count = unique_count; j->nu = nu; j->packed_size = packed_size;
    for (int i = 0; i < gs->rf.nb; i++) {
        const nvn_rbind *b = &gs->rf.b[i];
        if (b->kind == NVN_RB_SBUFP && b->rw && b->reg < 16 && S.ssbo[2][b->reg].addr)
            sbw_note(S.ssbo[2][b->reg].addr, S.ssbo[2][b->reg].size, SBW_GPU);
    }
    if (may_wait) vbuf_prepare(NULL, NULL, NULL);   // the run's captures bind later (geo_capture_one), so their streams are not bounded by their draws' ranges
    return 1;
}
static void geo_capture_one(id<MTL4RenderCommandEncoder> re, const geo_job *j) {
    if (!j->capture_words) return;
    g_draw_binst = j->first_instance;
    memset(g_vbuf_need, 0, sizeof g_vbuf_need);   // a later draw's bounds do not apply here (geo_prepare)
    enc_r_pso(re, j->capture); bind_vbufs(re);
    vconv_bind(re, j->vs, 1, &j->vertex_range, NULL);
    args_dirty_all(); encode_args(re, j->vs, 0, 1); enc_r_vbuf(re, 13, j->capture_info);
    if (j->nruns) for (size_t r = 0; r < j->nruns; r++) {
        if (r) enc_r_vbuf(re, 13, j->capture_info + r * sizeof(nvn_geo_capture));
        enc_r_draw_indexed(re, MTLPrimitiveTypePoint, j->runs[2 * r + 1], MTLIndexTypeUInt32, j->capture_ix + (uint64_t)j->runs[2 * r] * 4, (j->nu - j->runs[2 * r]) * 4, j->di.instances, j->di.base_vertex, j->first_instance);
    }
    else if (j->has_di) enc_r_draw_indexed(re, MTLPrimitiveTypePoint, j->unique_count, MTLIndexTypeUInt32, j->capture_ix, j->nu * 4, j->di.instances, j->di.base_vertex, j->first_instance);
    else enc_r_draw_instanced(re, MTLPrimitiveTypePoint, j->dw.first, j->dw.count, j->dw.instances, j->first_instance);
}
static void geo_gs_one(id<MTL4ComputeCommandEncoder> ce, const geo_job *j) {
    g_draw_binst = j->first_instance;
    g_geo_bind[0] = j->input; g_geo_bind[1] = j->output; g_geo_bind[2] = j->raw; g_geo_bind[3] = j->info; g_geo_bind[4] = j->compact_map ? j->compact_map : j->map;
    args_dirty_all(); encode_args(ce, j->gs, 2, 0); enc_c_pso(ce, j->gs->geo_pso);
    enc_c_dispatch(ce, MTLSizeMake((j->np + 31) / 32, j->dw.instances, j->gs->geo_invocations), MTLSizeMake(32, 1, 1));
    if (j->lean) {
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_args(ce, j->lean_job, 0); enc_c_pso(ce, g_geo_lean_k[j->direct_count ? 2 : 0]);
        enc_c_dispatch(ce, MTLSizeMake((j->groups + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
    }
    // its guest storage writes are noted (geo_prep, encode_args), so a drain's list of later writes holds them (g_wr_geo, drw_valid)
    swrite_note(j->pr); udc_note_draw(j->pr, 0); g_wr_epoch++; g_wr_geo++; g_mdx_epoch++;
}
static void geo_raster(const geo_job *j, int last_in_run) {
    g_geo_bind[1] = j->output;
    if (S.tf_on) tf_geometry(j->reader, j->packed, j->args, j->gs->geo_output == 1 ? 1 : j->gs->geo_output == 6 ? 2 : 3, j->first_instance);
    if (S.rast_discard || !j->emits) { if (last_in_run && g_enc_geo_nocolor) rp_end(); return; }
    NvProg *pr = j->pr, *raster = pr->geo_raster;
    if (!raster || raster->sh[0] != j->reader || raster->sh[1] != pr->sh[1]) {
        raster = [NvProg new]; raster->sh[0] = j->reader; raster->sh[1] = pr->sh[1]; pr->geo_raster = raster;
    }
    g_geo_vrange = j->lean ? &j->vertex_range : NULL; g_geo_count = j->direct_count;
    g_geo_program = raster; g_geo_indices = j->packed; g_geo_index_bytes = j->packed_size; g_geo_args = j->args;
    g_geo_omit_colors = !(j->reader->layer_out) && geo_color_omit();
    if (g_geo_omit_colors) g_geo_color_omits++;
    args_dirty_all();
    p_draw synthetic = {.prim = j->gs->geo_output == 1 ? 0 : j->gs->geo_output == 6 ? 1 : 4, .count = 3, .instances = 1, .base_instance = (int32_t)j->first_instance};
    draw(&synthetic, NULL);
    g_geo_omit_colors = 0; g_geo_program = nil; g_geo_vrange = NULL; g_geo_count = 0; args_dirty_all();
    // Ordinary draws may enable color writes, so they must open their own pass.
    if (last_in_run && g_enc_geo_nocolor) rp_end();
    static int once; if (!once++) MLOG("geometry compute: vertex %s, geometry %s, %llu input primitives, %d instances", j->vs->key, j->gs->key, (unsigned long long)j->np, j->dw.instances);
}
// Query reports/resets remain at their original places in raster replay. Each
// queried draw selects a separate visibility slot in the shared depth pass;
// vertex capture and GS compute do not contribute raster samples to a query.
static int geo_run_passes(uint32_t op, const void *a) {
    // Feedback offsets and saved counts change at their original replay point.
    if (op == OP_TF_STATE || op == OP_TF_BIND) return 0;
    if (op == OP_REPORT_COUNTER) {
        uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8);
        if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED && !geo_query_run()) return 0;
        sbw_note(ad, 16, SBW_REPORT); return 1;
    }
    if (op == OP_RESET_COUNTER) { uint64_t t; memcpy(&t, a, 8); return t != NVN_COUNTER_TYPE_SAMPLES_PASSED || geo_query_run(); }
    if (op == OP_SIGNAL_EVENT) return 1;
    return state_op(op, a);
}
static int geo_run_walk(id<MTL4RenderCommandEncoder> re) {
    int n = 1;
    if (!geo_hoist_on() || S.tf_on || g_geo_split_depth || g_defer_replaying || !g_exec_next || g_ret_n > 8) return n;
    typeof(S) s0 = S; typeof(R) r0 = R; int pd = g_pso_dirty, dd = g_dss_dirty;
    fz_cursor cur = {.p = g_exec_next, .ret_n = g_ret_n}; memcpy(cur.ret, g_ret, sizeof(*g_ret) * (size_t)g_ret_n);
    const uint8_t *p; op_hdr h; const void *a;
    static geo_run_access run;   // 128 KiB: not on the stack
    run.w.n = run.r.n = run.unknown = 0; geo_run_note(&run, &g_geo_run[0].access);
    for (int guard = 0; guard < 10000 && n < GEO_RUN && fz_next(&cur, &p, &h, &a); guard++) {
        if (geo_run_passes(h.op, a)) continue;
        if (h.op != OP_DRAW && h.op != OP_DRAW_IDX) break;
        p_draw dt; p_draw_idx it;
        const p_draw *dw = h.op == OP_DRAW ? draw_rec(a, h.size, &dt) : NULL;
        const p_draw_idx *di = h.op == OP_DRAW_IDX ? draw_idx_rec(a, h.size, &it) : NULL;
        if ((dw && dw->mdi_cnt) || (di && di->mdi_cnt)) break;
        NvProg *pr = S.prog ? prog_get(S.prog) : nil;
        if (!pr || !pr->sh[2] || !pr->sh[2]->geo_vertices) break;
        if (geo_mesh_fn(pr, dw, di)) break;
        int r = geo_prep(pr, dw, di, &g_geo_run[n], 0, 0);
        if (r <= 0) break;
        if (geo_run_conflict(&run, &g_geo_run[n].access)) { g_geo_dependency_splits++; break; }
        geo_run_note(&run, &g_geo_run[n].access);
        g_geo_run[n].op = p;
        geo_capture_one(re, &g_geo_run[n]);
        n++;
    }
    S = s0; R = r0; g_pso_dirty = pd; g_dss_dirty = dd;
    return n;
}
static void geo_draw(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0, uint32_t primitive_base) { cp_stamp start = cp_begin(); geo_draw_(pr, dw0, di0, primitive_base); cp_end(6, start); }
static void geo_draw_(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0, uint32_t primitive_base) {
    for (int i = 1; i < g_geo_nrun; i++) if (g_geo_run[i].op && g_geo_run[i].op == g_exec_cur) {
        g_geo_run[i].op = NULL; geo_raster(&g_geo_run[i], i == g_geo_nrun - 1); return;
    }
    g_geo_nrun = 0;
    if (!g_geo_split_depth && geo_mesh_draw(pr, dw0, di0)) return;
    // Land deferred commands before checking their writes against the CPU read.
    if (defer_has_active()) defer_fallback();
    lc_land(); rp_flush(); rp_end(); ubd_land();
    // This is already an encoder boundary. Start earlier GPU work before the
    // CPU prepares the next run; retain sub_cut's deferral/fence safety gates.
    if (geo_cut_on()) sub_cut();
    geo_job *j0 = &g_geo_run[0];
    if (geo_prep(pr, dw0, di0, j0, 1, primitive_base) <= 0) return;
    static id<MTLComputePipelineState> pack[5];   // count, parallel scan, write, fused small draw, legacy scan
    if (!pack[4]) { NSError *e = nil; id<MTLLibrary> lib = host_lib("geo_pack", &e);
        const char *fn[5] = {"geo_pack_count", "geo_pack_scan", "geo_pack_write", "geo_pack_small", "geo_pack_scan_legacy"};
        for (int i = 0; i < 5 && lib; i++) if (!(pack[i] = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(fn[i])] error:&e])) break;
        if (!pack[4]) { for (int i = 0; i < 5; i++) pack[i] = nil; MLOG("[RED] geometry index compaction: %s", e.localizedDescription.UTF8String); return; } }
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    if (g_geo_split_compute < 0) { const char *e = getenv("DM_NVN_GEOMETRY_COMPUTE"); g_geo_split_compute = e && !strcmp(e, "split"); }
    MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new]; rd.renderTargetWidth = rd.renderTargetHeight = rd.defaultRasterSampleCount = 1;
    id<MTL4RenderCommandEncoder> re = enc_r_begin(rd);
    geo_capture_one(re, j0);
    int n = g_geo_nrun = geo_run_walk(re);
    if (n > 1) { g_geo_hoisted += n - 1; g_geo_hoist_runs++; }
    enc_r_end(re);
    typeof(S) s0 = S; typeof(R) r0 = R; int pd = g_pso_dirty, dd = g_dss_dirty;
    fz_cursor cur = {.p = g_exec_next, .ret_n = g_ret_n};
    if (n > 1) memcpy(cur.ret, g_ret, sizeof(*g_ret) * (size_t)g_ret_n);
    id<MTL4ComputeCommandEncoder> ce = nil;
    int packs = 0;
    for (int i = 0; i < n; i++) {
        if (i) {
            const uint8_t *p; op_hdr h; const void *a;
            while (fz_next(&cur, &p, &h, &a) && p != g_geo_run[i].op) state_op(h.op, a);
        }
        if (!ce) ce = enc_c_begin(NULL);
        geo_gs_one(ce, &g_geo_run[i]);
        if (g_geo_split_compute) { enc_c_end(ce); ce = nil; }
        packs += g_geo_run[i].emits != 0 && !g_geo_run[i].direct_count;
    }
    if (n > 1) { S = s0; R = r0; g_pso_dirty = pd; g_dss_dirty = dd; }
    int legacy = geo_pack_legacy(), small = 0;
    for (int i = 0; i < n; i++) if (!legacy && g_geo_run[i].emits && !g_geo_run[i].direct_count && g_geo_run[i].groups <= 1024) small++;
    if (small) {
        if (!ce) ce = enc_c_begin(NULL);
        else [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_pso(ce, pack[3]);
        for (int i = 0; i < n; i++) if (g_geo_run[i].emits && !g_geo_run[i].direct_count && g_geo_run[i].groups <= 1024) {
            enc_c_args(ce, g_geo_run[i].job, 0); enc_c_dispatch(ce, MTLSizeMake(1, 1, 1), MTLSizeMake(128, 1, 1)); g_geo_pack_small++;
        }
        if (g_geo_split_compute) { enc_c_end(ce); ce = nil; }
    }
    if (packs > small) for (int stage = 0; stage < 3; stage++) {
        if (!ce) ce = enc_c_begin(NULL);
        else [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_pso(ce, pack[stage == 1 && legacy ? 4 : stage]);
        for (int i = 0; i < n; i++) if (g_geo_run[i].emits && !g_geo_run[i].direct_count && (legacy || g_geo_run[i].groups > 1024)) {
            enc_c_args(ce, g_geo_run[i].job, 0);
            enc_c_dispatch(ce, stage == 1 ? MTLSizeMake(1, 1, 1) : MTLSizeMake((g_geo_run[i].groups + 63) / 64, 1, 1), MTLSizeMake(stage == 1 ? (legacy ? 1 : 128) : 64, 1, 1));
            if (!stage) g_geo_pack_large++;
        }
        if (g_geo_split_compute) { enc_c_end(ce); ce = nil; }
    }
    for (int i = 0; i < n; i++) if (g_geo_run[i].lean && !g_geo_run[i].direct_count) {
        if (!ce) ce = enc_c_begin(NULL);
        else [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_pso(ce, g_geo_lean_k[1]); enc_c_args(ce, g_geo_run[i].lean_job, 0);
        enc_c_dispatch(ce, MTLSizeMake((g_geo_run[i].packed_size / 4 + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
    }
    if (ce) enc_c_end(ce);
    args_dirty_all(); g_enc_pso = nil; g_enc_dss = nil; g_ed.ok = 0;
    geo_raster(j0, n == 1);
}
