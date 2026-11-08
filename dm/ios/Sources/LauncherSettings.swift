// Launcher settings, validation and persistence. Shared by the iOS UI, the Mac app's settings window (dm/mac/Launcher.swift) and the native storage regression test.
import Foundation
import Combine

struct Setting: Decodable, Identifiable {
    let env, kind, label, note, def: String
    let section: String?   // heading on the settings screen; absent in older bundles
    let needs: String?     // "<VAR>=<value>": a sub-setting, drawn only while that setting has that value
    let level: String?     // "basic": shown first (the Mac window's Basic settings); any other or none: Advanced settings. The iPhone keeps one list
    let base: Base?        // render scale: the 3D views' size at 1.0, docked and handheld, for the size in pixels (the Mac window)
    struct Base: Decodable { let docked, handheld: [Int] }
    let options: [[String]]?   // choice and preset: [value, label]
    let presets: [String: [String: String]]?   // preset: by option value, the values it gives other rows (the option's third part, {env: value}); Custom's is empty
    let min, max, step: Double?   // range
    let on, off: String?   // toggle: the values the variable takes (1 / 0 when not given)
    var id: String { env }
    var sectionTitle: String { section ?? "Game" }
    var basic: Bool { level == "basic" }
    var isPreset: Bool { kind == "preset" }   // a row of other rows' values (LauncherModel preset): shown, never saved or put in the environment
    var onValue: String { on ?? "1" }
    var offValue: String { off ?? "0" }
    func isDefault(_ v: String) -> Bool {   // a range compares as numbers (a slider writes 1.00 for 1.0)
        kind == "range" ? abs((Double(v) ?? .nan) - (Double(def) ?? .nan)) < 1e-6 : v == def
    }
    func accepts(_ v: String) -> Bool {   // a saved choice the game still offers
        switch kind {
        case "choice", "preset": return options?.contains { $0[0] == v } ?? false
        case "range":
            guard !v.contains(where: { $0 == "x" || $0 == "X" }), let n = Double(v), n.isFinite, let lo = min, let hi = max else { return false }   // no hex: launcher.c's check agrees
            return n >= lo && n <= hi
        case "toggle": return v == onValue || v == offValue
        default: return true
        }
    }
    enum CodingKeys: String, CodingKey { case env, kind, label, note, def = "default", section, needs, level, base, options, min, max, step, on, off }
    // An option is [value, label], or [value, label, {env: value}] in a preset row: the parts are decoded one by one, since the third is a map
    private enum OptionPart: Decodable {
        case text(String), values([String: String])
        init(from d: Decoder) throws {
            let c = try d.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .text(s) } else { self = .values(try c.decode([String: String].self)) }
        }
    }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        env = try c.decode(String.self, forKey: .env); kind = try c.decode(String.self, forKey: .kind); label = try c.decode(String.self, forKey: .label)
        note = try c.decode(String.self, forKey: .note); def = try c.decode(String.self, forKey: .def)
        section = try c.decodeIfPresent(String.self, forKey: .section); needs = try c.decodeIfPresent(String.self, forKey: .needs); level = try c.decodeIfPresent(String.self, forKey: .level)
        base = try c.decodeIfPresent(Base.self, forKey: .base)
        min = try c.decodeIfPresent(Double.self, forKey: .min); max = try c.decodeIfPresent(Double.self, forKey: .max); step = try c.decodeIfPresent(Double.self, forKey: .step)
        on = try c.decodeIfPresent(String.self, forKey: .on); off = try c.decodeIfPresent(String.self, forKey: .off)
        let parts = try c.decodeIfPresent([[OptionPart]].self, forKey: .options)
        options = parts?.map { $0.compactMap { if case .text(let s) = $0 { return s } else { return nil } } }
        presets = kind == "preset" ? parts.map { Dictionary($0.compactMap { o -> (String, [String: String])? in
            guard case .text(let v)? = o.first else { return nil }
            return (v, o.compactMap { if case .values(let m) = $0 { return m } else { return nil } }.first ?? [:])
        }, uniquingKeysWith: { a, _ in a }) } : nil
    }
}
struct SettingsFile: Decodable { let settings: [Setting] }

final class LauncherModel: ObservableObject {
    let settings: [Setting]
    let fixed: Set<String>      // set by an env file or by the launch environment: shown, not changed
    let fromLaunch: Set<String>   // of those, the ones the launch environment set (a front end's launch overrides, Xcode; DM_ENV_LAUNCH_KEYS from launcher.c or main.m)
    let ignoreSaved: Bool   // DM_SAVED_CHOICES=0: start from the defaults, and leave the saved choices as they are
    @Published var values: [String: String] = [:]

    private let defaults: UserDefaults

    init(_ settings: [Setting], defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.settings = settings
        let env = ProcessInfo.processInfo.environment
        fromLaunch = Set((env["DM_ENV_LAUNCH_KEYS"] ?? "").split(separator: " ").map(String.init))
        fixed = fromLaunch.union((env["DM_ENV_FILE_KEYS"] ?? "").split(separator: " ").map(String.init))
        ignoreSaved = env["DM_SAVED_CHOICES"] == "0"
        if ignoreSaved { fputs("launcher: saved choices ignored (DM_SAVED_CHOICES=0)\n", stderr) }
        for s in settings where !s.isPreset {   // a preset row has no value of its own (preset below)
            if fixed.contains(s.env), let v = env[s.env] { values[s.env] = v; continue }
            let saved = ignoreSaved ? nil : defaults.object(forKey: "launcher." + s.env) as? String   // a string, as launcher.c reads it: not string(forKey:), which turns a number into one
            values[s.env] = saved.flatMap { s.accepts($0) ? $0 : nil } ?? s.def   // a saved choice the game no longer offers: its default
        }
    }

    var sections: [String] {
        settings.reduce(into: []) { if !$0.contains($1.sectionTitle) { $0.append($1.sectionTitle) } }
    }
    // A row with "needs" is a sub-setting of the row it names: drawn only while that setting has that value, so a
    // setting that cannot do anything is not offered (an upscaler's detail with the upscaler off, rumble strength
    // with rumble off). A row naming a setting this game does not have stays drawn, as it has nothing to hide behind
    func visible(_ s: Setting) -> Bool {
        guard let n = s.needs, let eq = n.firstIndex(of: "=") else { return true }
        let key = String(n[n.startIndex..<eq]), want = String(n[n.index(after: eq)...])
        guard let v = values[key] else { return true }
        return v == want
    }
    func rows(in section: String) -> [Setting] { settings.filter { $0.sectionTitle == section && visible($0) } }
    // The Mac window: Basic settings first, then Advanced settings, each in the list's sections and order
    func sections(basic: Bool) -> [String] {
        settings.filter { $0.basic == basic }.reduce(into: []) { if !$0.contains($1.sectionTitle) { $0.append($1.sectionTitle) } }
    }
    func rows(in section: String, basic: Bool) -> [Setting] { rows(in: section).filter { $0.basic == basic } }
    // A range's label: its percentage, with the size in pixels for the chosen console mode when the list has the base size (the render scale)
    func rangeLabel(_ s: Setting, _ v: Double) -> String {
        let p = "\(Int((v * 100).rounded())) %"
        guard let b = s.base else { return p }
        let size = (values["DM_OPMODE"] ?? "1") == "0" ? b.handheld : b.docked
        guard size.count == 2 else { return p }
        return "\(Int((Double(size[0]) * v).rounded())) × \(Int((Double(size[1]) * v).rounded())) (\(p))"
    }
    var changed: Bool { settings.contains { !$0.isPreset && !fixed.contains($0.env) && !$0.isDefault(values[$0.env] ?? $0.def) } }

    // Presets (kind preset): a row whose options each carry values for other rows (the option's third part: {env: value}). Choosing one puts those values into the rows, which are
    // saved as if chosen there; the row itself is never saved or put in the environment. It shows the preset whose values the rows have now, or the option without values (Custom)
    // when none does: so a row changed by hand, on this screen or in the in-game menu, shows Custom, and a new build whose defaults are a preset's values shows that preset. A value
    // for a row this list lacks is ignored; a row an env file or the launch fixed keeps its value, and the preset then shows Custom
    func preset(_ s: Setting) -> String {
        let opts = s.options ?? []
        for o in opts {
            let want = (s.presets?[o[0]] ?? [:]).filter { k, _ in settings.contains { $0.env == k } }
            if !want.isEmpty && want.allSatisfy({ values[$0.key] == $0.value }) { return o[0] }
        }
        return opts.first { (s.presets?[$0[0]] ?? [:]).isEmpty }?[0] ?? ""
    }
    func choosePreset(_ s: Setting, _ v: String) {
        guard let want = s.presets?[v], !want.isEmpty else { return }   // Custom: the rows stay as they are
        for row in settings where !fixed.contains(row.env) { if let x = want[row.env], row.accepts(x) { values[row.env] = x } }
        fputs("launcher: preset \(v)\n", stderr)
    }

    // Reset every launcher choice to the game's defaults on the screen; values fixed by an env file or at launch stay. The saved choices are forgotten when Play applies
    // the defaults (apply() keeps no default), so Quit after Reset keeps them, and a choice a launch override hides is kept
    func reset() {
        for s in settings where !fixed.contains(s.env) && !s.isPreset { values[s.env] = s.def }
        fputs("launcher: settings reset to the game's defaults\n", stderr)
    }

    // Before the game starts: every setting not set by an env file goes into the environment (DM_LANG replaces the packed language main.m set as a default). Only a choice that
    // differs from the game's default is stored: a setting the player left at its default follows the defaults of a later build (the settings it was tested with)
    func apply() {
        for s in settings where s.isPreset { fputs("launcher: preset \(preset(s))\n", stderr) }   // the rows below carry its values
        for s in settings where !s.isPreset {
            let v = values[s.env] ?? s.def
            if fixed.contains(s.env) { fputs("launcher: \(s.env) kept at \(v) (\(fromLaunch.contains(s.env) ? "launch environment" : "env file"))\n", stderr); continue }
            store(s, v)
            setenv(s.env, v, 1); fputs("launcher: \(s.env)=\(v)\(s.isDefault(v) ? "" : " (changed)")\n", stderr)
        }
    }
    private func store(_ s: Setting, _ v: String) {
        if !ignoreSaved { if s.isDefault(v) { defaults.removeObject(forKey: "launcher." + s.env) } else { defaults.set(v, forKey: "launcher." + s.env) } }
    }

    // A choice made while the game runs (the iOS in-game menu, GameMenu.swift): saved as Play saves it, so the settings screen shows it at the next start, and put in the environment.
    // A value an env file or the launch fixed stays
    func choose(_ s: Setting, _ v: String) {
        if s.isPreset { for (k, x) in s.presets?[v] ?? [:] { if let row = settings.first(where: { $0.env == k }) { choose(row, x) } }; return }   // each row saved and set as its own choice
        guard !fixed.contains(s.env), s.accepts(v) else { return }
        values[s.env] = v; store(s, v); setenv(s.env, v, 1)
        fputs("launcher: \(s.env)=\(v) (in-game menu)\n", stderr)
    }

    // DM_LAUNCHER=0: no screen, but every row still goes into the environment as Play would put it: the saved choices (only DM_SAVED_CHOICES=0 ignores them) and the list's
    // defaults, which may differ from the host's own (a build whose launcher.json names the device's tested values: Ocarina of Time's Frame rate 0 over the profile's 60; without
    // the export the host ran its frame step at 60 and the real-speed path at target 0 never ran). Nothing is written
    func applySaved() {
        for s in settings where !fixed.contains(s.env) && !s.isPreset {
            let v = values[s.env] ?? s.def
            setenv(s.env, v, 1); fputs("launcher: \(s.env)=\(v) (\(s.isDefault(v) ? "default" : "saved choice"))\n", stderr)
        }
    }
}
