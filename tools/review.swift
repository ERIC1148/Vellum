// Vellum.app · 扫描并调整
//
//   Vellum.app/Contents/MacOS/Vellum 图片...    （一般由 `vellum --review` 调起）
//
// 左：页面条（拖动排序、添加、删除）；中：原图 + 可拖动四角（拖动时出现放大镜）；右：实时预览；
// 底部：液态玻璃工具栏（模式、墨色、书页展平、旋转、四角、导出）。
// 图像处理全部交给 vellum.py：--inspect 取四角和模式，--plan 渲染预览、导出 PDF/图片、识别文字。
//
// 编译：vellum.py 首次使用时自动 swiftc -O -swift-version 5 review.swift，并打包成 bin/Vellum.app

import AppKit
import SwiftUI
import UniformTypeIdentifiers
import UserNotifications

// MARK: - 路径与图标

let toolDir: URL = {
    let exe = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
    var dir = exe.resolvingSymlinksInPath().deletingLastPathComponent()
    for _ in 0..<6 {
        if FileManager.default.fileExists(atPath: dir.appendingPathComponent("vellum.py").path) { return dir }
        dir = dir.deletingLastPathComponent()
    }
    return dir
}()
let vellumPy = toolDir.appendingPathComponent("vellum.py").path
let python = "/usr/bin/python3"

var iconCache: [String: NSImage] = [:]
/// assets/icons 里的 SVG，作为模板图（系统按前景色着色）
func icon(_ name: String, _ size: CGFloat = 20) -> some View {
    let img: NSImage = {
        if let i = iconCache[name] { return i }
        let i = NSImage(contentsOf: toolDir.appendingPathComponent("assets/icons/\(name).svg")) ?? NSImage()
        i.isTemplate = true
        iconCache[name] = i
        return i
    }()
    return Image(nsImage: img).renderingMode(.template).resizable().interpolation(.high)
        .frame(width: size, height: size)
}

// MARK: - 数据

struct PageInfo: Codable {
    let file: String
    let preview: String
    let width: Int
    let height: Int
    let corners: [[Double]]?
    let conf: Double
    let mode: String
}

let fullFrame = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]

struct ModeDef { let id: String; let name: String; let icon: String; let hint: String }
let modes: [ModeDef] = [
    ModeDef(id: "auto", name: "自动", icon: "mode-auto", hint: "按内容自动选择"),
    ModeDef(id: "doc", name: "文档", icon: "mode-doc", hint: "白底黑字，最清晰"),
    ModeDef(id: "color", name: "彩色", icon: "mode-color", hint: "保留印章、彩笔和标志"),
    ModeDef(id: "notes", name: "笔记", icon: "mode-notes", hint: "调色板压缩，体积小、保留彩笔"),
    ModeDef(id: "photo", name: "证件与奖状", icon: "mode-photo", hint: "保留底纹和花边"),
    ModeDef(id: "slide", name: "演示文稿", icon: "mode-slide", hint: "拍屏幕或投影，去摩尔纹"),
    ModeDef(id: "bw", name: "黑白", icon: "mode-bw", hint: "纯黑白，体积最小"),
    ModeDef(id: "none", name: "原图", icon: "mode-none", hint: "只裁切矫正，不调色"),
]
func modeDef(_ id: String) -> ModeDef { modes.first { $0.id == id } ?? modes[0] }
let inks: [(Int, String)] = [(0, "原样"), (1, "标准"), (2, "浓郁"), (3, "加粗")]
let accent = Color(red: 0.08, green: 0.48, blue: 1.0)

final class PageState: ObservableObject, Identifiable {
    let id = UUID()
    let info: PageInfo
    let image: NSImage
    let detected: [CGPoint]?
    @Published var corners: [CGPoint]
    @Published var mode = "auto"
    @Published var ink = 2
    @Published var rotate = 0
    @Published var book = false
    @Published var preview: NSImage?

    init(info: PageInfo) {
        self.info = info
        image = NSImage(contentsOfFile: info.preview) ?? NSImage()
        detected = info.corners?.map { CGPoint(x: $0[0], y: $0[1]) }
        corners = detected ?? fullFrame
    }

    var aspect: CGFloat { CGFloat(info.width) / CGFloat(max(info.height, 1)) }
    var effectiveMode: String { mode == "auto" ? info.mode : mode }
    var inkApplies: Bool { ["doc", "color", "notes", "bw"].contains(effectiveMode) }

    var plan: [String: Any] {
        ["file": info.file, "mode": mode, "ink": ink, "rotate": rotate, "book": book,
         "corners": corners.map { [Double($0.x), Double($0.y)] }]
    }
}

final class Model: ObservableObject {
    @Published var pages: [PageState] = []
    @Published var selected: UUID?
    @Published var status = "正在识别纸张边缘…"
    @Published var rendering = false
    @Published var busy = false
    @Published var ocr = true
    @Published var dragging: UUID?
    @Published var hovering: UUID?
    @Published var dropTargeted = false

    let tmp: URL
    private var previewProc: Process?
    private var debounce: DispatchWorkItem?
    private var token = 0
    private var loads = 0

    init() {
        tmp = FileManager.default.temporaryDirectory.appendingPathComponent("vellum-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    var page: PageState? { pages.first { $0.id == selected } ?? pages.first }
    var index: Int { pages.firstIndex { $0.id == page?.id } ?? 0 }

    // MARK: 载入 / 添加 / 删除 / 排序

    func load(_ files: [String]) {
        guard !files.isEmpty else { return }
        loads += 1
        let dir = tmp.appendingPathComponent("inspect-\(loads)")
        busy = true
        status = "正在识别纸张边缘…"
        DispatchQueue.global(qos: .userInitiated).async {
            let (code, out, err) = Model.run([vellumPy, "--inspect", dir.path] + files)
            DispatchQueue.main.async {
                self.busy = false
                guard code == 0, let infos = try? JSONDecoder().decode([PageInfo].self, from: out) else {
                    self.status = "无法读取图片：" + (String(data: err, encoding: .utf8) ?? "").suffix(160)
                    return
                }
                let new = infos.map(PageState.init)
                withAnimation(.spring(duration: 0.35)) { self.pages += new }
                if self.selected == nil { self.selected = new.first?.id }
                self.refreshStatus()
                self.requestPreview(now: true)
            }
        }
    }

    func refreshStatus() {
        let missing = pages.filter { $0.detected == nil }.count
        if pages.isEmpty { status = "把照片拖到这里，或点“添加页面”"; return }
        status = missing > 0 ? "有 \(missing) 页未识别到纸张边缘，请拖动四角框选"
                             : "已识别 \(pages.count) 页。拖动角点微调，拖动缩略图排序"
    }

    func addPages() {
        let panel = NSOpenPanel()
        panel.title = "添加页面"
        panel.prompt = "添加"
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image]
        if let win = NSApp.keyWindow {
            panel.beginSheetModal(for: win) { r in if r == .OK { self.load(panel.urls.map(\.path)) } }
        } else if panel.runModal() == .OK { load(panel.urls.map(\.path)) }
    }

    func delete(_ p: PageState) {
        guard let i = pages.firstIndex(where: { $0.id == p.id }) else { return }
        withAnimation(.spring(duration: 0.3)) {
            pages.remove(at: i)
            if selected == p.id { selected = pages.isEmpty ? nil : pages[min(i, pages.count - 1)].id }
        }
        refreshStatus()
        if page?.preview == nil { requestPreview(now: true) }
    }

    func move(_ dragged: UUID, before target: UUID) {
        guard dragged != target, let from = pages.firstIndex(where: { $0.id == dragged }),
              let to = pages.firstIndex(where: { $0.id == target }) else { return }
        withAnimation(.spring(duration: 0.25)) {
            pages.move(fromOffsets: IndexSet(integer: from), toOffset: to > from ? to + 1 : to)
        }
    }

    func select(_ id: UUID) {
        guard id != selected else { return }
        selected = id
        if page?.preview == nil { requestPreview(now: true) }
    }

    func step(_ d: Int) {
        let i = index + d
        if pages.indices.contains(i) { select(pages[i].id) }
    }

    // MARK: 预览

    func requestPreview(now: Bool = false) {
        debounce?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.renderPreview() }
        debounce = work
        DispatchQueue.main.asyncAfter(deadline: .now() + (now ? 0 : 0.2), execute: work)
    }

    private func renderPreview() {
        guard let pg = page else { return }
        previewProc?.terminate()
        token += 1
        let my = token
        let plan = tmp.appendingPathComponent("plan-\(my).json")
        let out = tmp.appendingPathComponent("preview-\(my).jpg")
        writePlan([pg], to: plan)
        let scale = min(1.0, 1600.0 / Double(max(pg.info.width, pg.info.height)))
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = [vellumPy, "--plan", plan.path, "-o", out.path, "--no-ocr",
                       "--scale", String(format: "%.3f", scale), "--quality", "88"]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { proc in
            DispatchQueue.main.async {
                guard my == self.token else { return }
                self.rendering = false
                if proc.terminationStatus == 0, let img = NSImage(contentsOf: out) { pg.preview = img }
            }
        }
        rendering = true
        do { try p.run(); previewProc = p } catch { rendering = false }
    }

    func applyToAll() {
        guard let cur = page else { return }
        for p in pages where p !== cur {
            p.mode = cur.mode; p.ink = cur.ink; p.book = cur.book; p.preview = nil
        }
        status = "已将模式、墨色和书页设置应用到全部 \(pages.count) 页"
    }

    // MARK: 导出

    private var firstName: String {
        URL(fileURLWithPath: pages.first?.info.file ?? "Vellum").deletingPathExtension().lastPathComponent
    }
    private var firstDir: URL {
        URL(fileURLWithPath: pages.first?.info.file ?? NSHomeDirectory()).deletingLastPathComponent()
    }

    private func savePanel(title: String, name: String, types: [UTType], then: @escaping (URL) -> Void) {
        let panel = NSSavePanel()
        panel.title = title
        panel.prompt = "导出"
        panel.nameFieldLabel = "名称："
        panel.nameFieldStringValue = name
        panel.directoryURL = firstDir
        panel.allowedContentTypes = types
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        let done: (NSApplication.ModalResponse) -> Void = { r in if r == .OK, let u = panel.url { then(u) } }
        if let win = NSApp.keyWindow { panel.beginSheetModal(for: win, completionHandler: done) }
        else { done(panel.runModal()) }
    }

    func exportPDF() {
        guard !pages.isEmpty, !busy else { return }
        savePanel(title: "导出 PDF", name: firstName + "_scan.pdf", types: [.pdf]) { url in
            var args = ["-o", url.path, "--open", "--notify"]
            if !self.ocr { args.append("--no-ocr") }
            self.runExport(args, working: "正在导出 PDF…", quitOnSuccess: true)
        }
    }

    func exportImages() {
        guard !pages.isEmpty, !busy else { return }
        savePanel(title: "导出图片", name: firstName + "_scan.png", types: [.png, .jpeg]) { url in
            self.runExport(["-o", url.path, "--each", "--open", "--notify"], working: "正在导出图片…", quitOnSuccess: true)
        }
    }

    func copyText() {
        guard !pages.isEmpty, !busy else { return }
        runExport(["--text-only", "--copy"], working: "正在识别文字…", quitOnSuccess: false)
    }

    private func runExport(_ extra: [String], working: String, quitOnSuccess: Bool) {
        busy = true
        status = working
        let plan = tmp.appendingPathComponent("plan-export-\(UUID().uuidString).json")
        writePlan(pages, to: plan)
        DispatchQueue.global(qos: .userInitiated).async {
            let (code, _, err) = Model.run([vellumPy, "--plan", plan.path] + extra)
            let log = String(data: err, encoding: .utf8) ?? ""
            DispatchQueue.main.async {
                self.busy = false
                if code != 0 {
                    self.status = "操作未完成：" + (log.split(separator: "\n").last.map(String.init) ?? "")
                } else if quitOnSuccess {
                    self.cleanup()
                    NSApp.terminate(nil)
                } else {
                    let n = log.split(separator: "\n").first { $0.contains("剪贴板") }
                        .flatMap { $0.split(separator: "·").last }.map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
                    self.status = n.isEmpty ? "未识别到文字" : "已拷贝到剪贴板 · \(n)"
                }
            }
        }
    }

    func cleanup() { try? FileManager.default.removeItem(at: tmp) }

    private func writePlan(_ ps: [PageState], to url: URL) {
        let data = try? JSONSerialization.data(withJSONObject: ["pages": ps.map(\.plan)])
        try? data?.write(to: url)
    }

    static func run(_ args: [String]) -> (Int32, Data, Data) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = args
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        do { try p.run() } catch { return (-1, Data(), Data("\(error)".utf8)) }
        let o = out.fileHandleForReading.readDataToEndOfFile()
        let e = err.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return (p.terminationStatus, o, e)
    }
}

// MARK: - 几何

func fitRect(aspect: CGFloat, in size: CGSize, inset: CGFloat) -> CGRect {
    let w = max(size.width - inset * 2, 10), h = max(size.height - inset * 2, 10)
    let (fw, fh) = w / h > aspect ? (h * aspect, h) : (w, w / aspect)
    return CGRect(x: (size.width - fw) / 2, y: (size.height - fh) / 2, width: fw, height: fh)
}

extension CGRect {
    func point(_ u: CGPoint) -> CGPoint { CGPoint(x: minX + u.x * width, y: minY + u.y * height) }
    func unit(_ p: CGPoint) -> CGPoint {
        CGPoint(x: min(max((p.x - minX) / width, 0), 1), y: min(max((p.y - minY) / height, 0), 1))
    }
}

// MARK: - 通用控件

final class HoverFlag: ObservableObject {
    @Published var on = false
}

/// 工具栏图标按钮：44pt 热区；悬停时出现圆形高亮，并在上方弹出名称
struct ToolButton: View {
    let symbol: String
    let title: String
    var detail: String? = nil
    var active = false
    let action: () -> Void
    @StateObject private var hover = HoverFlag()
    @Environment(\.isEnabled) private var enabled

    var body: some View {
        Button(action: action) {
            icon(symbol, 20)
                .foregroundStyle(active ? Color.white : Color.primary)
                .frame(width: 44, height: 44)
                .background(
                    Circle().fill(active ? accent : Color.primary.opacity(hover.on && enabled ? 0.08 : 0))
                )
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.35)
        .onHover { h in withAnimation(.easeOut(duration: 0.12)) { hover.on = h } }
        .overlay(alignment: .top) {
            if hover.on {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .fixedSize()
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .glassEffect(.regular, in: Capsule())
                    .offset(y: -40)
                    .transition(.opacity.combined(with: .offset(y: 4)))
                    .allowsHitTesting(false)
            }
        }
        .help(detail ?? title)
        .accessibilityLabel(Text(title))
    }
}

// MARK: - 视图

final class DragState: ObservableObject {
    @Published var index: Int?
}

struct CornerEditor: View {
    @ObservedObject var page: PageState
    let onCommit: () -> Void
    @StateObject private var drag = DragState()

    var body: some View {
        GeometryReader { geo in
            let r = fitRect(aspect: page.aspect, in: geo.size, inset: 44)
            let pts = page.corners.map { r.point($0) }
            ZStack {
                Image(nsImage: page.image)
                    .resizable().interpolation(.high)
                    .frame(width: r.width, height: r.height)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .shadow(color: .black.opacity(0.18), radius: 24, y: 10)
                    .position(x: r.midX, y: r.midY)
                Path { p in p.addRect(r); p.addLines(pts); p.closeSubpath() }
                    .fill(Color.black.opacity(0.45), style: FillStyle(eoFill: true))
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous).path(in: r))
                    .allowsHitTesting(false)
                Path { p in p.addLines(pts); p.closeSubpath() }
                    .fill(accent.opacity(0.06))
                    .allowsHitTesting(false)
                Path { p in p.addLines(pts); p.closeSubpath() }
                    .stroke(accent, style: StrokeStyle(lineWidth: 2.5, lineJoin: .round))
                    .shadow(color: accent.opacity(0.5), radius: 6)
                    .allowsHitTesting(false)
                ForEach(0..<4, id: \.self) { i in
                    ZStack {
                        Circle().fill(.white).frame(width: 24, height: 24)
                        Circle().strokeBorder(accent, lineWidth: 4).frame(width: 24, height: 24)
                    }
                    .shadow(color: .black.opacity(0.3), radius: 4, y: 2)
                    .scaleEffect(drag.index == i ? 1.3 : 1)
                    .animation(.spring(duration: 0.2), value: drag.index)
                    .frame(width: 56, height: 56)
                    .contentShape(Circle())
                    .position(pts[i])
                    .gesture(
                        DragGesture(minimumDistance: 0, coordinateSpace: .named("canvas"))
                            .onChanged { v in drag.index = i; page.corners[i] = r.unit(v.location) }
                            .onEnded { _ in drag.index = nil; onCommit() }
                    )
                    .onHover { inside in if inside { NSCursor.openHand.push() } else { NSCursor.pop() } }
                }
                if let i = drag.index {
                    Loupe(image: page.image, rect: r, point: pts[i])
                        .position(x: pts[i].x < geo.size.width / 2 ? geo.size.width - 120 : 120, y: 120)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .coordinateSpace(name: "canvas")
        }
    }
}

struct Loupe: View {
    let image: NSImage
    let rect: CGRect
    let point: CGPoint
    let zoom: CGFloat = 4
    let size: CGFloat = 168

    var body: some View {
        ZStack {
            Image(nsImage: image)
                .resizable().interpolation(.high)
                .frame(width: rect.width * zoom, height: rect.height * zoom)
                .offset(x: size / 2 - (point.x - rect.minX) * zoom, y: size / 2 - (point.y - rect.minY) * zoom)
                .frame(width: size, height: size, alignment: .topLeading)
                .clipShape(Circle())
            Path { p in
                p.move(to: CGPoint(x: size / 2, y: size / 2 - 16)); p.addLine(to: CGPoint(x: size / 2, y: size / 2 + 16))
                p.move(to: CGPoint(x: size / 2 - 16, y: size / 2)); p.addLine(to: CGPoint(x: size / 2 + 16, y: size / 2))
            }
            .stroke(accent, lineWidth: 2)
        }
        .frame(width: size, height: size)
        .padding(8)
        .glassEffect(.regular, in: Circle())
        .allowsHitTesting(false)
    }
}

struct PreviewPane: View {
    @ObservedObject var page: PageState
    let rendering: Bool

    var body: some View {
        GeometryReader { geo in
            ZStack {
                if let img = page.preview {
                    let r = fitRect(aspect: img.size.width / max(img.size.height, 1), in: geo.size, inset: 44)
                    Image(nsImage: img)
                        .resizable().interpolation(.high)
                        .frame(width: r.width, height: r.height)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .shadow(color: .black.opacity(0.18), radius: 24, y: 10)
                        .position(x: r.midX, y: r.midY)
                        .opacity(rendering ? 0.5 : 1)
                        .animation(.easeOut(duration: 0.15), value: rendering)
                }
                if rendering || page.preview == nil {
                    ProgressView().controlSize(.regular)
                        .frame(width: 56, height: 56)
                        .glassEffect(.regular, in: Circle())
                }
            }
            .overlay(alignment: .top) {
                Text("预览")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12).padding(.vertical, 5)
                    .glassEffect(.regular, in: Capsule())
                    .padding(.top, 8)
            }
        }
    }
}

struct Thumb: View {
    @ObservedObject var m: Model
    @ObservedObject var p: PageState
    let number: Int

    var body: some View {
        let selected = m.page?.id == p.id
        ZStack(alignment: .topTrailing) {
            Image(nsImage: p.preview ?? p.image)
                .resizable().interpolation(.high)
                .aspectRatio(p.preview.map { $0.size.width / max($0.size.height, 1) } ?? p.aspect, contentMode: .fit)
                .frame(width: 92)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(selected ? accent : Color.white.opacity(0.4), lineWidth: selected ? 3 : 1)
                )
                .shadow(color: .black.opacity(0.15), radius: 6, y: 3)
                .opacity(m.dragging == p.id ? 0.35 : 1)
            if m.hovering == p.id && m.pages.count > 1 {
                Button { m.delete(p) } label: {
                    icon("trash", 14).foregroundStyle(.white)
                        .frame(width: 28, height: 28)
                        .background(Circle().fill(Color.red.opacity(0.9)))
                }
                .buttonStyle(.plain)
                .help("删除这一页")
                .offset(x: 8, y: -8)
                .transition(.scale.combined(with: .opacity))
            }
        }
        .overlay(alignment: .bottomLeading) {
            HStack(spacing: 4) {
                Text("\(number)").font(.system(size: 11, weight: .bold)).monospacedDigit()
                if p.detected == nil { Circle().fill(Color.orange).frame(width: 6, height: 6) }
            }
            .padding(.horizontal, 8).padding(.vertical, 3)
            .glassEffect(.regular, in: Capsule())
            .offset(x: -6, y: 6)
        }
        .onHover { h in withAnimation(.easeOut(duration: 0.12)) { m.hovering = h ? p.id : (m.hovering == p.id ? nil : m.hovering) } }
        .onTapGesture { m.select(p.id) }
        .onDrag {
            m.dragging = p.id
            return NSItemProvider(object: p.id.uuidString as NSString)
        }
        .onDrop(of: [.text], delegate: ReorderDelegate(target: p.id, m: m))
    }
}

struct ReorderDelegate: DropDelegate {
    let target: UUID
    let m: Model
    func dropEntered(info: DropInfo) {
        if let d = m.dragging { m.move(d, before: target) }
    }
    func dropUpdated(info: DropInfo) -> DropProposal? { DropProposal(operation: .move) }
    func performDrop(info: DropInfo) -> Bool {
        m.dragging = nil
        m.requestPreview(now: true)
        return true
    }
}

struct Sidebar: View {
    @ObservedObject var m: Model

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("页面").font(.system(size: 13, weight: .semibold))
                Spacer()
                Text("\(m.pages.count)").font(.system(size: 13, weight: .semibold)).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 18).padding(.top, 16).padding(.bottom, 8)
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 18) {
                    ForEach(Array(m.pages.enumerated()), id: \.element.id) { i, p in
                        Thumb(m: m, p: p, number: i + 1)
                    }
                }
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity)
            }
            Button(action: m.addPages) {
                HStack(spacing: 6) { icon("page-add", 18); Text("添加页面").font(.system(size: 13, weight: .medium)) }
                    .frame(maxWidth: .infinity).frame(height: 40)
                    .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .glassEffect(.regular.interactive(), in: Capsule())
            .padding(12)
            .keyboardShortcut("o", modifiers: .command)
            .help("添加照片（⌘O），也可以直接把照片拖进窗口")
        }
        .frame(width: 136)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
    }
}

struct Toolbar: View {
    @ObservedObject var m: Model
    @ObservedObject var page: PageState

    var body: some View {
        GlassEffectContainer(spacing: 14) {
            HStack(spacing: 14) {
                // 模式
                Menu {
                    ForEach(modes, id: \.id) { d in
                        Button {
                            page.mode = d.id
                        } label: {
                            Text(d.id == "auto" ? "自动（当前识别为\(modeDef(page.info.mode).name)）" : d.name)
                            Text(d.hint)
                        }
                    }
                } label: {
                    HStack(spacing: 8) {
                        icon(modeDef(page.effectiveMode).icon, 20)
                        Text(page.mode == "auto" ? "自动 · \(modeDef(page.info.mode).name)" : modeDef(page.mode).name)
                            .font(.system(size: 14, weight: .semibold))
                        icon("chevron-down", 14).foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 16).frame(height: 44)
                    .contentShape(Capsule())
                }
                .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
                .fixedSize()
                .glassEffect(.regular.interactive(), in: Capsule())
                .help("处理模式")
                .accessibilityLabel(Text("处理模式"))

                // 墨色
                HStack(spacing: 2) {
                    ForEach(inks, id: \.0) { k in
                        Button { page.ink = k.0 } label: {
                            Text(k.1).font(.system(size: 13, weight: page.ink == k.0 ? .semibold : .regular))
                                .foregroundStyle(page.ink == k.0 ? .white : .primary)
                                .frame(width: 52, height: 36)
                                .background(Capsule().fill(page.ink == k.0 ? accent : .clear))
                                .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(4)
                .glassEffect(.regular, in: Capsule())
                .opacity(page.inkApplies ? 1 : 0.4)
                .disabled(!page.inkApplies)
                .help("墨色浓度：笔迹太淡时选“浓郁”或“加粗”")

                // 页面操作：一整条玻璃，按钮在里面各自高亮（相邻玻璃不会互相融合变形）
                HStack(spacing: 4) {
                    ToolButton(symbol: "book", title: "书页展平",
                               detail: "校正书本弯曲，还原书脊附近被压窄的文字", active: page.book) {
                        page.book.toggle(); m.requestPreview(now: true)
                    }
                    Divider().frame(height: 22).padding(.horizontal, 2)
                    ToolButton(symbol: "rotate-left", title: "向左旋转") {
                        page.rotate = (page.rotate + 270) % 360; m.requestPreview(now: true)
                    }
                    ToolButton(symbol: "rotate-right", title: "向右旋转") {
                        page.rotate = (page.rotate + 90) % 360; m.requestPreview(now: true)
                    }
                    Divider().frame(height: 22).padding(.horizontal, 2)
                    ToolButton(symbol: "corners-auto", title: "自动边缘", detail: "恢复自动识别的纸张边缘") {
                        page.corners = page.detected ?? fullFrame; m.requestPreview(now: true)
                    }
                    .disabled(page.detected == nil)
                    ToolButton(symbol: "corners-full", title: "整张照片", detail: "不裁切，使用整张照片") {
                        page.corners = fullFrame; m.requestPreview(now: true)
                    }
                    if m.pages.count > 1 {
                        Divider().frame(height: 22).padding(.horizontal, 2)
                        ToolButton(symbol: "apply-all", title: "应用到全部",
                                   detail: "将模式、墨色和书页设置应用到全部页面") { m.applyToAll() }
                    }
                }
                .padding(4)
                .glassEffect(.regular, in: Capsule())

                Spacer(minLength: 16)

                Text(m.status)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 260, alignment: .trailing)

                // OCR
                Button { m.ocr.toggle() } label: {
                    HStack(spacing: 6) {
                        icon("ocr", 18)
                        Text("文字识别").font(.system(size: 13, weight: .medium))
                        icon("check", 14).opacity(m.ocr ? 1 : 0)
                    }
                    .foregroundStyle(m.ocr ? accent : .secondary)
                    .padding(.horizontal, 14).frame(height: 44)
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .glassEffect(.regular.interactive(), in: Capsule())
                .help("导出的 PDF 可搜索、可复制文字（本机识别，不上传）")

                // 导出
                HStack(spacing: 0) {
                    Button(action: m.exportPDF) {
                        HStack(spacing: 8) {
                            if m.busy { ProgressView().controlSize(.small).tint(.white) }
                            else { icon("export-pdf", 20) }
                            Text("导出 PDF").font(.system(size: 14, weight: .semibold))
                        }
                        .foregroundStyle(.white)
                        .padding(.leading, 18).padding(.trailing, 12).frame(height: 44)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.defaultAction)
                    Rectangle().fill(.white.opacity(0.35)).frame(width: 1, height: 22)
                    Menu {
                        Button { m.exportPDF() } label: { Text("导出 PDF…") }
                        Button { m.exportImages() } label: { Text("导出图片…") }
                        Divider()
                        Button { m.copyText() } label: { Text("识别文字并拷贝") }
                    } label: {
                        icon("chevron-down", 16).foregroundStyle(.white)
                            .frame(width: 40, height: 44).contentShape(Rectangle())
                    }
                    .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden).fixedSize()
                    .accessibilityLabel(Text("更多导出方式"))
                }
                .glassEffect(.regular.tint(accent).interactive(), in: Capsule())
                .disabled(m.busy || m.pages.isEmpty)
            }
        }
        .onChange(of: page.mode) { m.requestPreview(now: true) }
        .onChange(of: page.ink) { m.requestPreview(now: true) }
    }
}

struct ContentView: View {
    @ObservedObject var m: Model

    var body: some View {
        ZStack(alignment: .bottom) {
            LinearGradient(colors: [Color(nsColor: .windowBackgroundColor),
                                    accent.opacity(0.07)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            if let page = m.page {
                HStack(spacing: 0) {
                    Sidebar(m: m)
                        .padding(.leading, 14).padding(.top, 40).padding(.bottom, 88)
                    CornerEditor(page: page) { m.requestPreview() }
                    PreviewPane(page: page, rendering: m.rendering)
                }
                .padding(.bottom, 76)
                Toolbar(m: m, page: page)
                    .padding(.horizontal, 18)
                    .padding(.bottom, 16)
            } else {
                VStack(spacing: 14) {
                    if m.busy { ProgressView() } else { icon("page-add", 40).foregroundStyle(accent) }
                    Text(m.status).font(.system(size: 15)).foregroundStyle(.secondary)
                    if !m.busy {
                        Button("添加页面…", action: m.addPages).buttonStyle(.glassProminent).controlSize(.large)
                    }
                }
                .padding(36)
                .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 32, style: .continuous))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if m.dropTargeted {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .strokeBorder(accent, style: StrokeStyle(lineWidth: 3, dash: [10, 8]))
                    .padding(10)
                    .allowsHitTesting(false)
            }
        }
        .frame(minWidth: 1180, minHeight: 720)
        .onDrop(of: [.fileURL], isTargeted: Binding(get: { m.dropTargeted }, set: { m.dropTargeted = $0 })) { providers in
            var urls: [String] = []
            let group = DispatchGroup()
            for p in providers {
                group.enter()
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    if let u = url { DispatchQueue.main.async { urls.append(u.path) } }
                    group.leave()
                }
            }
            group.notify(queue: .main) { m.load(urls.sorted()) }
            return true
        }
        .background(
            Group {
                Button("") { m.step(-1) }.keyboardShortcut(.leftArrow, modifiers: .command)
                Button("") { m.step(1) }.keyboardShortcut(.rightArrow, modifiers: .command)
                Button("") { if let p = m.page, m.pages.count > 1 { m.delete(p) } }.keyboardShortcut(.delete, modifiers: .command)
                Button("") { m.cleanup(); NSApp.terminate(nil) }.keyboardShortcut(.cancelAction)
            }
            .opacity(0)
        )
    }
}

// MARK: - 启动

/// 点按通知：打开通知里带的文件（导出的 PDF / 图片）
func openFromNotification(_ response: UNNotificationResponse) {
    if let path = response.notification.request.content.userInfo["open"] as? String {
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }
}

/// 通知模式：`Vellum --notify 标题 正文 [--open 路径]`。由 vellum.py 调用，以 Vellum 的名义和图标发系统通知，发完即退出。
final class Notifier: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    let title: String, body: String, open: String?
    init(title: String, body: String, open: String?) { self.title = title; self.body = body; self.open = open }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let center = UNUserNotificationCenter.current()
        center.delegate = self
        center.requestAuthorization(options: [.alert, .sound]) { granted, _ in
            guard granted else { DispatchQueue.main.async { NSApp.terminate(nil) }; return }
            let content = UNMutableNotificationContent()
            content.title = self.title
            content.body = self.body
            content.sound = .default
            if let o = self.open { content.userInfo = ["open": o] }
            center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)) { _ in
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { NSApp.terminate(nil) }
            }
        }
        // 兜底：最多停留 90 秒（首次会等用户在系统弹窗里允许通知）
        DispatchQueue.main.asyncAfter(deadline: .now() + 90) { NSApp.terminate(nil) }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound])
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                withCompletionHandler completionHandler: @escaping () -> Void) {
        openFromNotification(response)
        completionHandler()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    let model = Model()
    let files: [String]
    var window: NSWindow!
    var handledNotification = false

    init(files: [String]) { self.files = files }

    func applicationWillFinishLaunching(_ notification: Notification) {
        UNUserNotificationCenter.current().delegate = self
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        if files.isEmpty {
            // 可能是点按通知把 App 拉起来的：稍等一下，确认不是再开空窗口
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                if !self.handledNotification { self.showWindow() }
            }
        } else {
            showWindow()
        }
    }

    func showWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1440, height: 920),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = "Vellum"
        window.subtitle = files.count == 1 ? URL(fileURLWithPath: files[0]).lastPathComponent : "扫描并调整"
        window.titlebarAppearsTransparent = true
        window.contentView = NSHostingView(rootView: ContentView(m: model))
        window.setFrameAutosaveName("vellum.review")
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        if files.isEmpty { model.refreshStatus() } else { model.load(files) }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                withCompletionHandler completionHandler: @escaping () -> Void) {
        handledNotification = true
        openFromNotification(response)
        completionHandler()
        if window == nil { NSApp.terminate(nil) }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { model.cleanup() }
}

let args = Array(CommandLine.arguments.dropFirst()).filter { !$0.hasPrefix("-psn") }
let app = NSApplication.shared

if let i = args.firstIndex(of: "--notify"), args.count > i + 2 {
    let open = args.firstIndex(of: "--open").flatMap { args.count > $0 + 1 ? args[$0 + 1] : nil }
    app.setActivationPolicy(.accessory)  // 不出现在程序坞
    let notifier = Notifier(title: args[i + 1], body: args[i + 2], open: open)
    app.delegate = notifier
    app.run()
    exit(0)
}

let files = args
app.setActivationPolicy(.regular)
let mainMenu = NSMenu()
let appItem = NSMenuItem()
mainMenu.addItem(appItem)
let appMenu = NSMenu()
appMenu.addItem(withTitle: "关闭窗口", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
appMenu.addItem(withTitle: "退出 Vellum", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
appItem.submenu = appMenu
app.mainMenu = mainMenu
let delegate = AppDelegate(files: files)
app.delegate = delegate
app.run()
