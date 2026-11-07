// Vellum OCR · macOS Vision 文字识别引擎（离线，系统自带模型）
//
//   vellum-ocr [--langs zh-Hans,en-US] [--fast] 图片...      → stdout 输出 JSON
//   vellum-ocr --pdf-text 文件.pdf                          → 用 PDFKit 读出 PDF 的文字层（自检用）
//   vellum-ocr --list-langs                                 → 列出支持的识别语言
//
// JSON：[{file, width, height, lines: [{text, conf, quad: [[x,y]×4]}]}]
// quad 顺序：左下、右下、右上、左上；坐标归一化到 0–1，原点在左下（和 PDF 坐标系一致）。
//
// 编译：swiftc -O -swift-version 5 ocr.swift -o vellum-ocr

import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import Vision

struct Line: Codable {
    let text: String
    let conf: Float
    let quad: [[Double]]
}

struct Page: Codable {
    let file: String
    let width: Int
    let height: Int
    let lines: [Line]
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

var langs = ["zh-Hans", "zh-Hant", "en-US"]
var fast = false
var files: [String] = []
var args = Array(CommandLine.arguments.dropFirst())

while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--langs":
        guard !args.isEmpty else { fail("--langs 需要参数") }
        langs = args.removeFirst().split(separator: ",").map(String.init)
    case "--fast":
        fast = true
    case "--list-langs":
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        print((try? req.supportedRecognitionLanguages())?.joined(separator: ", ") ?? "")
        exit(0)
    case "--pdf-text":
        guard !args.isEmpty, let doc = PDFDocument(url: URL(fileURLWithPath: args.removeFirst())) else {
            fail("读不了 PDF")
        }
        print(doc.string ?? "")
        exit(0)
    default:
        files.append(a)
    }
}
if files.isEmpty { fail("用法：vellum-ocr [--langs zh-Hans,en-US] [--fast] 图片...") }

var pages: [Page] = []
for f in files {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: f) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail("读不了图片：\(f)") }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = fast ? .fast : .accurate
    req.usesLanguageCorrection = true
    req.recognitionLanguages = langs
    let handler = VNImageRequestHandler(cgImage: img, options: [:])
    do { try handler.perform([req]) } catch { fail("识别失败：\(error)") }
    var lines: [Line] = []
    for obs in req.results ?? [] {
        guard let cand = obs.topCandidates(1).first else { continue }
        let pts = [obs.bottomLeft, obs.bottomRight, obs.topRight, obs.topLeft]
        lines.append(Line(text: cand.string, conf: cand.confidence,
                          quad: pts.map { [Double($0.x), Double($0.y)] }))
    }
    pages.append(Page(file: f, width: img.width, height: img.height, lines: lines))
}
let data = try! JSONEncoder().encode(pages)
FileHandle.standardOutput.write(data)
