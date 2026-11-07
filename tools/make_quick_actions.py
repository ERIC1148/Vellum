#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成访达右键“快速操作”（Automator 服务）：
  review —— 扫描并调整（首选）：打开检查窗口，拖四角、切模式和墨色、实时预览，导出时选位置和文件名
  scan   —— 扫描并导出 PDF：自动模式一步到位，找不到边缘时转到检查窗口


用法：make_quick_actions.py <Services 目录> <vellum 命令路径> <review 名字> <scan 名字>
结构参照 macOS 自带 Automator 生成的“运行 Shell 脚本”工作流。
"""
import os
import plistlib
import shutil
import sys
import uuid

LOG = '"$HOME/Library/Logs/vellum.log"'

def command(bin_path, kind):
    head = f'echo "=== $(date "+%F %T") ===" >> {LOG}\n'
    if kind == "review":
        return head + f'"{bin_path}" --review "$@" >> {LOG} 2>&1\n'
    if kind == "text":
        # 识别出的文字只进剪贴板，不写进日志
        return head + f'"{bin_path}" --text-only --copy --notify "$@" 2>> {LOG} > /dev/null\n'
    # 有页面没找到边缘时自动转到检查窗口
    return head + f'"{bin_path}" --review-if-unsure --notify --open "$@" >> {LOG} 2>&1\n'


def arg(i, name, default):
    return {"default value": default, "name": name, "required": "0", "type": "0", "uuid": str(i)}


def wflow(cmd):
    return {
        "AMApplicationBuild": "534",
        "AMApplicationVersion": "2.10",
        "AMDocumentVersion": "2",
        "actions": [{
            "action": {
                "AMAccepts": {"Container": "List", "Optional": True, "Types": ["com.apple.cocoa.string"]},
                "AMActionVersion": "2.0.3",
                "AMApplication": ["Automator"],
                "AMParameterProperties": {k: {} for k in (
                    "COMMAND_STRING", "CheckedForUserDefaultShell", "inputMethod", "shell", "source")},
                "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
                "ActionBundlePath": "/System/Library/Automator/Run Shell Script.action",
                "ActionName": "Run Shell Script",
                "ActionParameters": {
                    "COMMAND_STRING": cmd,
                    "CheckedForUserDefaultShell": True,
                    "inputMethod": 1,  # 作为参数传入（"$@"）
                    "shell": "/bin/zsh",
                    "source": "",
                },
                "BundleIdentifier": "com.apple.RunShellScript",
                "CFBundleVersion": "2.0.3",
                "CanShowSelectedItemsWhenRun": False,
                "CanShowWhenRun": True,
                "Category": ["AMCategoryUtilities"],
                "Class Name": "RunShellScriptAction",
                "InputUUID": str(uuid.uuid4()).upper(),
                "Keywords": ["Shell", "Script", "Command", "Run", "Unix"],
                "OutputUUID": str(uuid.uuid4()).upper(),
                "UUID": str(uuid.uuid4()).upper(),
                "UnlocalizedApplications": ["Automator"],
                "arguments": {
                    "0": arg(0, "inputMethod", 0),
                    "1": arg(1, "CheckedForUserDefaultShell", False),
                    "2": arg(2, "source", ""),
                    "3": arg(3, "COMMAND_STRING", ""),
                    "4": arg(4, "shell", "/bin/sh"),
                },
                "conversionLabel": 0,
                "isViewVisible": 1,
                "location": "309.000000:305.000000",
                "nibPath": "/System/Library/Automator/Run Shell Script.action/Contents/Resources/Base.lproj/main.nib",
            },
            "isViewVisible": 1,
        }],
        "connectors": {},
        "workflowMetaData": {
            "applicationBundleIDsByPath": {},
            "applicationPaths": [],
            "inputTypeIdentifier": "com.apple.Automator.fileSystemObject",
            "outputTypeIdentifier": "com.apple.Automator.nothing",
            "presentationMode": 15,
            "processesInput": False,
            "serviceInputTypeIdentifier": "com.apple.Automator.fileSystemObject",
            "serviceOutputTypeIdentifier": "com.apple.Automator.nothing",
            "serviceProcessesInput": False,
            "systemImageName": "NSActionTemplate",
            "useAutomaticInputType": False,
            "workflowTypeIdentifier": "com.apple.Automator.servicesMenu",
        },
    }


def info_plist(name):
    return {
        "NSServices": [{
            "NSBackgroundColorName": "background",
            "NSIconName": "NSActionTemplate",
            "NSMenuItem": {"default": name},
            "NSMessage": "runWorkflowAsService",
            "NSSendFileTypes": ["public.image", "public.folder"],
        }]
    }


def make(services, name, cmd):
    root = os.path.join(services, name + ".workflow")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "Contents"))
    with open(os.path.join(root, "Contents", "Info.plist"), "wb") as f:
        plistlib.dump(info_plist(name), f)
    with open(os.path.join(root, "Contents", "document.wflow"), "wb") as f:
        plistlib.dump(wflow(cmd), f)
    print(f"    {root}")


def main():
    services, bin_path, name_review, name_scan = sys.argv[1:5]
    make(services, name_review, command(bin_path, "review"))
    make(services, name_scan, command(bin_path, "scan"))


if __name__ == "__main__":
    main()
